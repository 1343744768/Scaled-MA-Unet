import os
import cv2
import argparse
import onnxruntime
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from utils.utils import cvtColor, preprocess_input, resize_image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def image_test(image, imagename, args):
    onnx_path, savepath = args.model_path, args.save_path
    colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
              (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
              (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 12)]
    device = torch.cuda.is_available()
    image = cvtColor(image)
    input_size = args.input_shape
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    image_data, nw, nh = resize_image(image, input_size)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'] if device else ['CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
    ort_outs = ort_session.run(None, ort_inputs)
    pr = torch.from_numpy(ort_outs[0])
    pr = F.softmax(pr[0].permute(1, 2, 0), dim=-1).cpu().numpy()
    pr = pr[int((input_size[0] - nh) // 2): int((input_size[0] - nh) // 2 + nh), \
         int((input_size[1] - nw) // 2): int((input_size[1] - nw) // 2 + nw)]
    pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
    pr = pr.argmax(axis=-1)
    seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
    for c in range(args.num_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    image = Image.fromarray(np.uint8(seg_img))
    image.save(os.path.join(savepath, imagename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", default=21, type=int, help='Actual category quantity plus 1 (background)')
    parser.add_argument("--img_path", default='test/', type=str, help='test image path')
    parser.add_argument("--save_path", default='onnx_save/', type=str, help='result path')
    parser.add_argument("--model_path", default='model.onnx', type=str, help='onnx model path')
    parser.add_argument("--input_shape", default=(640, 640), type=tuple, help='Image size for CNN after Data Augmentation, (H, W)')
    parser.add_argument("--device_id", default='0', type=str, help="Number of cuda want to use.")
    args = parser.parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    imgpath = args.img_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for i in os.listdir(imgpath):
        imagename = i
        img = os.path.join(imgpath, i)
        image = Image.open(img)
        image_test(image, imagename, args)
