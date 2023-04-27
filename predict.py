import colorsys
import copy
import time
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torch import nn
from nets.MA_UNet import MA_Unet_T, MA_Unet_S, MA_Unet_B, MA_Unet_L
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_metrics import compute_mIoU, show_results

class Unet_tools(object):
    def __init__(self, args, **kwargs):
        self.model_type, self.model_path, self.num_classes, self.input_shape, self.blend, self.cuda, self.use_pos_embed, self.model_name, self.backbone_name\
            = args.model_type, args.model_path, args.num_classes, args.input_shape, args.blend, args.cuda, args.use_pos_embed, args.model_name, args.backbone_name
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.colors = [x for sublist in self.colors for x in sublist]
        self.generate()

    def generate(self, onnx=False):
        if self.model_type is not None:
            if self.model_type == 'tiny':
                self.net = MA_Unet_T(num_classes=self.num_classes, input_size=self.input_shape[0], use_pos_embed=self.use_pos_embed)
            elif self.model_type == 'small':
                self.net = MA_Unet_S(num_classes=self.num_classes, input_size=self.input_shape[0], use_pos_embed=self.use_pos_embed)
            elif self.model_type == 'base':
                self.net = MA_Unet_B(num_classes=self.num_classes, input_size=self.input_shape[0], use_pos_embed=self.use_pos_embed)
            else:
                self.net = MA_Unet_L(num_classes=self.num_classes, input_size=self.input_shape[0], use_pos_embed=self.use_pos_embed)

        else:
            from segmentation_models_pytorch import Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN
            print('testing with %s, backbone used with %s' % (self.model_name, self.backbone_name))
            model_n = eval(self.model_name)
            self.net = model_n(
                encoder_name=self.backbone_name,
                encoder_weights=None,
                in_channels=3,
                classes=self.num_classes,
            )

        device = torch.device('cuda' if self.cuda else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                # self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image):
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
        image = Image.fromarray(np.uint8(pr))
        image.putpalette(self.colors)
        image = image.convert('RGB')
        if self.blend:
            image = Image.blend(old_img,image,0.7)
        
        return image

    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_miou_png(self, image):
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='predict', type=str, help='[predict, validate, pth2onnx]')
    parser.add_argument("--cuda", default=True, type=bool, help='use cuda')
    parser.add_argument("--num_classes", default=21, type=int, help='Actual category quantity plus 1 (background)')
    parser.add_argument("--blend", default=False, type=bool, help='blend result image and ori image')
    parser.add_argument("--dataset", default='VOCdevkit', type=str, help='Dataset path')
    parser.add_argument("--device_id", default='0', type=str, help="Number of cuda want to use.")
    parser.add_argument("--input_shape", default=[640, 640], type=list, help='Image size for CNN after Data Augmentation')
    parser.add_argument("--model_path", default='logs/best_weights.pth', type=str, help='model path')

    #### predict for MA-UNet
    parser.add_argument("--model_type", default='base', type=str, help=" ['tiny', 'small', 'base', 'large', None] ")
    parser.add_argument("--use_pos_embed", default=False, type=bool, help='If true, input size must equal the input size during training')

    #### predict for other model, setting same as training
    ### if u want to test other models, please set model_type=None in line 196, otherwise test MA-UNet
    ### https://github.com/qubvel/segmentation_models.pytorch
    parser.add_argument("--model_name", default='UnetPlusPlus', type=str, help="[Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]")
    parser.add_argument("--backbone_name", default='vgg16_bn', type=str, help='More than 100 backbone are available, see github for details')

    ####prediction mode, predict image
    parser.add_argument("--test_img_path", default='test/', type=str)
    parser.add_argument("--result_save_path", default='save/', type=str)

    ####validation mode, get mean IOU for testsets
    parser.add_argument("--name_classes", default=["_background_", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"], type=list)
    parser.add_argument("--validation_result_path", default='miou_save', type=str, help='validation images in val.txt')

    ####pth2onnx mode, convert pytorch model to onnx.  not applicable with cuda 10.2
    parser.add_argument("--simplify_onnx", default=True, type=bool)
    parser.add_argument("--onnx_save_path", default='model.onnx', type=str, help='onnx model save path')

    args = parser.parse_args()
    unet = Unet_tools(args)

    
    if args.mode == 'predict':
        path = args.test_img_path
        savep = args.result_save_path
        if not os.path.exists(savep):
            os.makedirs(savep)
        start = time.time()
        for i in os.listdir(path):
            img = os.path.join(path, i)
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image)
                r_image.save(os.path.join(savep, i))
        end = time.time()
        print('finish, time costed:',  end - start)

    if args.mode == 'validate':
        num_classes, name_classes, VOCdevkit_path, miou_out_path = args.num_classes, args.name_classes, args.dataset, args.validation_result_path
        image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
        gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
        pred_dir = os.path.join(miou_out_path, 'detection-results')

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Get predict result.")
        img_mode = os.listdir(os.path.join(VOCdevkit_path, "VOC2007/JPEGImages"))[0].split('.')[1]
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + "." + img_mode)
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + "." + img_mode))
        print("Get predict result done.")

        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

    if args.mode == "pth2onnx":
        simplify, onnx_save_path = args.simplify_onnx, args.onnx_save_path
        unet.convert_to_onnx(simplify, onnx_save_path)
