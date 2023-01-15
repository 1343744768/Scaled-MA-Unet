import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input
from random import shuffle

class MAEDataset(Dataset):
    def __init__(self, dataset_path, input_shape, train=True, max_h_size=5, max_w_size=5, fill_color=(128, 128, 128), coeffi_num=2):
        super(MAEDataset, self).__init__()
        self.input_shape = input_shape
        h, w = input_shape
        self.num_holes = coeffi_num * min(h//max_h_size, w//max_w_size)
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_color
        self.dataset_path = os.path.join(dataset_path, "VOC2007/JPEGImages")
        img_list = os.listdir(self.dataset_path)
        shuffle(img_list)
        if train:
            self.img_list = img_list[:int(len(img_list)*0.8)]
        else:
            self.img_list = img_list[int(len(img_list) * 0.8):]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_index = self.img_list[index]
        ori_jpg = Image.open(os.path.join(self.dataset_path, img_index))
        ori_jpg, mae_jpg = self.process_data(ori_jpg, self.input_shape)
        return mae_jpg, ori_jpg

    def process_data(self, image, input_shape):
        image = cvtColor(image)
        h, w = input_shape
        iw, ih = image.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        new_image, new_label = self.random_eraser(new_image)
        return new_image, new_label

    def random_eraser(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        mae_jpg = np.copy(img)
        h = img.shape[0]
        w = img.shape[1]
        for _ in range(self.num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(max(0, y - self.max_h_size // 2), 0, h)
            y2 = np.clip(max(0, y + self.max_h_size // 2), 0, h)
            x1 = np.clip(max(0, x - self.max_w_size // 2), 0, w)
            x2 = np.clip(max(0, x + self.max_w_size // 2), 0, w)
            mae_jpg[y1: y2, x1: x2, :] = self.fill_value

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mae_jpg = Image.fromarray(cv2.cvtColor(mae_jpg, cv2.COLOR_BGR2RGB))
        img = np.transpose(preprocess_input(np.array(img, np.float64)), [2, 0, 1])
        mae_jpg = np.transpose(preprocess_input(np.array(mae_jpg, np.float64)), [2, 0, 1])
        return img, mae_jpg


def mae_dataset_collate(batch):
    images = []
    pngs = []
    for img, png in batch:
        images.append(img)
        pngs.append(png)
    images = np.array(images)
    pngs = np.array(pngs)
    return images, pngs

