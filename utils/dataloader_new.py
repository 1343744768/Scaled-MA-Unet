import os
import cv2
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.utils import cvtColor, preprocess_input


class Cutout():
    def __init__(self, max_h_size=20, max_w_size=20, fill_value=255):
        self.num_holes = random.randint(50, 150)
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def __call__(self, img):
        img = np.array(img)
        h = img.shape[0]
        w = img.shape[1]
        for _ in range(self.num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(max(0, y - self.max_h_size // 2), 0, h)
            y2 = np.clip(max(0, y + self.max_h_size // 2), 0, h)
            x1 = np.clip(max(0, x - self.max_w_size // 2), 0, w)
            x2 = np.clip(max(0, x + self.max_w_size // 2), 0, w)
            try:
                img[y1: y2, x1: x2, :] = self.fill_value
            except:
                img[y1: y2, x1: x2] = 0
        img = Image.fromarray(img)

        return img


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path,
                 mosaic=0., cutout=0., flip_rl=0.5, flip_ud=0.5, color_trans=0.5, randn_crop=0.5, randn_affine=0.5,
                 gaussian=0.5, rotate=(-90, 90), scale=(0.8, 1.2), trans=(0.1, 0.1), end_mosaic=False):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.mosaic = mosaic
        self.end_mosaic = end_mosaic
        self.dataset_path = dataset_path
        self.data_aug = transforms.Compose([
            transforms.RandomApply([transforms.RandomAffine(degrees=rotate, translate=trans, scale=scale, fill=0)],
                                   p=randn_affine),
            transforms.RandomApply([transforms.RandomCrop((input_shape))], p=randn_crop),
            transforms.RandomHorizontalFlip(p=flip_rl),
            transforms.RandomVerticalFlip(p=flip_ud),
            transforms.RandomApply([Cutout()], p=cutout),
        ])

        self.data_aug4mosaic = transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(degrees=rotate)], p=randn_affine),
            transforms.RandomHorizontalFlip(p=flip_rl),
            transforms.RandomVerticalFlip(p=flip_ud),
            transforms.RandomApply([Cutout()], p=cutout),
        ])

        self.image_aug = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.8, 1.2])],
                p=color_trans),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(random.choice([5, 7, 9, 11, 13])), sigma=random.uniform(0.1, 2.0))],
                                   p=gaussian)
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        mosaic_prob = random.uniform(0, 1)
        if mosaic_prob <= self.mosaic and self.train and not self.end_mosaic:
            lines = [x.rstrip('\n') for x in random.sample(self.annotation_lines, 3)]
            lines.append(name)
            random.shuffle(lines)
            img, label = self.use_mosaic(lines)

        else:
            img = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + '.jpg'))
            label = Image.open(
                os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + '.png'))
            img, label = self.get_random_data(img, label, self.input_shape, aug=self.train)

        img = np.transpose(preprocess_input(np.array(img, np.float64)), [2, 0, 1])
        label = np.array(label)
        label[label >= self.num_classes] = self.num_classes

        seg_labels_for_dice = np.eye(self.num_classes + 1)[label.reshape([-1]).astype(np.uint8)]
        seg_labels_for_dice = seg_labels_for_dice.reshape(
            (int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return img, label, seg_labels_for_dice

    def get_random_data(self, image, label, input_shape, aug=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        h, w = input_shape
        iw, ih = image.size
        scale = max(w / iw, h / ih)*1.2
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        # augmentation
        if aug:
            seed = random.randint(0, 2147483647)
            self.setup_seed(seed)
            image = self.data_aug(image)
            self.setup_seed(seed)
            label = self.data_aug(label)
            image = self.image_aug(image)

        iw, ih = image.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        new_image = Image.new('RGB', (w, h), (0, 0, 0))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        new_label = Image.new('L', (w, h), 0)
        new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
        # self.show_image(new_image, new_label)

        return new_image, new_label

    def use_mosaic(self, img_list):
        h, w = self.input_shape
        new_ar = w / h
        min_offset_x = 0.3
        min_offset_y = 0.3
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.6
        image_datas = []
        label_datas = []
        index = 0

        place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]
        img_path = os.path.join(self.dataset_path, 'VOC2007/JPEGImages')
        lab_path = os.path.join(self.dataset_path, 'VOC2007/SegmentationClass')

        for im in img_list:
            img = os.path.join(img_path, im + '.jpg')
            lab = os.path.join(lab_path, im + '.png')
            # 打开图片
            image = Image.open(img)
            label = Image.open(lab)

            image = cvtColor(image)
            label = Image.fromarray(np.array(label))

            scale = random.uniform(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            label = label.resize((nw, nh), Image.NEAREST)

            seed = random.randint(0, 2147483647)
            self.setup_seed(seed)
            image = self.data_aug4mosaic(image)
            self.setup_seed(seed)
            label = self.data_aug4mosaic(label)
            image = self.image_aug(image)

            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h), (0, 0, 0))
            new_image.paste(image, (dx, dy))
            new_label = Image.new('L', (w, h), 0)
            new_label.paste(label, (dx, dy))
            image_data = np.array(new_image)
            label_data = np.array(new_label)

            index = index + 1
            image_datas.append(image_data)
            label_datas.append(label_data)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
        cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]

        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_label = np.zeros([h, w])
        new_label[:cuty, :cutx] = label_datas[0][:cuty, :cutx]
        new_label[cuty:, :cutx] = label_datas[1][cuty:, :cutx]
        new_label[cuty:, cutx:] = label_datas[2][cuty:, cutx:]
        new_label[:cuty, cutx:] = label_datas[3][:cuty, cutx:]

        # self.show_image(new_image, new_label)

        return new_image, new_label

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def show_image(self, img, label):
        import colorsys
        if self.num_classes <= 21:
            colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                      (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                      (192, 0, 128),
                      (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                      (0, 64, 128), (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        color = [x for sublist in colors for x in sublist]
        if isinstance(img, Image.Image):
            img = img
            lab = label
        else:
            img = Image.fromarray((img).astype(np.uint8))
            lab = Image.fromarray((label).astype(np.uint8))
        lab.putpalette(color)
        aa = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bb = cv2.cvtColor(np.array(lab.convert('RGB')), cv2.COLOR_RGB2BGR)
        cv2.imshow('1', aa)
        cv2.imshow('2', bb)
        print(aa.shape, bb.shape)
        cv2.waitKey(0)

    # DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels


# if __name__ == '__main__':
#     with open("VOCdevkit\VOC2007\ImageSets\Segmentation/train.txt", "r") as f:
#         train_lines = f.readlines()
#     data = myDataset(train_lines, [800, 800], 7, True, 'VOCdevkit')
#     train_loader = DataLoader(data, batch_size=1)
#     for n, (i, j, k) in enumerate(train_loader):
#         print(1)
