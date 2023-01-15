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

'''
This is just a test code,
To test the restoration of masked images based on the MAE pre-training model, 
and to observe the effect of the pre-training model.
'''

model_path = '/local/sunyu/scale-unet/logs/exp-050-train_loss0.001-val_loss0.001.pth'  # pre-training model path
img_path = r'VOCdevkit/VOC2007/JPEGImages/05.png'  # test image
coeffi_num = 2   # Coefficients of masked images
hole_size = (5, 5)   # Size of masked area, (h, w)
fill_color = (0, 0, 0)

# Select a training model, num_classes must equal 3
model = MA_Unet_S(num_classes=3, input_size=640, use_pos_embed=False).cuda()
model.load_state_dict(torch.load(model_path))
model.eval()
image = Image.open(img_path)
img, nw, nh = resize_image(image, (640, 640))
img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
mae_jpg = np.copy(img)
h = img.shape[0]
w = img.shape[1]
num_holes = coeffi_num * min(h//hole_size[0], w//hole_size[1])
for _ in range(num_holes):
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(max(0, y - hole_size[0] // 2), 0, h)
    y2 = np.clip(max(0, y + hole_size[0] // 2), 0, h)
    x1 = np.clip(max(0, x - hole_size[1] // 2), 0, w)
    x2 = np.clip(max(0, x + hole_size[1] // 2), 0, w)
    mae_jpg[y1: y2, x1: x2, :] = fill_color

img = Image.fromarray(cv2.cvtColor(mae_jpg, cv2.COLOR_BGR2RGB))
image_data = np.expand_dims(np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1)), 0)
image_data = torch.from_numpy(image_data).cuda()
predict = model(image_data).squeeze(0).transpose(0, 1).transpose(1, 2)
result_img = np.array(predict.cpu().detach().numpy()*255).astype(np.uint8)
result_img = Image.fromarray(result_img)
img.save('ori_image.jpg')
result_img.save('restore_img.jpg')
