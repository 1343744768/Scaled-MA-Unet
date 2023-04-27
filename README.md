Scaled-MA-Unet is an improved Multi-Attention-Unet that has been divided into four versions of different scales. The original version can be found in the following code repository. In addition, we provide an API for other models from segmentation-models-pytorch, such as Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, and DeepLabV3Plus.

Quick Start Examples
Install
Copy code
pip install -r requirements.txt 
It is worth noting that python 3.7 and torch 1.8 are recommended.

Preparation of datasets
All data should be placed in the directory "VOCdevkit/VOC2007/".
The name of the original image and its corresponding label must be consistent, but their formats can be different (important).
For example, "Image: cat_00001.jpg ; Label: cat_00001.png"
Put all the original images in the folder "JPEGImages" and all the labels in the folder "SegmentationClass".
Run "voc2unet" to divide the training sets and test sets.
If your label is not in png format, modify the code in line 17 as follows:
arduino
Copy code
if seg.endswith(".png"):
You can also modify the 12 lines of code to divide the training set and test set according to other proportions. The default ratio is 8:2.
makefile
Copy code
train_percent = 0.8
Finally, four text files will be generated in the "VOCdevkit/VOC2007/ImageSets/" directory.
Training
Training with Multi-GPU (recommended).
Set distributed = True and run the following command:
css
Copy code
python -m torch.distributed.launch --nproc_per_node=num_gpu train.py
If memory is not released after training, use pgrep python | xargs kill -s 9.
Training with single GPU.
Run the following command:
Copy code
python train.py
It is worth noting that in the hyperparameters, num_classes should be set to the number of categories plus 1. For example, if you want to segment cats and dogs in the images, although there are only two categories, you need to set it to 3 because the label of the background is 0.
Prediction and Validation
See predict.py for details.

Details of Multi-Attention UNet
You can learn the details of Multi-Attention UNet through the paper as follows, and please cite our papers if the code is useful for your papers. Thank you!

MDPI and ACS Style:
Sun, Y.; Bi, F.; Gao, Y.; Chen, L.; Feng, S. A Multi-Attention UNet for Semantic Segmentation in Remote Sensing Images. Symmetry 2022, 14, 906. https://doi.org/10.3390/sym14050906

AMA Style:
Sun Y, Bi F, Gao Y, Chen L, Feng S. A Multi-Attention UNet for Semantic Segmentation in Remote Sensing Images. Symmetry. 2022; 14(5):906. https://doi.org/10.3390/sym14050906
