# Scaled-MA-Unet
An improved Multiattention-Unet is divided into four versions of different scales. The original version can refer to [code](https://github.com/1343744768/Multiattention-UNet). In addition, we also provide API for other models from segmentation-models-pytorch, such as Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus.

Quick Start Examples
========================
Install
------------------------
``` 
pip install -r requirements.txt 
``` 
* It is worth noting that python 3.7 and torch 1.8 are recommended.

Preparation of datasets
------------------------
1. All data should be placed in directory “VOCdevkit/VOC2007/”. <br>
   * The name of original image and its corresponding label must be consistent, their format can be different(important) <br>
      `Image: cat_00001.jpg ; Label: cat_00001.png`
2. Put all the original images in folder “JPEGImages” and all the labels in folder “SegmentationClass”.<br>
3. Run "voc2unet" to divides training sets and test sets. <br>
   * If your label is not in png format, modify the code in line 17, as follow: <br>
       `if seg.endswith(".png"):`
   * You can also modify the 12 lines of code to divide the training set and test set according to other proportions, as follow. The default ratio is 8:2. <br>
       `train_percent = 0.8`
   * Finally, four text files will be generated in the "VOCdevkit/VOC2007/ImageSets/" directory.

Training
------------------------
1. Training with Multi-GPU. （recommended） <br>
    ```
    set distributed = True <br>
    python -m torch.distributed.launch --nproc_per_node=num_gpu train.py
    ```
    If the memory is not released after training, use `pgrep python | xargs kill -s 9` <br>
 
2. Training with single GPU. <br>
    ```
    python train.py
    ```
    * It is worth noting that in the hyperparameters, num_classes should be set to the number of categories plus 1. <br>
      For example, if you want to segmentation cat and dog in the images, although there are only two categories, <br>
      you need to set it to 3, because the label of the background is 0. 

Prediction and Validation
------------------------
See `predict.py` for details

Details of Multi-Attention UNet
------------------------
You can learn the details of Multi-Attention UNet through the paper as follow, and please cite our papers if the code is useful for your papers. Thank you! <br>

 * MDPI and ACS Style <br>
Sun, Y.; Bi, F.; Gao, Y.; Chen, L.; Feng, S. A Multi-Attention UNet for Semantic Segmentation in Remote Sensing Images. Symmetry 2022, 14, 906. https://doi.org/10.3390/sym14050906 <br>

 * AMA Style <br>
Sun Y, Bi F, Gao Y, Chen L, Feng S. A Multi-Attention UNet for Semantic Segmentation in Remote Sensing Images. Symmetry. 2022; 14(5):906. https://doi.org/10.3390/sym14050906 <br>

 * Chicago/Turabian Style <br>
Yu Sun, Fukun Bi, Yangte Gao, Liang Chen, and Suting Feng. 2022. "A Multi-Attention UNet for Semantic Segmentation in Remote Sensing Images" Symmetry 14, no. 5: 906. https://doi.org/10.3390/sym14050906 <br>


Reference
------------------------
https://github.com/bubbliiiing/unet-pytorch  <br>
https://github.com/yassouali/pytorch-segmentation  <br>
https://github.com/qubvel/segmentation_models.pytorch  <br>
