import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.MA_UNet import MA_Unet_T, MA_Unet_S, MA_Unet_B, MA_Unet_L
from nets.unet_training import weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils_fit import fit_one_epoch
from torch.optim.lr_scheduler import _LRScheduler
import warnings
warnings.filterwarnings('ignore')
import argparse
import math

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def main(args):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    Cuda = True if torch.cuda.is_available() else False
    warmup_epoch, num_epoch, distributed, sync_bn, fp16, num_classes, model_path, input_shape, use_pos_embed, resume \
        = args.warmup_epoch, args.num_epoch, args.distributed, args.sync_bn, args.amp, args.num_classes, args.model_path, args.input_shape, args.use_pos_embed, args.resume

    pretrained = True if model_path is not None else False

    eval_period, dice_loss, focal_loss, num_workers, VOCdevkit_path, save_dir, save_period, Freeze_Train, batch_size, lr= \
        args.eval_period, args.dice_loss, args.focal_loss, args.num_workers, args.dataset, args.save_dir, args.save_period, args.Freeze_Train, args.batch_size, args.lr

    mosaic, end_mosaic_epoch, cutout, rotate, flip_rl, flip_ud, scale, color_transform = args.mosaic, args.end_mosaic_epoch, args.cutout, args.rotate, args.flip_rl, args.flip_ud, args.scale, args.color_transform

    cls_weights = np.ones([num_classes], np.float32)
    ngpus_per_node = torch.cuda.device_count()

    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("GPU Device Count : ", ngpus_per_node)
    else:
        local_rank = 0

    if local_rank == 0:
        print(args)

    if args.model_type is not None:
        if args.model_type == 'tiny':
            if local_rank == 0:
                print('use tiny model')
            model = MA_Unet_T(num_classes=num_classes, input_size=input_shape[0], use_pos_embed=use_pos_embed)
        elif args.model_type == 'small':
            if local_rank == 0:
                print('use small model')
            model = MA_Unet_S(num_classes=num_classes, input_size=input_shape[0], use_pos_embed=use_pos_embed)
        elif args.model_type == 'base':
            if local_rank == 0:
                print('use basic model')
            model = MA_Unet_B(num_classes=num_classes, input_size=input_shape[0], use_pos_embed=use_pos_embed)
        else:
            if local_rank == 0:
                print('use large model')
            model = MA_Unet_L(num_classes=num_classes, input_size=input_shape[0], use_pos_embed=use_pos_embed)

    else:
        from segmentation_models_pytorch import Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN
        if local_rank == 0:
            print('training with %s, backbone used with %s' % (args.model_name, args.backbone_name))
        model_name = eval(args.model_name)
        backbone = args.backbone_name
        weights = "imagenet" if args.use_pretrained else None
        model = model_name(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )

    if not resume:
        if pretrained:
            if local_rank == 0:
                print('Load pre_weights from {}.'.format(model_path))
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location='cpu')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        else:
            if distributed:
                if local_rank == 0:
                    if args.model_type is not None or not args.use_pretrained:
                        weights_init(model)
                    torch.save(model.state_dict(), "initial_weights.pt")
                dist.barrier()
                model.load_state_dict(torch.load("initial_weights.pt", map_location='cpu'))
                if local_rank == 0:
                    if os.path.exists("initial_weights.pt"):
                        os.remove("initial_weights.pt")
            else:
                weights_init(model)

    if local_rank == 0:
        log_dir = os.path.join(save_dir, "loss")
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    if sync_bn and ngpus_per_node > 1 and distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if local_rank == 0:
            print('use sync_bn')
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:    # DDP
            model = model.cuda(local_rank)
            if Freeze_Train or args.model_type == None:
                find_para = True
            else:
                find_para = False
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_para)
            model_without_ddp = model.module
            cudnn.benchmark = True
        else:
            if torch.cuda.device_count() > 1:        # DP
                model = torch.nn.DataParallel(model)
                cudnn.benchmark = True
                model = model.cuda()
                model_without_ddp = model.module
            else:                                   # Single GPU
                cudnn.benchmark = True
                model = model.cuda()
                model_without_ddp = model


    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if args.adam:
        optimizer = optim.Adam(model_without_ddp.parameters(), lr)
    else:
        optimizer = optim.SGD(model_without_ddp.parameters(), lr, momentum=0.843, weight_decay=0.00036, nesterov=True)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path,  mosaic=mosaic, rotate=rotate, cutout=cutout, scale=scale, flip_rl=flip_rl, flip_ud=flip_ud, color_trans=color_transform)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    # iter_per_epoch = len(train_dataset)
    warmup_scheduler = WarmUpLR(optimizer, epoch_step * warmup_epoch)

    if args.lrf is not None:
        lf = lambda x: ((1 + math.cos(x * math.pi / args.num_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=args.lrf)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.96)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler, val_sampler = None, None
        shuffle = True

    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler)
    valid_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler)

    if resume:
        if local_rank == 0:
            print('resume from last weights')
        checkpoint = torch.load(os.path.join(save_dir, "last_weights.pth"), map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if fp16:
            scaler.load_state_dict(checkpoint["scaler"])
    else:
        start_epoch = 0

    if local_rank == 0:
        eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, period=eval_period)
    else:
        eval_callback = None

    for epoch in range(start_epoch, num_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= end_mosaic_epoch:
            train_loader.dataset.end_mosaic = True
        fit_one_epoch(model, model_without_ddp, loss_history, eval_callback, optimizer, epoch, warmup_scheduler, warmup_epoch, epoch_step, epoch_step_val, train_loader, valid_loader,
                      num_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank, lr_scheduler)

        if epoch >= warmup_epoch:
            lr_scheduler.step()

        if distributed:
            dist.barrier()

    # if local_rank == 0:
    #     loss_history.writer.close()

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### training MA-UNet, if u want to train other models, please set model_type = None in line 251
    parser.add_argument("--model_type", default='base', type=str, help=" ['tiny', 'small', 'base', 'large', None] ")
    parser.add_argument("--use_pos_embed", default=False, type=bool, help='Whether to enable absolute position embedding, input size of prediction after enabling will be fixed')
    parser.add_argument("--model_path", default=None, type=str, help='Pretrained model, if not None, use it')
    parser.add_argument("--Freeze_Train", default=False, type=bool, help='Freeze the backbone to train')
    parser.add_argument("--Freeze_epoch", default=None, type=int, help='Number epochs of freeze the backbone to train')

    ### training other model        pip install segmentation_models_pytorch
    ### if u want to train other models, please set model_type=None in line 251, otherwise training MA-UNet
    ### https://github.com/qubvel/segmentation_models.pytorch
    parser.add_argument("--model_name", default='UnetPlusPlus', type=str, help="[Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]")
    parser.add_argument("--backbone_name", default='vgg16_bn', type=str, help='More than 100 backbone are available, see github for details')
    ### ['vgg16_bn', 'resnet34', 'resnext50_32x4d', 'timm-resnest50d', 'densenet121', 'efficientnet-b0' , ...]
    parser.add_argument("--use_pretrained", default=True, type=bool, help='Backbone initialization using pretrained weight from Imagenet')

    ### DDP mode. If distributed=True training with DDP mode, use python -m torch.distributed.launch --nproc_per_node=number_GPUs train.py
    parser.add_argument("--local_rank", default=-1, type=int, help="Don't change it")
    parser.add_argument("--distributed", default=False, type=bool, help="Use DDP for training")
    parser.add_argument("--sync_bn", default=False, type=bool, help='Use sync batch normalization, only used in DDP')
    parser.add_argument("--amp", default=False, type=bool, help="Use torch.cuda.amp for mixed precision training, only used in DDP")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device_id", default='0', type=str, help="Numbers of cuda want to use. if you only have one GPU, default=0")

    ### Hyperparameter
    parser.add_argument("--input_shape", default=[640, 640], type=list, help='Image size for CNN after Data Augmentation')
    parser.add_argument("--adam", default=True, type=bool, help="True: adam, False: SGD; If number of data < 2000 use adam, else use SGD")
    parser.add_argument("--lr", default=1e-3, type=float, help='Learning rate.')
    parser.add_argument("--lrf", default=1e-6, type=float, help='Final learning rate. If it is not None, use CosineAnnealing, else use lr_step')
    parser.add_argument("--num_epoch", default=200, type=int, help='total epochs to train')
    parser.add_argument("--warmup_epoch", default=3, type=int, help='num epochs for warm up')
    parser.add_argument("--num_classes", default=21, type=int, help='Actual category quantity plus 1 (background)')
    parser.add_argument("--batch_size", default=16, type=int, help='Total batchsize')
    parser.add_argument("--dataset", default='VOCdevkit', type=str, help='Dataset path')
    """
    --dataset/
        --VOC2007/
            --JPEGImages/    (original image)
            --SegmentationClass/    (label image, the naming method must be consistent with the original image)
            --ImageSets/
                --Segmentation
                    --train.txt
                    --val.txt
                    --test.txt
                    (During training, only files train.txt and val.txt are required)
    """
    parser.add_argument("--save_dir", default='logs', type=str, help='Models save path')
    parser.add_argument("--save_period", default=5, type=int)
    parser.add_argument("--resume", action='store_true', help='resume most recent training')
    parser.add_argument("--dice_loss", default=True, type=bool, help='if num_classes<10, use this')
    parser.add_argument("--focal_loss", default=True, type=bool, help='if all kinds of samples is unbalanced, use this, else ce_loss is default set')
    parser.add_argument("--eval_period", default=5, type=int, help='Evaluate the performance of the valid datasets after training the number epoch')

    ###data pro
    parser.add_argument("--mosaic", default=0.0, type=float, help='Probability of using mosaic for data enhancement')
    parser.add_argument("--end_mosaic_epoch", default=50, type=float, help='after this epoch, end mosaic datapro')
    parser.add_argument("--cutout", default=0.0, type=float, help='Probability of using cutout for data enhancement')
    # Boundary mode of rotate: we use cv2.BORDER_REFLECT_101; You can refer to the link bellow:
    # https://blog.csdn.net/weixin_37804469/article/details/126727003?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166727335716782425684093%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166727335716782425684093&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-2-126727003-null-null.142^v62^control,201^v3^control_1,213^v1^t3_control1&utm_term=borderMode&spm=1018.2226.3001.4187
    parser.add_argument("--rotate", default=0.5, type=float, help='Probability of using rotate for data enhancement, (-90, 90)')
    parser.add_argument("--flip_rl", default=0.5, type=float, help='Probability of using flip right and left for data enhancement')
    parser.add_argument("--flip_ud", default=0.0, type=float, help='Probability of using flip up and down for data enhancement')
    parser.add_argument("--scale", default=(0.5, 1.5), type=tuple, help='Using scale for data enhancement, scaled randomly from default set (0.5, 1.5)')
    parser.add_argument("--color_transform", default=True, type=bool, help='Use color transform data enhancement')
    args = parser.parse_args()
    main(args)
