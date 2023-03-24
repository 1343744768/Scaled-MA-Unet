import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import torchvision.ops


class CBA(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, s=1, p=1, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.act = nn.LeakyReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DCv1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DCv1, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = torchvision.ops.deform_conv2d(input=x.float(),
                                          offset=offset.float(),
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=(self.padding, self.padding),
                                          stride=self.stride,
                                          )
        return x


class DCv2(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, offset_groups=1, with_mask=True):
        super().__init__()
        assert in_dim % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(x, offset=offset, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)
        return x


class TransEncoder(nn.Module):
    def __init__(self, channel, num_head, num_layer, num_patches, use_pos_embed):
        super(TransEncoder, self).__init__()
        self.channel = channel
        self.use_pos_embed = use_pos_embed
        translayer = nn.TransformerEncoderLayer(d_model=channel, nhead=num_head)
        self.trans = nn.TransformerEncoder(translayer, num_layers=num_layer)
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, channel))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        if self.use_pos_embed:
            x = x.flatten(2).transpose(1, 2) + self.pos_embed
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.trans(x)
        x = x.transpose(1, 2).view(-1, self.channel, int(h), int(w))
        return x

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_dim, out_dim, shortcut=True):
        super().__init__()
        self.cv1 = DCv1(in_dim, out_dim)
        self.cv2 = DCv1(out_dim, out_dim)
        self.add = shortcut and in_dim == out_dim

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, inplanes, planes, shortcut=True):
        super().__init__()
        self.cv1 = CBA(inplanes, int(planes/2), 1, 1, 0)
        self.cv2 = CBA(inplanes, int(planes/2), 1, 1, 0)
        self.cv3 = CBA(planes, planes, 1, 1, 0)
        self.m = Bottleneck(int(planes/2), int(planes/2), shortcut)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = CBA(c1, c_, 1, 1, 0)
        self.cv2 = CBA(c_ * 4, c2, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class res_dcblock(nn.Module):
    def __init__(self, inplanes, planes):
        super(res_dcblock, self).__init__()
        self.conv1 = nn.Sequential(DCv1(inplanes, planes), nn.BatchNorm2d(planes), nn.LeakyReLU())
        self.conv2 = nn.Sequential(DCv1(planes, planes), nn.BatchNorm2d(planes))
        self.act = nn.LeakyReLU()
        if inplanes != planes:
            self.downsample = CBA(inplanes, planes, k=1, s=1, p=0, g=1, d=1, act=False)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out


class sim_residual_block(nn.Module):
    def __init__(self, inplanes, planes):
        super(sim_residual_block, self).__init__()
        self.conv1 = CBA(inplanes, planes)
        self.conv2 = CBA(planes, planes, act=False)

        # self.activaton = nn.Sigmoid()
        # self.e_lambda = 1e-4

        if inplanes != planes:
            self.downsample = CBA(inplanes, planes, k=1, s=1, p=0, g=1, d=1, act=False)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        # b, c, h, w = out.size()
        # n = w * h - 1
        # x_minus_mu_square = (out - out.mean(dim=[2, 3], keepdim=True)).pow(2)
        # y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # out = out * self.activaton(y)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.leaky_relu(out)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, inchannel, channel, reduction=16, spatial_kernel=7):
        super(AttentionLayer, self).__init__()
        self.Dconv = CBA(inchannel, channel, 1, 1, 0)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Dconv(x)
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class unetUp(nn.Module):
    def __init__(self, basic_block, cusize, upsize, depths, index, up_='nearest'):
        super(unetUp, self).__init__()
        if up_ == 'ConvT':
            self.up = nn.ConvTranspose2d(cusize, upsize, 4, 2, 1, bias=False)
            n = 2
        else:
            self.up = nn.Upsample(scale_factor=2)
            n = 3
        self.att = AttentionLayer(n*upsize, upsize)
        self.conv2 = nn.Sequential(
                *[basic_block(upsize, upsize) for j in range(depths[::-1][index])]
            )

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.att(outputs)
        outputs = self.conv2(outputs)
        return outputs


class MAUnet(nn.Module):
    def __init__(self, basicblock=None, depths=(1, 1, 1, 1), dims=(64, 128, 256, 512), trans_layers=(4, 1), num_classes=21, input_size=640, use_pos_embed=True, up_='ConvT'):
        super(MAUnet, self).__init__()
        self.dims = dims
        patchsize = (input_size//(2**(len(depths))))**2
        self.down = CBA(3, dims[0], 6, 2, 2)
        self.stages = nn.ModuleList()

        for i in range(len(dims)-1):
            stage = nn.Sequential(
                *[basicblock(dims[i], dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)

        self.downsample_layers = nn.ModuleList()

        for i in range(len(dims)-1):
            self.downsample_layers.append(CBA(dims[i], dims[i + 1], 3, 2, 1))

        if trans_layers[1] != 0:
            self.transformer_encoder = TransEncoder(channel=dims[-1], num_head=trans_layers[0], num_layer=trans_layers[1], num_patches=patchsize, use_pos_embed=use_pos_embed)
        else:
            self.transformer_encoder = basicblock(dims[-1], dims[-1])

        self.up_layers = nn.ModuleList()

        if up_ == 'ConvT':
            self.up = nn.ConvTranspose2d(dims[0], dims[0], 4, 2, 1, bias=False)
        else:
            self.up = nn.Upsample(scale_factor=2)

        for i in range(len(dims)-1):
            cusize = dims[len(self.dims) - 1 - i]
            upsize = dims[len(self.dims) - 2 - i]
            self.up_layers.append(unetUp(basicblock, cusize, upsize, depths[:-1], i, up_))

        self.seg_head = nn.Conv2d(dims[0], num_classes, 1)

    def forward(self, x):
        x = self.down(x)
        layers = []
        for i in range(len(self.dims)-1):
            x = self.stages[i](x)
            layers.append(x)
            x = self.downsample_layers[i](x)

        x = self.transformer_encoder(x)

        for i in range(len(self.dims)-1):
            x = self.up_layers[i](layers[len(self.dims)-2-i], x)
        x = self.up(x)
        x = self.seg_head(x)
        return x

def MA_Unet_T(basicblock=sim_residual_block, depths=(1, 1, 1, 1), dims=(32, 64, 128, 256), trans_layers=(4, 1), num_classes=21, input_size=512, use_pos_embed=True):
    model = MAUnet(basicblock=basicblock, depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes, input_size=input_size, use_pos_embed=use_pos_embed)
    return model

def MA_Unet_S(basicblock=sim_residual_block, depths=(2, 2, 4, 2), dims=(64, 128, 256, 512), trans_layers=(8, 3), num_classes=21, input_size=512, use_pos_embed=True):
    model = MAUnet(basicblock=basicblock, depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes, input_size=input_size, use_pos_embed=use_pos_embed)
    return model

def MA_Unet_B(basicblock=sim_residual_block, depths=(2, 3, 6, 2, 1), dims=(64, 128, 256, 512, 512), trans_layers=(8, 6), num_classes=21, input_size=640, use_pos_embed=True):
    model = MAUnet(basicblock=basicblock, depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes, input_size=input_size, use_pos_embed=use_pos_embed)
    return model

def MA_Unet_L(basicblock=sim_residual_block, depths=(2, 3, 9, 2, 1), dims=(64, 128, 256, 512, 1024), trans_layers=(16, 6), num_classes=21, input_size=800, use_pos_embed=True):
    model = MAUnet(basicblock=basicblock, depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes, input_size=input_size, use_pos_embed=use_pos_embed)
    return model

# If GPU has enough memory, you can extend depths and dims of model

# model = MA_Unet_S().cuda()
# # a = torch.randn(1, 3, 512, 512).cuda()
# # y = model(a)
# # print(y.size())
# from torchinfo import summary
# summary(model, input_size=(1, 3, 512, 512))

