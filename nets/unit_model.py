import torch
import torch.nn as nn
import torch.nn.functional as F


# class CBA(nn.Module):
#     def __init__(self, in_dim, out_dim, k=3, s=1, p=1, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(in_dim, out_dim, k, s, p, groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(out_dim)
#         self.act = nn.LeakyReLU() if act else nn.Identity()
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

class CBA(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, s=1, p=1, g=1, d=1, bn=True, bn_first=False, act=True):
        super().__init__()
        self.bn_first = bn_first
        self.conv = nn.Conv2d(in_dim, out_dim, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_dim) if bn else nn.Identity()
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.bn(self.act(self.conv(x))) if self.bn_first else self.act(self.bn(self.conv(x)))


class TransEncoder(nn.Module):
    def __init__(self, channel, num_head, num_layer, num_patches, use_pos_embed):
        super(TransEncoder, self).__init__()
        self.channel = channel
        self.use_pos_embed = use_pos_embed
        translayer = nn.TransformerEncoderLayer(d_model=channel, nhead=num_head, batch_first=True)
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


class res_block(nn.Module):
    def __init__(self, inplanes, planes):
        super(res_block, self).__init__()
        self.conv1 = CBA(inplanes, planes, act=False)
        self.conv2 = CBA(planes, planes, bn=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return out


class basicblock(nn.Module):
    def __init__(self, inplanes, branch_ratio=0.25):
        super(basicblock, self).__init__()

        gc = int(inplanes * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, kernel_size=(3, 3), padding=(1, 1), groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(7, 7), padding=(3, 3), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(11, 11), padding=(5, 5), groups=gc)
        self.split_indexes = (inplanes - 3 * gc, gc, gc, gc)
        self.bn = nn.BatchNorm2d(inplanes)
        self.mlp = nn.Sequential(nn.Conv2d(inplanes, 4 * inplanes, kernel_size=1, stride=1, padding=0),
                                 nn.GELU(),
                                 nn.Conv2d(4 * inplanes, inplanes, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(inplanes)
                                 )
        self.gamma = nn.Parameter(1e-6 * torch.ones((inplanes)), requires_grad=True)

    def forward(self, x):
        short_cut = x
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x_n = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)
        x_n = self.bn(x_n)
        x_n = self.mlp(x_n)
        x_n = x_n.mul(self.gamma.reshape(1, -1, 1, 1))
        x_n += short_cut
        return x_n


class unetUp(nn.Module):
    def __init__(self, cusize, upsize, up_='ConvT', k=4, s=2, p=1):
        super(unetUp, self).__init__()
        if up_ == 'ConvT':
            self.up = nn.ConvTranspose2d(cusize, upsize, k, s, p, bias=False)
            n = 2
        else:
            self.up = nn.Upsample(scale_factor=2)
            n = 3
        self.conv1 = CBA(n * upsize, upsize, act=False)
        self.gamma = nn.Parameter(1e-6 * torch.ones((upsize)), requires_grad=True)
        self.conv2 = CBA(upsize, upsize, bn=False)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = outputs.mul(self.gamma.reshape(1, -1, 1, 1))
        outputs = self.conv2(outputs)
        return outputs


class MAUnet(nn.Module):
    def __init__(self, basicblock=None, depths=(1, 1, 1, 1), dims=(64, 128, 256, 512), trans_layers=(4, 1),
                 num_classes=21, input_size=640, use_pos_embed=True, up_='ConvT'):
        super(MAUnet, self).__init__()
        self.dims = dims
        patchsize = (input_size // (2 ** len(depths))) ** 2
        self.down = CBA(3, dims[0], 3, 2, 1)
        self.stages = nn.ModuleList()

        for i in range(len(dims) - 1):
            stage = nn.Sequential(
                *[basicblock(dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)

        self.downsample_layers = nn.ModuleList()

        for i in range(len(dims) - 1):  # 0 1 2
            self.downsample_layers.append(CBA(dims[i], dims[i + 1], 3, 2, 1, bn_first=True, act=False))

        if trans_layers[1] != 0:
            self.transformer_encoder = TransEncoder(channel=dims[-1], num_head=trans_layers[0],
                                                    num_layer=trans_layers[1], num_patches=patchsize,
                                                    use_pos_embed=use_pos_embed)
        else:
            self.transformer_encoder = basicblock(dims[-1], dims[-1])

        self.up_layers = nn.ModuleList()

        if up_ == 'ConvT':
            self.up = nn.ConvTranspose2d(dims[0], dims[0], 4, 2, 1, bias=False)
        else:
            self.up = nn.Upsample(scale_factor=2)

        for i in range(len(dims) - 1):
            cusize = dims[len(self.dims) - 1 - i]
            upsize = dims[len(self.dims) - 2 - i]
            self.up_layers.append(unetUp(cusize, upsize, up_))

        self.seg_head = nn.Conv2d(dims[0], num_classes, 1)

    def forward(self, x):
        x = self.down(x)
        layers = []
        for i in range(len(self.dims) - 1):
            x = self.stages[i](x)
            layers.append(x)
            x = self.downsample_layers[i](x)

        x = self.transformer_encoder(x)

        for i in range(len(self.dims) - 1):
            x = self.up_layers[i](layers[len(self.dims) - 2 - i], x)
        x = self.up(x)
        x = self.seg_head(x)
        return x


def MA_Unet_T(basicblock=basicblock, depths=(2, 2, 2, 2), dims=(64, 128, 256, 512), trans_layers=(4, 1), num_classes=21,
              input_size=512, use_pos_embed=True):
    model = MAUnet(basicblock=basicblock, depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes,
                   input_size=input_size, use_pos_embed=use_pos_embed)
    return model


def MA_Unet_S(basicblock=basicblock, depths=(3, 3, 6, 9, 3), dims=(48, 96, 192, 384, 768), trans_layers=(8, 3), num_classes=21,
              input_size=640, use_pos_embed=True):
    model = MAUnet(basicblock=basicblock, depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes,
                   input_size=input_size, use_pos_embed=use_pos_embed)
    return model


def MA_Unet_B(basicblock=basicblock, depths=(3, 6, 9, 9, 6), dims=(64, 128, 256, 512, 1024), trans_layers=(8, 6),
              num_classes=21, input_size=800, use_pos_embed=True):
    model = MAUnet(basicblock=basicblock, depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes,
                   input_size=input_size, use_pos_embed=use_pos_embed)
    return model


def MA_Unet_L(basicblock=basicblock, depths=(6, 6, 9, 12, 6), dims=(96, 192, 384, 768, 1536), trans_layers=(16, 6),
              num_classes=21, input_size=1024, use_pos_embed=True):
    model = MAUnet(basicblock=basicblock, depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes,
                   input_size=input_size, use_pos_embed=use_pos_embed)
    return model

# If GPU has enough memory, you can extend depths and dims of model

# model = MA_Unet_S().cuda()
# # a = torch.randn(1, 3, 512, 512).cuda()
# # y = model(a)
# # print(y.size())
# from torchinfo import summary
# summary(model, input_size=(1, 3, 512, 512))

