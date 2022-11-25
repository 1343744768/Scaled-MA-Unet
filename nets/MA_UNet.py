import torch
import torch.nn as nn


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
        input_x = x
        if self.use_pos_embed:
            x = x.flatten(2).transpose(1, 2) + self.pos_embed
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.trans(x)
        x = input_x + x.transpose(1, 2).view(-1, self.channel, int(h), int(w))
        return x

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

def conv1x1(in_planes, out_planes):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.1, inplace=True))

class srblock(nn.Module):
    def __init__(self, inplanes, planes):
        super(srblock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.activaton = nn.Sigmoid()
        self.e_lambda = 1e-4

        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        b, c, h, w = out.size()
        n = w * h - 1
        x_minus_mu_square = (out - out.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        out = out * self.activaton(y)

        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class unetUp(nn.Module):
    def __init__(self, upsize, depths, ii):
        super(unetUp, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(upsize*2, upsize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(upsize),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(upsize*2, upsize, kernel_size=3, padding=1),
            nn.BatchNorm2d(upsize),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.att = CBAMLayer(upsize)
        self.conv2 = nn.Sequential(
                *[srblock(upsize, upsize) for j in range(depths[::-1][ii])]
            )

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.att(outputs) * outputs
        outputs = self.conv2(outputs)
        return outputs

class MAUnet(nn.Module):
    def __init__(self, depths=(1, 1, 1, 1, 1, 1), dims=(32, 64, 128, 256, 512, 1024), trans_layers=(4, 1), num_classes=21, input_size=640, use_pos_embed=True):
        super(MAUnet, self).__init__()
        self.dims = dims
        patchsize = (input_size//(2**(len(depths)-1)))**2
        self.stages = nn.ModuleList()
        for i in range(len(dims)-1):
            inplace = dims[i] if i != 0 else 3
            place = dims[i]
            stage = nn.Sequential(
                *[srblock(inplace, place) for j in range(depths[i])]
            )
            self.stages.append(stage)

        self.downsample_layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[i+1]),
                nn.LeakyReLU(0.1, inplace=True)
            ))

        if trans_layers[1] != 0:
            self.transformer_encoder = TransEncoder(channel=dims[-1], num_head=trans_layers[0], num_layer=trans_layers[1], num_patches=patchsize, use_pos_embed=use_pos_embed)
        else:
            self.transformer_encoder = srblock(dims[-1], dims[-1])

        self.up_layers = nn.ModuleList()
        for i in range(len(dims)-1):  # 0 1 2 3 4
            upsize = dims[len(self.dims)-2-i]
            self.up_layers.append(unetUp(upsize, depths[:-1], i))
        self.final = nn.Conv2d(dims[0], num_classes, 1)

    def forward(self, x):
        layers = []
        for i in range(len(self.dims)-1):
            x = self.stages[i](x)
            layers.append(x)
            x = self.downsample_layers[i](x)

        x = self.transformer_encoder(x)
        for i in range(len(self.dims)-1):
            x = self.up_layers[i](layers[len(self.dims)-2-i], x)
        x = self.final(x)
        return x

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


def MA_Unet_T(depths=(1, 1, 1, 1, 1), dims=(24, 48, 96, 192, 384), trans_layers=(0, 0), num_classes=21, input_size=512, use_pos_embed=True):
    model = MAUnet(depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes, input_size=input_size, use_pos_embed=use_pos_embed)
    return model

def MA_Unet_S(depths=(1, 1, 1, 1, 1, 1), dims=(24, 48, 96, 192, 384, 768), trans_layers=(4, 1), num_classes=21, input_size=640, use_pos_embed=True):
    model = MAUnet(depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes, input_size=input_size, use_pos_embed=use_pos_embed)
    return model

def MA_Unet_B(depths=(1, 1, 1, 1, 1, 1), dims=(32, 64, 128, 256, 512, 1024), trans_layers=(4, 1), num_classes=21, input_size=800, use_pos_embed=True):
    model = MAUnet(depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes, input_size=input_size, use_pos_embed=use_pos_embed)
    return model

def MA_Unet_L(depths=(1, 2, 2, 2, 2, 1), dims=(48, 96, 192, 384, 768, 1536), trans_layers=(8, 3), num_classes=21, input_size=800, use_pos_embed=True):
    model = MAUnet(depths=depths, dims=dims, trans_layers=trans_layers, num_classes=num_classes, input_size=input_size, use_pos_embed=use_pos_embed)
    return model

# If GPU has enough memory, you can extend depths and dims of model

# model = MA_Unet_S()
# a = torch.randn(1, 3, 512, 512)
# y = model(a)
# print(y.size())
# from torchinfo import summary
# summary(model, input_size=(1, 3, 640, 640))

