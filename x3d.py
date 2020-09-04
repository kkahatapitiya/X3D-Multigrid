import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=(1,stride,stride),
                     padding=1,
                     bias=False,
                     groups=in_planes
                     )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=(1,stride,stride),
                     bias=False)


class Bottleneck(nn.Module):
    #expansion = 1 #4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1x1(in_planes, planes[0])
        self.bn1 = nn.BatchNorm3d(planes[0])
        self.conv2 = conv3x3x3(planes[0], planes[0], stride)
        self.bn2 = nn.BatchNorm3d(planes[0])
        self.conv3 = conv1x1x1(planes[0], planes[1])
        self.bn3 = nn.BatchNorm3d(planes[1])
        self.swish = nn.Hardswish()
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1 = nn.Linear(planes[1], planes[1]//4) # //16 in original SE-Net
        self.fc2 = nn.Linear(planes[1]//4, planes[1])
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.swish(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.swish(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Squeeze-and-Excitation

        se_w = self.global_pool(out).view(-1, out.shape[1])
        se_w = self.fc1(se_w)
        se_w = self.swish(se_w)
        se_w = self.fc2(se_w)
        se_w = self.sigmoid(se_w)
        out = out * se_w.view(-1, out.shape[1], 1, 1, 1)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.swish(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 #no_max_pool=True,
                 shortcut_type='B',
                 widen_factor=1.0,
                 dropout=0.7,
                 n_classes=400):
        super(ResNet, self).__init__()

        block_inplanes = [(int(x * widen_factor),int(y * widen_factor)) for x,y in block_inplanes]

        self.in_planes = block_inplanes[0][1]
        #self.no_max_pool = no_max_pool
        #self.gamma_t = gamma_t

        self.conv1_s = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(1, 3, 3),
                               stride=(1, 2, 2),
                               padding=(0, 1, 1),
                               bias=False)
        self.conv1_t = nn.Conv3d(self.in_planes,
                               self.in_planes,
                               kernel_size=(3, 1, 1),
                               stride=(1, 1, 1),
                               padding=(1, 0, 0),
                               bias=False,
                               groups=self.in_planes)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.swish = nn.Hardswish()
        #self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,
                                       block_inplanes[0],
                                       layers[0],
                                       shortcut_type,
                                       stride=2)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.conv5 = nn.Conv3d(block_inplanes[3][1],
                               block_inplanes[3][0],
                               kernel_size=(1, 1, 1),
                               stride=(1, 1, 1),
                               padding=(0, 0, 0),
                               bias=False)
        self.bn5 = nn.BatchNorm3d(block_inplanes[3][0])
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc1 = nn.Linear(block_inplanes[3][0], 2048)
        self.fc2 = nn.Linear(2048, n_classes)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu') # Unsupported nonlinearity hardswish
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes[1]:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes[1],
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes[1], stride),
                    nn.BatchNorm3d(planes[1]))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes[1]
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.conv1_t(x)
        x = self.bn1(x)
        x = self.swish(x)
        #if not self.no_max_pool:
        #    x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.swish(x)

        # *** for localization, no pool in Temporal dim ***
        #b,c,t,h,w = x.shape
        #x = self.avgpool(x.view(b,c*t,h,w)).view(b,c,t,1,1)
        x = self.avgpool(x)

        #x = x.view(x.size(0), -1)
        x = x.squeeze(4).squeeze(3).permute(0,2,1) # B T C

        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = x.permute(0,2,1) # B C T

        return x

def get_inplanes(version):
    planes = {'S':[(54,24), (108,48), (216,96), (432,192)],
              'M':[(54,24), (108,48), (216,96), (432,192)],
              'XL':[(72,32), (162,72), (306,136), (630,280)]}

    return planes[version]

def get_blocks(version):
    blocks = {'S':[3,5,11,7],
              'M':[3,5,11,7],
              'XL':[5,10,25,15]}

    return blocks[version]

def generate_model(x3d_version, **kwargs):

    #gamma_t = {'S':13, 'M':16, 'XL':16}[x3d_version]
    model = ResNet(Bottleneck, get_blocks(x3d_version), get_inplanes(x3d_version), **kwargs)

    return model
