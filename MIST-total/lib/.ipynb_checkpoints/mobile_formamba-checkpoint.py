import time

import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from torch.autograd import Variable
# from WTConv import WTConv2d
from lib.WTConv import WTConv2d

# conv_bn为网络的第一个卷积块，步长为2
def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


# conv_dw为深度可分离卷积
def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        # 3x3卷积提取特征，步长为2
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        # 1x1卷积，步长为1
        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_channels):
        super(MobileNet, self).__init__()
        self.layer1 = nn.Sequential(
            # 第一个卷积块，步长为2，压缩一次
            conv_bn(n_channels, 24, 1),  # 416,416,3 -> 208,208,32
            conv_dw(24, 48, 2),  # 208,208,32 -> 208,208,64
            conv_dw(48, 96, 2),  # 208,208,64 -> 104,104,128
            WTConv2d(96,96),
            WTConv2d(96,96),
            # conv_dw(96, 96, 1),
        )
        self.layer2 = nn.Sequential(
            conv_dw(96, 192, 2),
            WTConv2d(192,192),
            WTConv2d(192,192),
            # conv_dw(192, 192, 1),
            # conv_dw(192, 192, 1),

        )
        self.layer3 = nn.Sequential(
            conv_dw(192, 384, 2),
            WTConv2d(384,384),
            WTConv2d(384,384),
            # conv_dw(384, 384, 1),
            # conv_dw(384, 384, 1),

        )
        # 26,26,512 -> 13,13,1024
        self.layer4 = nn.Sequential(
            conv_dw(384, 768, 2),
            WTConv2d(768,768),
            WTConv2d(768,768),
            # conv_dw(768, 768, 1),
            # conv_dw(768, 768, 1),
        )
        self.res = []
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        self.res = [x1,x2,x3,x4]
        for i in range(len(self.res)):
             self.res[i] = self.res[i].permute(0, 2, 3, 1)
        return self.res


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNet(n_channels=3).cuda()
    # model.to(device)
    summary(model, input_size=(3, 256, 256))

    model = MobileNet(n_channels=3).cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    wave = model(x)
    for i in range(len(wave)):
        print(wave[i].shape)