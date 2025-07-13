# 双树复小波变换与逆变换模块
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from torch.nn.functional import kl_div
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_planes, bias=False)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1,output_padding=0, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,output_padding= output_padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.inch = in_planes
    def forward(self, x):

        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class DepthwiseSeparableTransConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableTransConv2d, self).__init__()
        self.depthwise = nn.ConvTranspose2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_padding=output_padding, groups=in_planes, dilation=dilation, bias=bias)
        self.pointwise = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class DWT(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(DWT, self).__init__()
        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        #传统卷积示例
        self.conv1 = BasicConv2d(outchannel,outchannel)
        self.conv2 = BasicConv2d(inchannel, outchannel)
        self.conv3 = BasicConv2d(outchannel, outchannel)
        self.change = TransBasicConv2d(outchannel,outchannel)
        
        #纯深度可分离卷积示例
        # self.conv1 = DepthwiseSeparableConv2d(outchannel,outchannel)
        # self.conv2 = DepthwiseSeparableConv2d(inchannel, outchannel)
        # self.conv3 = DepthwiseSeparableConv2d(outchannel, outchannel)
        # self.change = DepthwiseSeparableTransConv2d(outchannel,outchannel)

    def forward(self, x, y):
        # print('x',x.shape)

        y = self.change(self.conv2(y))
        # print('y',y.shape)

        Xl, Xh = self.DWT(x)
        Yl, Yh = self.DWT(y)

        # print('Xl',Xl.shape)
        # print('Yl', Yl.shape)
        x_y = self.conv1(Xl)+self.conv1(Yl)
        # print('x_y',x_y.shape)
        # print('Xh',Xh.shape)
        # print('Yh',Yh.shape)
        x_m = self.IWT((x_y,Xh))
        y_m = self.IWT((x_y,Yh))
        # print('x_m',x_m.shape)
        # print('y_m',y_m.shape)
        out = self.conv3(x_m + y_m)
        return out

if __name__ == '__main__':
    # 构建模型
    model = DWT(inchannel=3, outchannel=3)
    x = torch.randn(1, 3, 512, 512)  # 随机生成形状为 (1, 3, 256, 256) 的张量
    y = torch.randn(1, 3, 256, 256)  # 随机生成形状为 (1, 3, 256, 256) 的张量
    output = model(x, y)
    print("Output shape:", output.shape)
    #卷积测试
    # 创建一个 BasicConv2d 实例
    # conv_layer = BasicConv2d(in_planes=3, out_planes=16, kernel_size=3, stride=1, padding=1)
    # x = torch.randn(1, 3, 64, 64)  # 随机生成形状为 (1, 3, 64, 64) 的张量
    # output = conv_layer(x)
    # print("Output shape:", output.shape)

    #深度可分离卷积示例
    # conv_layer = DepthwiseSeparableConv2d(in_planes=3, out_planes=16, kernel_size=3, stride=1, padding=1)
    # x = torch.randn(1, 3, 64, 64)  # 随机生成形状为 (1, 3, 64, 64) 的张量
    # output = conv_layer(x)
    # print("Output shape:", output.shape)

    #反卷积示例：
    # trans_conv_layer = TransBasicConv2d(in_planes=3, out_planes=16, kernel_size=2, stride=2, padding=0,output_padding=0)
    # trans_conv_layer = DepthwiseSeparableTransConv2d(in_planes=3, out_planes=16, kernel_size=2, stride=2, padding=0, output_padding=0)
    # x = torch.randn(1, 3, 32, 32)  # 随机生成形状为 (1, 3, 32, 32) 的张量
    # output = trans_conv_layer(x)
    # print("Output shape:", output.shape)