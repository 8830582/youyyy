import torch
import torch.nn as nn
import torch.fft


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
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0,
                 bias=False):
        super(TransBasicConv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding,
                                         dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
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

class FFTModule2(nn.Module):#双输入模式
    def __init__(self, inchannel, outchannel):
        super(FFTModule, self).__init__()
        self.conv1 = BasicConv2d(inchannel, outchannel)
        self.conv2 = BasicConv2d(inchannel, inchannel)
        self.conv3 = BasicConv2d(outchannel, outchannel)
        self.change = TransBasicConv2d(outchannel, outchannel)

        # self.conv1 = DepthwiseSeparableConv2d(inchannel,outchannel)
        # self.conv2 = DepthwiseSeparableConv2d(inchannel, inchannel)
        # self.conv3 = DepthwiseSeparableConv2d(outchannel, outchannel)
        # self.change = DepthwiseSeparableTransConv2d(outchannel,outchannel)

    def forward(self, x, y):
        y = self.conv2(y)

        # Apply FFT
        X_fft = torch.fft.fftn(x, dim=(-2, -1))
        Y_fft = torch.fft.fftn(y, dim=(-2, -1))

        # Combine the real and imaginary parts separately
        X_fft_combined = torch.view_as_real(X_fft)
        Y_fft_combined = torch.view_as_real(Y_fft)

        Xl = X_fft_combined[..., 0]
        Yl = Y_fft_combined[..., 0]
        Xh = X_fft_combined[..., 1]
        Yh = Y_fft_combined[..., 1]
        print('xl,xh:',Xl.shape,Xh.shape)
        print('Yl,Yh:',Yl.shape,Yh.shape)

        x_y_real = self.conv1(Xl) + self.conv1(Yl)
        x_y_imag = self.conv1(Xh) + self.conv1(Yh)

        x_y_combined = torch.view_as_complex(torch.stack((x_y_real, x_y_imag), dim=-1))

        # Apply inverse FFT
        x_m = torch.fft.ifftn(x_y_combined, dim=(-2, -1)).real
        y_m = torch.fft.ifftn(x_y_combined, dim=(-2, -1)).real

        out = self.conv3(x_m + y_m)
        return out

class FFTModule(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(FFTModule, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(inchannel, outchannel)
        self.conv2 = DepthwiseSeparableConv2d(inchannel, inchannel)
        self.conv3 = DepthwiseSeparableConv2d(outchannel, outchannel)
        self.change = DepthwiseSeparableTransConv2d(outchannel, outchannel)

    def forward(self, x):
        x = self.conv2(x)

        # Apply FFT
        X_fft = torch.fft.fftn(x, dim=(-2, -1))

        # Combine the real and imaginary parts separately
        X_fft_combined = torch.view_as_real(X_fft)

        Xl = X_fft_combined[..., 0]
        Xh = X_fft_combined[..., 1]
        # print('Xl, Xh:', Xl.shape, Xh.shape)

        x_real = self.conv1(Xl)
        x_imag = self.conv1(Xh)

        x_combined = torch.view_as_complex(torch.stack((x_real, x_imag), dim=-1))

        # Apply inverse FFT
        x_m = torch.fft.ifftn(x_combined, dim=(-2, -1)).real

        out = self.conv3(x_m)
        return out


# 测试 FFTModule
if __name__ == "__main__":
    ########### 双输入模式#############
    # 创建模型实例
    # model = FFTModule2(inchannel=3, outchannel=16)
    # # 模拟输入数据
    # x = torch.randn(1, 3, 256, 256)  # 输入张量 x
    # y = torch.randn(1, 3, 256, 256)  # 输入张量 y
    # # 运行前向传播
    # output = model(x, y)
    # print("Output shape:", output.shape)

    ########### 单输入模式#############
    # 创建模型实例
    model = FFTModule(inchannel=3, outchannel=3)
    # 模拟输入数据
    x = torch.randn(1, 3, 256, 256)  # 输入张量 x
    # 运行前向传播
    output = model(x)
    print("Output shape:", output.shape)