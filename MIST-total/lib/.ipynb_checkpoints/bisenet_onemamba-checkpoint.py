#bisenetv3带有一层segment——mamba块，无位置编码，无fft与walvet，
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

from lib.models_mamba import create_block
from lib.wavelet import DWT
from lib.fft import FFTModule
# from models_mamba import create_block
# from wavelet import DWT
# from fft import FFTModule

backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_planes, bias=False)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)

class mamba_block(nn.Module):
    def __init__(self,C):
        super(mamba_block, self).__init__()
        self.mamba = create_block(d_model=C)
    def forward(self, x):
        hidden_states,residual = self.mamba(x)
        return hidden_states

class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 32, 3, stride=2),
            ConvBNReLU(32, 32, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(32, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )
        # self.mamba1 =mamba_block(32)
        # self.mamba2 =mamba_block(64)
        # self.mamba3 =mamba_block(128)
    def forward(self, x):
        # print("detailed breach x:",x.shape)
        # feat1 = self.S1(x)
        # B, C, H, W = feat1.shape
        # feat1 = feat1.view(B, C, H * W).permute(0, 2, 1)
        # # print('feat1.shape:',feat1.shape)
        # feat1 = self.mamba1(feat1)
        # feat1 = feat1.permute(0, 2, 1).view(B, C, H, W)
        # # print("detailed breach feat:", feat1.shape)
        # feat2 = self.S2(feat1)
        # B, C, H, W = feat2.shape
        # feat2 = feat2.view(B, C, H * W).permute(0, 2, 1)
        # feat2 = self.mamba2(feat2)
        # feat2 = feat2.permute(0, 2, 1).view(B, C, H, W)
        # # print("detailed breach feat2:", feat.shape)
        # feat3 = self.S3(feat2)
        # B, C, H, W = feat3.shape
        # feat3 = feat3.view(B, C, H * W).permute(0, 2, 1)
        # feat3 = self.mamba3(feat3)
        # feat3 = feat3.permute(0, 2, 1).view(B, C, H, W)
        # print("detailed breach feat3:", feat.shape)       
        
        #无mamba分支
        feat1 = self.S1(x)
        # print("detailed breach feat1:", feat1.shape) 
        feat2 = self.S2(feat1)
        # print("detailed breach feat2:", feat2.shape) 
        feat3 = self.S3(feat2)
        # print("detailed breach feat3:", feat3.shape) 
        return feat1,feat2,feat3


class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 32, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)#1,3,256,256,128 ——>1,16,128,128
        feat_left = self.left(feat)#1,16,128,128 ->
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        # print('feat shape',feat.shape)
        # print('x shape',x.shape)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        # print("SegmentBranch x shape:",x.shape)
        feat2 = self.S1S2(x) #([1, 32, 64, 64]
        # print("SegmentBranch feat2 shape:", feat2.shape)
        feat3 = self.S3(feat2) #([1, 64, 32, 32]
        # print("SegmentBranch feat3 shape:", feat3.shape)
        feat4 = self.S4(feat3) #
        # print("SegmentBranch feat4 shape:", feat4.shape)
        feat5_4 = self.S5_4(feat4)
        # print("SegmentBranch feat5_4 shape:", feat5_4.shape)
        feat5_5 = self.S5_5(feat5_4)
        # print("SegmentBranch feat5_5 shape:", feat5_5.shape)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out



class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat2 = self.conv_out(feat)
        return feat2


class BiSeNet_onemamba(nn.Module):

    def __init__(self, n_classes, aux_mode='train'):
        super(BiSeNet_onemamba, self).__init__()
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        ## TODO: what is the number of mid chan ?
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        if self.aux_mode == 'train':
            self.aux2 = SegmentHead(32, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(64, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(128, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
        self.init_weights()

#segment branch
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()


        # self.FFT1 = FFTModule(inchannel=32, outchannel=32)
        # self.FFT2 = FFTModule(inchannel=64, outchannel=64)
        # self.FFT3 = FFTModule(inchannel=128, outchannel=128)

        # self.DWT1 = DWT(inchannel=32, outchannel=32)
        # self.DWT2 = DWT(inchannel=64, outchannel=64)
        # self.DWT3 = DWT(inchannel=128, outchannel=128)

        # self.dewconv1 = DepthwiseSeparableConv2d(32, 32, 3, stride=2)
        # self.dewconv2 = DepthwiseSeparableConv2d(64, 64, 3, stride=2)
        # self.dewconv3 = DepthwiseSeparableConv2d(128, 128, 3, stride=2)

        # self.br1 = nn.Sequential(
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
        # self.br2 = nn.Sequential(
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.br3 = nn.Sequential(
        #     nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )
        self.mamba1 =mamba_block(32)
        self.mamba2 =mamba_block(64)
        self.mamba3 =mamba_block(128)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        size = x.size()[2:]
#detail branch
        d_feat1,d_feat2,d_feat3 = self.detail(x)
        # print('d_feat1,d_feat2,d_feat3',d_feat1.shape,d_feat2.shape,d_feat3.shape)

#example##########fuse1################
        # feat2 = self.FFT1(feat2)
        # fuse1 = self.DWT1(d_feat1,feat2)
        # fuse1 = self.dewconv1(fuse1)
        # feat2 = fuse1+feat2
        # feat2 = self.br1(feat2)
        #############mamba1##########
        feat2 = self.S1S2(x) #([1, 32, 64, 64]
        B, C, H, W = feat2.shape
        feat2 = feat2.view(B, C, H * W).permute(0, 2, 1)
        feat2=self.mamba1(feat2)
        feat2 = feat2.permute(0, 2, 1).view(B, C, H, W)
        ############mamba2################
        feat3=self.S3(feat2)
        B, C, H, W = feat3.shape
        feat3 = feat3.view(B, C, H * W).permute(0, 2, 1)
        feat3=self.mamba2(feat3)
        feat3 = feat3.permute(0, 2, 1).view(B, C, H, W)
        ################mamba3#######################
        feat4=self.S4(feat3)
        B, C, H, W = feat4.shape
        feat4 = feat4.view(B, C, H * W).permute(0, 2, 1)
        feat4=self.mamba3(feat4)
        feat4 = feat4.permute(0, 2, 1).view(B, C, H, W)
        ##########
        feat5_4 = self.S5_4(feat4)
        
        feat_s = self.S5_5(feat5_4)

        # feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        # print('feat2, feat3, feat4, feat5_4, feat_s', feat2.shape, feat3.shape, feat4.shape, feat5_4.shape, feat_s.shape)

#fused branch
        # print('fuse1,fuse2,fuse3',fuse1.shape,fuse2.shape,fuse3.shape)
        feat_head = self.bga(d_feat3, feat_s)




        logits = self.head(feat_head)
        if self.aux_mode == 'train':
            # logits_aux2 = self.aux2(feat2)
            # logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            # return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
            return logits,logits_aux4, logits_aux5_4
        elif self.aux_mode == 'eval':
            return logits
        elif self.aux_mode == 'pred':
            pred = logits.argmax(dim=1)
            return pred
        else:
            raise NotImplementedError

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # self.load_pretrain()


    def load_pretrain(self):
        state = modelzoo.load_url(backbone_url)
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=False)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params



if __name__ == "__main__":
    #  x = torch.randn(16, 3, 1024, 2048)
    #  detail = DetailBranch()
    #  feat = detail(x)
    #  print('detail', feat.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  stem = StemBlock()
    #  feat = stem(x)
    #  print('stem', feat.size())
    #
    #  x = torch.randn(16, 128, 16, 32)
    #  ceb = CEBlock()
    #  feat = ceb(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 32, 16, 32)
    #  ge1 = GELayerS1(32, 32)
    #  feat = ge1(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 16, 16, 32)
    #  ge2 = GELayerS2(16, 32)
    #  feat = ge2(x)
    #  print(feat.size())
    #
    #  left = torch.randn(16, 128, 64, 128)
    #  right = torch.randn(16, 128, 16, 32)
    #  bga = BGALayer()
    #  feat = bga(left, right)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 128, 64, 128)
    #  head = SegmentHead(128, 128, 19)
    #  logits = head(x)
    #  print(logits.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  segment = SegmentBranch()
    #  feat = segment(x)[0]
    #  print(feat.size())
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 256, 256).to(device)

    model = BiSeNet_total(n_classes=5,aux_mode='eval').to(device)
    model.eval()
    outs = model(x)
    print('output:',outs.shape)
    # for out in outs:
    #     print(out.size())
    # print(logits.size())

    #  for name, param in model.named_parameters():
    #      if len(param.size()) == 1:
    #          print(name)
