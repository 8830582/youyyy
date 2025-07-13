#测试模型的dice系数
import torch
import torchvision
import torch.nn.functional as F
import argparse
import random
import os
import numpy as np
from utils.utils import dice_coefficient,calculate_iou_per_class
from utils.utils import test_single_volume1
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis
from utils.dataset_CHAOS_new import get_data_loader
from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from torch.utils.data import DataLoader
# from lib.some_models import Unet,UnetPlusPlus
# from lib.vmunet import VMUNet
# from lib.Segnet import Segnet
# from lib.bisenetv1 import BiSeNetV1
# from lib.bisenetv2 import BiSeNetV2
# from lib.erfnet import Net as erfnet
# from lib.UNet_MobileNet import mobile_UNet
from lib.segformer import SegFormer
from tqdm import tqdm
import matplotlib.colors as mcolors
from torchsummary import summary
from segmentation_mask_overlay import overlay_masks
# import torchprofile
from thop import profile
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='dice_test')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching',)
parser.add_argument('--root_path', type=str,
                    default=r'/root/autodl-tmp/CHAOS_Train/Train_Sets/MR/', help='root dir for data')
#chaos: '/root/autodl-tmp/CHAOS_Train/Train_Sets/MR/'
#Synapse:root_path:r'/root/autodl-tmp/train_npz'
#Synapse:volume_path: r'/root/autodl-tmp/test_vol_h5'
parser.add_argument('--volume_path', type=str,
                    default=r'/root/autodl-tmp/test_vol_h5', help='root dir for validation volume data')

#这里修改数据集
parser.add_argument('--dataset', type=str,
                    default='Chaos', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate') #0.001
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input') #224
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--pretrain_weights_path', type=str,
                    default='/root/MIST-total/model_pth/chaos/choas_mamba_wave_unet/best.pth', help='model_dict_pretrain')
# parser.add_argument('--pretrain_weights_path', type=str,
#                     default='/root/MIST-total/model_pth/Synapse/Synapse_m_w_unet/best.pth', help='model_dict_pretrain')

#'/root/MIST-total/model_pth/Synapse/Synapse_m_w_unet/best.pth'
#choas_mamba_wave_unet/best.pth

# unet:'/root/MIST-total/model_pth/chaos/chaos_Unet/best.pth'
args = parser.parse_args("AAA".split())



if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #model
    # mobile_net = Unet(in_channels=1,out_channels=5).cuda()
    # mobile_net = UnetPlusPlus(in_channels=1,num_classes=5, deep_supervision=False).to('cuda')
    # mobile_net = BiSeNetV1(n_classes = 5, aux_mode='train').to('cuda')
    # mobile_net = BiSeNetV2(n_classes=5,aux_mode='train').to('cuda')
    # mobile_net = mobile_UNet(n_channels=1, num_classes=5).cuda()
    # mobile_net=erfnet(num_classes=5,).cuda()
    # mobile_net = Segnet(1,5).to('cuda')
    mobile_net = SegFormer(num_classes = 5,phi = 'b0', pretrained = False).cuda()
    # mobile_net = VMUNet(input_channels = 3,num_classes=5,depths=[1,1,1,1], depths_decoder=[1,1,1,1],drop_path_rate=0.2,load_ckpt_path ='/root/MIST-total/model_pth/mamba_pretrain/vmamba_small_e238_ema.pth' ).cuda()
    
#------------------------information------------------------------
    summary(mobile_net, input_size=(1, 256, 256))
    # input_tensor = torch.randn(1, 1, 256, 256)
    dummy_input = torch.randn(1, 1, 256, 256).to('cuda')
    flops, params = profile(mobile_net, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
    # flops = FlopCountAnalysis(mobile_net.to('cpu'), input_tensor)
    # print(f"FLOPs: {flops.total()}")
    # flops = torchprofile.profile_macs(mobile_net.to('cpu'), input_tensor)
    # print(f"FLOPs: {flops}")
    
    # Computational complexity and Number of parameters
    macs, params = get_model_complexity_info(mobile_net, (1, args.img_size, args.img_size), as_strings=True,print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))










#计算FPS 非优势区域，不适用
    import time
    input_tensor = torch.randn(1, 1, 256, 256).cuda()
    
    # Warm-up 运行，确保 GPU 加载完成，避免冷启动影响测量结果
    with torch.no_grad():
        for _ in range(10):
            mobile_net(input_tensor)
    # 开始计时
    start_time = time.time()
    
    # 设定推理次数
    num_iterations = 100
    
    # 进行多次推理
    with torch.no_grad():
        for _ in range(num_iterations):
            mobile_net(input_tensor)
    
    # 结束计时
    end_time = time.time()
    
    # 计算总时间
    total_time = end_time - start_time
    
    # 计算 FPS
    fps = num_iterations / total_time
    
    print(f"Model FPS: {fps:.2f}")
