#!/usr/bin/env python
# coding: utf-8
#单独训练小网络
# In[1]:
# train MIST

import argparse
import logging
import os
import time
import random
import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn

from lib.networks import MIST_CAM
from lib.UNet_MobileNet import mobile_UNet
# from lib.bisenetv1 import BiSeNetV1
# from lib.bisenetv2 import BiSeNetV2
# from lib.some_models import Unet,UnetPlusPlus
# from lib.vmunet import VMUNet
# from lib.Segnet import Segnet
# from lib.erfnet import Net as erfnet
# from lib.bisenet_total import BiSeNet_total
# from lib.bisenet_mp_w import BiSeNet_total
# from lib.dfanet import DFANet
from lib.segformer import SegFormer
from lib.bisenet_onemamba import BiSeNet_onemamba
from trainer_singlemodel2 import trainer_synapse

from torchsummaryX import summary
from ptflops import get_model_complexity_info
from torchstat import stat


import gc

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

# In[3]:


parser = argparse.ArgumentParser(description='Searching longest common substring. '
                                             'Uses Ukkonen\'s suffix tree algorithm and generalized suffix tree. '
                                             'Written by Ilya Stepanov (c) 2013')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching', )
parser.add_argument('--root_path', type=str,
                    default=r'/root/autodl-tmp/train_npz', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default=r'/root/autodl-tmp/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iterations number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=48, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')  # 0.001
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')  # 224
parser.add_argument('--seed', type=int,
                    default=3407, help='random seed')
# parser.add_argument('--pretrain_minimodel_weights_path', type=str,
#                     default='', help='MIST_model_dict_pretrain')
parser.add_argument('--load_path', type=str,default='')
# '/root//MIST-total/model_pth/modile_unetSynapse256/modile_unet_pretrain_epo100_bs32_lr0.0001_256_s3407_run165853/best.pth'
#/home/jovyan/work/MIST/MISTmain240523/model_pth/MIST_CAM_loss_MUTATION_w3_7_Synapse256/Segnet_pretrain_epo300_bs64_lr0.0001_256_s2222_run081203/ast.pth', help='mobildeUnet_model_dict_pretrain
args = parser.parse_args("SegFormer".split())

# In[4]:


# args


# In[ ]:


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.deterministic:
        cudnn.benchmark = True  # 自动寻找最优的算法来进行深度学习计算
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    args.exp = 'SegFormer' + dataset_name + str(args.img_size)
    snapshot_path = "model_pth/{}/{}".format(args.exp, 'SegFormer')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    current_time = time.strftime("%H%M%S")
    print("The current time is", current_time)
    snapshot_path = snapshot_path + '_run' + current_time

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # 加载网络MIST网络  
    # net = MERIT_Parallel_Modified3(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')
    # MISTnet = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size, args.img_size), img_size_s2=(224, 224),
    #                    model_scale='tiny', decoder_aggregation='additive', interpolation='bilinear')
    # if args.pretrain_weights_path:
    #     MISTnet.load_state_dict(torch.load(args.pretrain_weights_path), strict=False)
    #     print('model is load_dict', args.pretrain_weights_path)
    # print('Model %s created, param count: %d' %
    #       ('MIST_CAM: ', sum([m.numel() for m in MISTnet.parameters()])))

    # MISTnet = MISTnet.cuda()

    # 加载mobile_UNet网络 在这里更换网络
    # mobile_net = mobile_UNet(n_channels=1, num_classes=5)

    # mobile_net = UnetPlusPlus(in_channels=1,num_classes=5, deep_supervision=False)

    # mobile_net = Unet(in_channels=1,out_channels=5)
    # mobile_net = BiSeNetV1(n_classes = 5, aux_mode='train')
    # mobile_net = BiSeNetV2(n_classes=5,aux_mode='train')
    # mobile_net = BiSeNet_total(n_classes=5,aux_mode='train')
    # mobile_net = BiSeNet_onemamba(n_classes=5,aux_mode='train')
    # mobile_net = Segnet(1,5)
    # mobile_net=erfnet(num_classes=5,)
    # mobile_net = VMUNet(input_channels = 3,num_classes=5,depths=[2,2,2,2], depths_decoder=[2,2,2,1],drop_path_rate=0.2,load_ckpt_path ='/root/MIST-total/model_pth/mamba_pretrain/vmamba_small_e238_ema.pth' )
    # mobile_net = VMUNet(input_channels = 3,num_classes=5,depths=[1,1,1,1], depths_decoder=[1,1,1,1],drop_path_rate=0.2,load_ckpt_path ='/root/MIST-total/model_pth/mamba_pretrain/vmamba_small_e238_ema.pth' )
    # mobile_net.load_from()
    #dfanet
    # ch_cfg=[[8,48,96],
    #     [240,144,288],
    #     [240,144,288]]
    # mobile_net=DFANet(ch_cfg,64,5)
    #segformer
    mobile_net = SegFormer(num_classes = 5,phi = 'b2', pretrained = False)
    
    # mobile_net = UNet(n_channels=1, num_classes=args.num_classes)
    # 是否对mobile_net导入预训练权重进行迁移学习
    if args.load_path:
        # model_dict = mobile_net.state_dict()
        # model_path = args.load_path
        # pretrained_dict = torch.load(model_path, map_location=device)
        mobile_net.load_state_dict(torch.load(args.load_path), strict=True)
        print("{} is loaded".format(args.load_path))
        # 筛除不加载的层结构
        # pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        # # 更新当前网络的结构字典
        # model_dict.update(pretrained_dict)
        # model_dict.load_state_dict(model_dict)
        # logging.info(f'Model loaded from {args.load}')
    mobile_net.cuda()

    macs, params = get_model_complexity_info(mobile_net, (1, args.img_size, args.img_size), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    trainer = {'Synapse': trainer_synapse, }
    trainer[dataset_name](args,mobile_net, snapshot_path)  # 第一个是教师网络，第二个是学生网络





