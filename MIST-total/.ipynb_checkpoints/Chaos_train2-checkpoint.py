#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from lib.networks import MIST_CAM

from trainer_mistchaos import trainer_chaos
from torchsummaryX import summary
from ptflops import get_model_complexity_info
from torchstat import stat
from lib.some_models import Unet,UnetPlusPlus
# from lib.Segnet import Segnet
from lib.vmunet import VMUNet
# from lib.bisenetv1 import BiSeNetV1
from lib.bisenetv2 import BiSeNetV2
# from lib.UNet_MobileNet import mobile_UNet
# In[2]:


import gc
gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'


# In[3]:


parser = argparse.ArgumentParser(description='Searching longest common substring. '
                    'Uses Ukkonen\'s suffix tree algorithm and generalized suffix tree. '
                    'Written by Ilya Stepanov (c) 2013')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching',)
parser.add_argument('--root_path', type=str,
                    default=r'/root/autodl-tmp/CHAOS_Train/Train_Sets/MR/', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='', help='root dir for validation volume data')
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
                    default=192, help='batch_size per gpu')
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
                    default='', help='model_dict_pretrain')
args = parser.parse_args("AAA".split())


# In[4]:


args


# In[ ]:


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True #自动寻找最优的算法来进行深度学习计算
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
        'Chaos': {
            'root_path': args.root_path,
            'num_classes': args.num_classes,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.is_pretrain = True

    args.exp = 'MISTCHAOS256' + dataset_name + str(args.img_size)
    snapshot_path = "model_pth/{}/{}".format(args.exp, 'MISTCHAOS256')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    current_time = time.strftime("%H%M%S")
    print("The current time is", current_time)
    snapshot_path = snapshot_path +'_run'+current_time

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # net = MERIT_Parallel_Modified3(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')
    # net = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')
    # if args.pretrain_weights_path:
    #     net.load_state_dict(torch.load(args.pretrain_weights_path), strict=False)
    #     print('model is load_dict',args.pretrain_weights_path)
    # print('Model %s created, param count: %d' %
    #                  ('MIST_CAM: ', sum([m.numel() for m in net.parameters()])))
    # net = net.cuda()

    #unet
    # net = Unet(in_channels=1,out_channels=5).cuda()
    # net = Segnet(1,5).cuda()
    net = mobile_UNet(n_channels=1, num_classes=5).cuda()
    # net = BiSeNetV1(n_classes = 5, aux_mode='train').cuda()
    # net = BiSeNetV2(n_classes=5,aux_mode='train').cuda()
   #mamba unet
    # net = VMUNet(input_channels = 3,num_classes=5,depths=[1,1,1,1], depths_decoder=[1,1,1,1],drop_path_rate=0.2,load_ckpt_path ='/root/MIST-total/model_pth/mamba_pretrain/vmamba_small_e238_ema.pth' ).cuda()
    # net.load_from()
    
    # macs, params = get_model_complexity_info(net, (3, args.img_size, args.img_size), as_strings=True,
    #                                        print_per_layer_stat=False, verbose=True)
    # print('macs:',macs)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    trainer = {'Chaos': trainer_chaos,}
    trainer[dataset_name](args, net, snapshot_path)





