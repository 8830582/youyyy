#!/usr/bin/env python
# coding: utf-8

# In[1]:
# train MIST

import argparse
import logging
import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from lib.networks import MIST_CAM
from lib.vmunet import VMUNet

from trainer_Synapse import trainer_synapse

from torchsummaryX import summary
from ptflops import get_model_complexity_info
from torchstat import stat

# In[2]:


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
                    default=r'/home/jovyan/work/MIST/train_npz', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default=r'/home/jovyan/work/MIST/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iterations number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')  # 0.001
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')  # 224
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--pretrain_weights_path', type=str,
                    default='/home/jovyan/work/MIST/MIST-main240523/model_pth/best_mist_pth/best.pth', help='MIST_model_dict_pretrain')
parser.add_argument('--load', type=str,
                    default='', help='mobildeUnet_model_dict_pretrain')
args = parser.parse_args("AAA".split())

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
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

    args.exp = 'test_MIST_w3_7_' + dataset_name + str(args.img_size)
    snapshot_path = "model_pth/{}/{}".format(args.exp, 'mobile_UNet')
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
    # Loading the MIST network
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    #teacher net
    # net = MERIT_Parallel_Modified3(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')
    MISTnet = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size, args.img_size), img_size_s2=(224, 224),model_scale='tiny', decoder_aggregation='additive', interpolation='bilinear')
    if args.pretrain_weights_path:
        MISTnet.load_state_dict(torch.load(args.pretrain_weights_path), strict=False)
        print('model is load_dict', args.pretrain_weights_path)

    print('Model %s created, param count: %d' %
          ('MIST_CAM: ', sum([m.numel() for m in MISTnet.parameters()])))

    MISTnet = MISTnet.cuda()
    # student_model
    mobile_net = VMUNet(input_channels = 3,num_classes=5,depths=[1,1,1,1], depths_decoder=[1,1,1,1],drop_path_rate=0.2,load_ckpt_path ='/root/MIST-total/model_pth/mamba_pretrain/vmamba_small_e238_ema.pth' ).cuda()
    # Whether to import pre-trained weights for transfer learning for student_model
    if args.load:
        model_dict = mobile_net.state_dict()
        model_path = args.load
        pretrained_dict = torch.load(model_path, map_location=device)
            # Filter out layer structures that are not loaded
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        # Update the structure dictionary of the current network
        model_dict.update(pretrained_dict)
        model_dict.load_state_dict(model_dict)
        print(f'Model loaded from',args.load)
    mobile_net.cuda()
    macs, params = get_model_complexity_info(mobile_net, (1, args.img_size, args.img_size), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity:',macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    trainer = {'Synapse': trainer_synapse, }
    trainer[dataset_name](args, MISTnet, mobile_net, snapshot_path)





