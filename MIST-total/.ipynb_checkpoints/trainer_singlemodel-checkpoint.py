# 此为原版

import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.utils import powerset
from utils.utils import one_hot_encoder
from utils.utils import DiceLoss
from utils.utils import val_single_volume,val_single_volume_1out
#辅助组件
from lr_scheduler import WarmupPolyLrScheduler
from ohem_ce_loss import OhemCELoss


def inference(args, model, best_performance):
    if hasattr(model, 'aux_mode'):
        model.aux_mode = 'eval'
    db_test = Synapse_dataset(base_dir=args.volume_path, split="val_vol", list_dir=args.list_dir,
                              nclass=args.num_classes)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume_1out(image, label, model, classes=args.num_classes,
                                     patch_size=[args.img_size, args.img_size],
                                     case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance


def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(4)]
    
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    # iterator = tqdm(range(max_epoch), ncols=70)
    lc1, lc2 = 0.3, 0.7  # 0.3, 0.7
    lr_list = [0.15,0.25,0.6]
    # ss = [[0],[1],[2],[3]]

        ## lr scheduler
    # lr_schdr = WarmupPolyLrScheduler(optimizer, power=0.9,max_iter=max_epoch, warmup_iter=1000,warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    
    for epoch_num in range(args.max_epochs):
        loss = 0.0
        if hasattr(model, 'aux_mode'):
            model.aux_mode = 'train'
        for i_batch, sampled_batch in enumerate(trainloader):
            
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            # print('i_batch的shape:', image_batch.shape)
            # print(image_batch.shape)

            
            logits, *logits_aux = model(image_batch)
            #主要损失
            loss_ce = criteria_pre(logits, label_batch)
            loss_dice = dice_loss(logits, label_batch, softmax=True)
            #辅助损失
            loss_aux = [crit(lgt, label_batch) for crit, lgt in zip(criteria_aux, logits_aux)]
            #总损失
            loss = lc1 * loss_ce + lc2 * loss_dice + sum(loss_aux)

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))

        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))

        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)

        performance = inference(args, model, best_performance)

        # save_interval = 100

        if (best_performance <= performance):
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
        # 隔一阵epoch保存一次权重
        # if (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            break

    writer.close()
    return "Training Finished!"
