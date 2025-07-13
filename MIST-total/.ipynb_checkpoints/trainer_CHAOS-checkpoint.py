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
# from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.dataset_CHAOS import get_data_loader
from utils.utils import powerset
from utils.utils import one_hot_encoder
from utils.utils import DiceLoss
from medpy import metric
from utils.utils import val_single_volume,val_single_volume_1out

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0
        
def inference(args, model, best_performance):
    
    testloader = get_data_loader(path=args.root_path,batch_size=args.batch_size,train_tag='train',shuffle=True,img_size = args.img_size,num_workers=2)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, (image_batch,label_batch) in tqdm(enumerate(testloader)):
        image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(0).cpu().detach().numpy()
        # print("in val：",label_batch.shape)
        with torch.no_grad():
            outputs = model(image_batch)
            print('之前两个shape:',outputs.shape,label_batch.shape)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
        print('之后两个shape:',prediction.shape,label_batch.shape)
        metric_i = []
        for i in range(1, args.num_classes):
            metric_i.append(calculate_dice_percase(prediction == i, label_batch == i))
        
        metric_list += np.array(metric_i)
        metric_list = metric_list / len(testloader)
        
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance


def trainer_chaos(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    trainloader = get_data_loader(path=args.root_path,batch_size=args.batch_size,train_tag='train',shuffle=True,img_size = args.img_size,num_workers=2)

    print("The length of train set is: {}".format(len(trainloader)))

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                          worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

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
    
    for epoch_num in range(args.max_epochs):
        model.train()
        for i_batch,(image_batch,label_batch) in enumerate(trainloader):
            # image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            # print('i_batch的shape:', image_batch.shape)
            # print(image_batch.shape)
            iout = model(image_batch)
            loss = 0.0
            

            loss_ce = ce_loss(iout, label_batch[:].long())
            loss_dice = dice_loss(iout, label_batch, softmax=True)
            loss += (lc1 * loss_ce + lc2* loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
