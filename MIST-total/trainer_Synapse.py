import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
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

            
def inference(args, model, best_performance):
    if hasattr(model, 'aux_mode'):
        model.aux_mode = 'eval'
    db_test = Synapse_dataset(base_dir=args.volume_path, split="val_vol", list_dir=args.list_dir, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume_1out(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def trainer_synapse(args,teacher,student,snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    batch_size = args.batch_size
    print('batch_size:',args.batch_size)
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        student = nn.DataParallel(student)

    student.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(student.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    #iterator = tqdm(range(max_epoch), ncols=70)
    
    l = [0, 1, 2, 3]
    ss = [x for x in powerset(l)]
    #ss = [[0],[1],[2],[3]]
    print(ss)

    for epoch_num in range(args.max_epochs):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()

            with torch.no_grad():  # The teacher network does not use backpropagation
                techer_preds = teacher(image_batch)
            # student model forward
            student_preds = student(image_batch)
            student_loss = 0
            #hard loss
            lc1, lc2 = 0.4, 0.6  # 0.4, 0.6 Weights for both loss functions
            weights = [1.0,0.3,0.5]#Weights for the 3 outputs
            for i in range(len(student_preds)):
                try:
                    loss_ce = ce_loss(student_preds[i],label_batch[:].long())
                    loss_dice = dice_loss(student_preds[i], label_batch, softmax=True)
                except Exception:
                    print('image_batch:',student_preds[i].shape)
                    print('label_batch[:]:',label_batch[:].shape)
                student_loss +=weights[i]*(lc1 * loss_ce + lc2 * loss_dice)
            student_loss = student_loss/3
            
            total_loss = 0.0

            temp = 1.0  # temperature
            alpha = 0.3 #Total loss weight

            teach_iout= 0.0    
            for idx in range(len(techer_preds)):
                teach_iout = teach_iout+techer_preds[idx]

            # out = torch.argmax(torch.softmax(teach_iout, dim=1), dim=1).squeeze(0)
            print("teach_iout.shape:",teach_iout.shape)

            temp = torch.tensor(temp)
            # teach_iout = torch.tensor(teach_iout)
            #soft_loss
            # ditillation_loss = ce_loss(student_preds/temp,teach_iout/temp,)
            # print("student_preds,teach_iout:",type(student_preds),type(teach_iout))
            # print("student_preds,teach_iout:",student_preds.shape,teach_iout.shape)
            ditillation_loss = ce_loss(
            F.log_softmax(student_preds[0]/temp, dim = 1,),
            F.softmax(teach_iout/temp, dim = 1))

            total_loss =(1 - alpha) * student_loss +alpha * ditillation_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', total_loss, iter_num)
            

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, total_loss.item(), lr_))                
             
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, total_loss.item(), lr_))        
       
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(student.state_dict(), save_mode_path)
        performance = inference(args, student, best_performance)
        
        if(best_performance <= performance):
           best_performance = performance
           save_mode_path = os.path.join(snapshot_path, 'best.pth')
           torch.save(student.state_dict(), save_mode_path)
           logging.info("save student to {}".format(save_mode_path))

        save_interval = 100
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(student.state_dict(), save_mode_path)
            logging.info("save student to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(student.state_dict(), save_mode_path)
            logging.info("save student to {}".format(save_mode_path))
            break
    writer.close()
    return "Training Finished!"
