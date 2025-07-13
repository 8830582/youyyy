# import albumentations as A
import torch
import math
import random
import os
import cv2
import shutil
import numpy as np
import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.nn.functional as F
import pydicom

idx={#种类与类别
    '0':0,
    '63':1,
    '126':2,
    '189':3,
    '252':4
}
def pixel_to_Id(array):
    ix,jx=array.shape
    array=array.astype(str)
    for i in range(ix):
        for j in range(jx):
            pixel=array[i][j]
            pixelid=idx[pixel]
            array[i][j]=pixelid
    array=array.astype("int32")
    return array
    

def get_MRdata(path):
    mr_img_path = []
    mr_mask_path = []
    # path='./CHAOS_Train/Train_Sets/MR/'
    dirs1 = glob.glob(path + '*')
    for i in dirs1:
        MR_img_dir = i + '/T2SPIR/DICOM_anon'
        MR_mask_dir = i + '/T2SPIR/Ground'
        for file in os.listdir(MR_img_dir):
            img = MR_img_dir + '/{}'.format(file)
            mr_img_path.append(img)
        for file in os.listdir(MR_mask_dir):
            mask = MR_mask_dir + '/{}'.format(file)
            mr_mask_path.append(mask)
            mr_img_path.sort()
            mr_mask_path.sort()
            # print(mr_img_path[:5],mr_mask_path[:5])
    return mr_img_path, mr_mask_path

class coCHAOSDataset(Dataset):
    def __init__(self,img,mask,transformer,label_tranformer):
        self.img=img
        self.mask=mask
        self.transformer=transformer
        self.label_tranformer=label_tranformer
    def __getitem__(self,index):
        img=self.img[index]
        mask=self.mask[index]

        img_open=pydicom.read_file(img)
        img_arrayR=img_open.pixel_array
        img_arrayR = np.array(img_arrayR, dtype=np.float32)
        ###读取为PIL
        img_arrayPIC=Image.fromarray(img_arrayR)
        #转换resize
        img_tensor=self.transformer(img_arrayPIC)##resize
        ###
        ###读取图片
        mask_open=Image.open(mask)
        mask_array=np.array(mask_open)        
        ###矩阵像素转label
        mask_pixel_to_id=pixel_to_Id(mask_array)        
        ###读取为PIL
        mask_label=Image.fromarray(mask_pixel_to_id)
        ##reisze
        mask_label=self.label_tranformer(mask_label)
        mask_label=np.array(mask_label)
        #numpy tensor
        mask_tensor=torch.from_numpy(mask_label)
        
        mask_tensor=torch.squeeze(mask_tensor).type(torch.long)
        return img_tensor,mask_tensor
    
    def __len__(self):
        return len(self.img)
        
def get_data_loader(path, batch_size=4,train_tag='train',shuffle=True,img_size = 256,num_workers=0,num_limit = 500) -> torch.utils.data.DataLoader:
    """
        Build the train loader. This is a standard loader (not episodic)
    """
    # path = '../../lists/train-kvasir/'
    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
    ])
    label_transformer = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
    ])
            # transforms.ToTensor(),
    mr_img_path, mr_mask_path = get_MRdata(path)
    assert len(mr_img_path) == len(mr_mask_path), \
    f"The number of MRI images ({len(mr_img_path)}) does not match the number of masks ({len(mr_mask_path)})."
    num_limit = num_limit
    if train_tag == 'train':
        length = len(mr_img_path)
        print("The length of train set is: {}".format(len(mr_img_path[:num_limit])))
        data = coCHAOSDataset(mr_img_path[:num_limit], mr_mask_path[:num_limit], image_transformer,label_transformer)
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 pin_memory=False)
    elif train_tag == 'val':
        length = len(mr_img_path)
        print("The length of val set is: {}".format(len(mr_img_path[num_limit:])))
        data = coCHAOSDataset(mr_img_path[num_limit:], mr_mask_path[num_limit:], image_transformer,label_transformer)
        # data = coCHAOSDataset(mr_img_path, mr_mask_path, image_transformer,label_transformer, train_tag=True)
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 pin_memory=False)
    else:
        length = len(mr_img_path)
        random_number = random.randint(0, 300)
        print("The length of test set is: {}".format(len(mr_img_path)))
        data = coCHAOSDataset(mr_img_path, mr_mask_path, image_transformer,label_transformer)
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 pin_memory=False)
    return dataloader,length


if __name__ == '__main__':
    train_path = r"/root/autodl-tmp/CHAOS_Train/Train_Sets/MR/"
    dl_train = get_data_loader(path=train_path,batch_size=4,train_tag='train',shuffle=True,img_size = 256,num_workers=2)
    # dl_test = get_data_loader(path=train_path,batch_size=4,train_tag='val',)
    image,label= next(iter(dl_train))
    print("train_set:",image.shape, label.shape)
    print(print("image tensor_float type:", image.dtype))
    print(print("label tensor_float type:", label.dtype))