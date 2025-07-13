import albumentations as A
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

# def getCHAOSdata(path):
#     path1 = path +'CT/'
#     dirs1=glob.glob(path1+'*')
#     mr_img_path=[]
#     mr_mask_path=[]
#     for i in dirs1:
#         ct_img_dir=i+'/DICOM_anon'
#         ct_mask_dir=i+'/Ground'
#         for file in os.listdir(ct_img_dir):
#             img=ct_img_dir+'/{}'.format(file)
#             mr_img_path.append(img)
#         for file in os.listdir(ct_mask_dir):
#             mask=ct_mask_dir+'/{}'.format(file)
#             mr_mask_path.append(mask)
#     return mr__path,mr_mask_path

idx={#种类与类别
    '0':0,
    '63':1,
    '126':2,
    '189':3,
    '252':4
}


def list_to_tuple(data):

    if isinstance(data, list):
        return tuple(list_to_tuple(x) for x in data)
    elif isinstance(data, dict):
        return {key: list_to_tuple(value) for key, value in data.items()}
    else:
        return data


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
    def __init__(self,
                 img, mask,
                 img_transformer,
                 label_tranformer,
                 train_tag=True,
                 normalize=False):
        self.img = img
        self.mask = mask
        self.img_transformer = img_transformer
        self.label_tranformer = label_tranformer
        self.train_tag = train_tag
        self.idx = {0: 0, 63: 1, 126: 2, 189: 3, 252: 4}
    def __len__(self):
        return len(self.img)
    def make_odd(self, num):
        num = math.ceil(num)
        if num % 2 == 0:
            num += 1
        return num
    def med_augment(self, img, mask, level, number_branch, mask_i=False):
        strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]
        # transform = A.Compose([
        #     A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        #     A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        #     A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        #     A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        #     A.GaussianBlur(blur_limit=(3, self.make_odd(3 + 0.8 * level)), p=0.2 * level),
        #     A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        #     A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None,
        #              rotate_method='largest_box',
        #              crop_border=False, p=0.2 * level),
        #     A.HorizontalFlip(p=0.2 * level),
        #     A.VerticalFlip(p=0.2 * level),
        #     A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
        #              shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
        #              keep_ratio=True, p=0.2 * level),
        #     A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
        #              shear={'x': (0, 2 * level), 'y': (0, 0)}
        #              , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
        #              keep_ratio=True, p=0.2 * level),  # x
        #     A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
        #              shear={'x': (0, 0), 'y': (0, 2 * level)}
        #              , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
        #              keep_ratio=True, p=0.2 * level),
        #     A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None,
        #              rotate=None,
        #              shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
        #              keep_ratio=True, p=0.2 * level),
        #     A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None,
        #              rotate=None,
        #              shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
        #              keep_ratio=True, p=0.2 * level)])
        transform = [A.ColorJitter(brightness=list_to_tuple([0.04 * level, 0.04 * level]), contrast=list_to_tuple([0, 0]), saturation=list_to_tuple([0, 0]), hue=list_to_tuple([0, 0]), p=0.2 * level),
                     A.ColorJitter(brightness=list_to_tuple([0, 0]), contrast=list_to_tuple([0.04 * level, 0.04 * level]), saturation=list_to_tuple([0, 0]), hue=list_to_tuple([0, 0]), p=0.2 * level),
                     A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
                     A.Sharpen(alpha=list_to_tuple([0.04 * level, 0.1 * level]), lightness=list_to_tuple([1, 1]), p=0.2 * level),
                     A.GaussianBlur(blur_limit=list_to_tuple([3, self.make_odd(3 + 0.8 * level)]), p=0.2 * level),
                     A.GaussNoise(var_limit=list_to_tuple([2 * level, 10 * level]), mean=0, per_channel=True, p=0.2 * level),
                     A.Rotate(limit=list_to_tuple([4 * level, 4 * level]), interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box', crop_border=False, p=0.2 * level),
                     A.HorizontalFlip(p=0.2 * level),
                     A.VerticalFlip(p=0.2 * level),
                     A.Affine(scale=list_to_tuple([1 - 0.04 * level, 1 + 0.04 * level]), translate_percent=None, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
                     A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear={'x': list_to_tuple([0, 2 * level]), 'y': list_to_tuple([0, 0])}, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
                     A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear={'x': list_to_tuple([0, 0]), 'y': list_to_tuple([0, 2 * level])}, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
                     A.Affine(scale=None, translate_percent={'x': list_to_tuple([0, 0.02 * level]), 'y': list_to_tuple([0, 0])}, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
                     A.Affine(scale=None, translate_percent={'x': list_to_tuple([0, 0]), 'y': list_to_tuple([0, 0.02 * level])}, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level)
        ]
        for i in range(number_branch):
            if number_branch != 4:
                employ = random.choice(strategy)
            else:
                index = random.randrange(len(strategy))
                employ = strategy.pop(index)
            level, shape = random.sample(transform[:6], employ[0]), random.sample(transform[6:], employ[1])
            img_transform = A.Compose([*level, *shape])
            random.shuffle(img_transform.transforms)
            if mask_i:
                transformed = img_transform(image=img, mask=mask)
                transformed_image, transformed_mask = transformed['image'], transformed['mask']
                return transformed_image, transformed_mask
            else:
                transformed = img_transform(image=img)
                transformed_image = transformed['image']
                return transformed_image

    def __getitem__(self, index):
        # ========= Read query image and label=========================
        image_path = self.img[index]
        label_path = self.mask[index]
        assert label_path[-14:-4] == image_path[-14:-4],'Data mismatch'
        image_open = pydicom.read_file(image_path)
        image = image_open.pixel_array
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #         print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labelimg = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # print("image_path:",image_path)
        # print("label_path:",label_path)
        
        for key, value in self.idx.items():
            labelimg[labelimg == key] = value

        # ========= data agument=========================
        if self.train_tag:
            image = np.expand_dims(image, axis=-1)
            image = image.astype(np.uint8)
            image, label = self.med_augment(image, labelimg, level=5, number_branch=4, mask_i=True)
            image = np.squeeze(image)
            image = Image.fromarray(image)
            label = Image.fromarray(label)
            # unique, inverse, counts = np.unique(label, return_inverse=True, return_counts=True)
            # print(' train labels unique is',unique)
        else:
            image = Image.fromarray(image)
            label = Image.fromarray(labelimg)
            # unique, inverse, counts = np.unique(label, return_inverse=True, return_counts=True)
            # print('val labels unique is',unique)
        
        image = self.img_transformer(image)
        mask_label = self.label_tranformer(label)
        mask_label=np.array(mask_label)
        mask_tensor=torch.from_numpy(mask_label)
        mask_tensor = mask_tensor.unsqueeze(0)
        # mask_tensor=mask_tensor.type(torch.long)
        image = image.float()
        mask_tensor = mask_tensor.float()
        
        # print('transformer labels unique is', np.unique(mask_tensor))
        # print('image',image.shape)
        # print('mask_tensor',mask_tensor.shape)
        # print('float labels unique is', np.unique(label))
        # print(image.dtype)
        # print(mask_tensor.dtype)
        # unique = torch.unique(mask)
        # print(unique)

        # return image, mask_tensor,image_path
        return image, mask_tensor,image_path

def get_data_loader(path, batch_size=4,train_tag='train',shuffle=True,img_size = 256,num_workers=0) -> torch.utils.data.DataLoader:
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
    # print("mr_img_path:",len(mr_img_path))
    # print("mr_mask_path:",len(mr_mask_path))
    assert len(mr_img_path) == len(mr_mask_path), \
    f"The number of MRI images ({len(mr_img_path)}) does not match the number of masks ({len(mr_mask_path)})."
    num_limit = 500
    if train_tag == 'train':
        # print("The length of train set is: {}".format(len(mr_img_path)))
        data = coCHAOSDataset(mr_img_path[:num_limit], mr_mask_path[:num_limit], image_transformer,label_transformer, train_tag=True)
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 pin_memory=False)
    elif train_tag == 'val':
        print("The length of val set is: {}".format(len(mr_img_path)))
        data = coCHAOSDataset(mr_img_path[num_limit:], mr_mask_path[num_limit:], image_transformer,label_transformer, train_tag=False)
        # data = coCHAOSDataset(mr_img_path, mr_mask_path, image_transformer,label_transformer, train_tag=True)
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 pin_memory=False)
    else:
        # random_number = random.randint(0, 300)
        print("The length of test set is: {}".format(len(mr_img_path)))
        data = coCHAOSDataset(mr_img_path, mr_mask_path, image_transformer,label_transformer, train_tag=False)
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 pin_memory=False)
    return dataloader

# def get_test_loader(path, batch_size=1,train_tag=False,shuffle=False) -> torch.utils.data.DataLoader:
#     """
#         Build the train loader. This is a standard loader (not episodic)
#     """
#     mr_i_path, mr_mask_path = get_MRdata(path)
#     print("The length of test set is: {}".format(len(mr__path)))
#     test_transformer = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((400, 400), antialias=True),
#     ])
#     # ===================== Build loader =====================
#     test_data = coCHAOSDataset(mr_im_path[1000:1100], mr_mask_path[1000:1100], test_transformer, train_tag=train_tag)
#     test_loader = torch.utils.data.DataLoader(test_data,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               num_workers=0,
#                                               pin_memory=False)
#     return test_loader
if __name__ == '__main__':
    # path = './CHAOS_Train/Train_Sets/'
    # train_path = r'E:\1pro\exp1_chaos\data\CHAOS_Train\Train_Sets\\'
    # path = r'E:/1pro/exp1_muti_chaos/data/CHAOS_Train/Train_Sets/MR/'
    # "/root/autodl-tmp/CHAOS_Train/Train_Sets/MR/"
    # "/root/autodl-tmp/CHAOS_Test/Test_Sets/MR/"
    train_path = r"/root/autodl-tmp/CHAOS_Train/Train_Sets/MR/"
    dl_train = get_data_loader(path=train_path,batch_size=4,train_tag='train',shuffle=True,img_size = 256,num_workers=2)
    # dl_test = get_data_loader(path=train_path,batch_size=4,train_tag='val',)
    image,label= next(iter(dl_train))
    print("train_set:",image.shape, label.shape)
    print("image tensor_float type:", image.dtype)
    print("label tensor_float type:", label.dtype)
    #--------------------------------------------------------------------------
    # image,label= next(iter(dl_test))
    # print("val_set:",image.shape, label.shape)
    # print(image.shape, label.shape)
    # img_transformer = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((512, 512)),
    #
    # ])
    # img_path, mask_path = getCHAOSdata(train_path)
    # train_data = coCHAOSDataset(img_path, mask_path, img_transformer)
    # dl_train = DataLoader(train_data, batch_size=4, shuffle=True)
    # img, prompt, mask, sup_image, sup_label, image_path = next(iter(dl_train))
    # print(img.shape, prompt.shape, mask.shape, sup_image.shape, sup_label.shape)

    # print(image.shape, label.shape)