'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' 
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020 
'''

from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
from pathlib import Path
from skimage import io
from skimage.transform import resize
from PIL import Image
from math import sqrt
import json

SIZE = 0

def resize_and_pad(image, size, pad_color=0):
    h, w = image.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(image.shape) is 3 and not isinstance(pad_color,(list, tuple, np.ndarray)):  # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img



class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
#        new_image_x = image_x/ 255.0
        return {'image_x': new_image_x, 'val_map_x': val_map_x , 'spoofing_label': spoofing_label}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        val_map_x = np.array(val_map_x)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'val_map_x': torch.from_numpy(val_map_x.astype(np.float)).float(),'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()} 


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir, size,  transform=None):

        global SIZE
        char_lst = []
        with open(info_list) as f:
            lines = f.readlines()
            for line in lines:
                tmp_ = line.replace('\n','')
                char_lst.append(tmp_)

        self.landmarks_frame = char_lst
        self.info_list = info_list
        self.root_dir = root_dir
        self.transform = transform
        SIZE = size


    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #file_imgs = self.landmarks_frame[idx][0].split("/")
#         file_imgs = self.landmarks_frame[idx][0]
#         file_img = file_imgs[len(file_imgs)-1]
#         kind_img = file_imgs[len(file_imgs)-2]
#         num_img = file_imgs[len(file_imgs)-3]
        self.info_list_ = self.info_list.split("/")

        #image_path = os.path.join(self.root_dir, self.landmarks_frame[idx][0])
        image_path = self.landmarks_frame[idx]
        bbox_path = " "

        if "live" in image_path:
            val_map_path = ''
            image_x, val_map_x = self.get_single_image_x(image_path, val_map_path, bbox_path, flag=True)
            spoofing_label = 1
        else:
            val_map_path = " "
            image_x = self.get_single_image_x(image_path, val_map_path, bbox_path, flag=False)# , videoname)
            spoofing_label = 0
            val_map_x = np.zeros((32, 32))

        sample = {'image_x': image_x, 'val_map_x':val_map_x , 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path, val_map_path,  bbox_path, flag=True):#videoname):
        
        # RGB
        image_x_temp = cv2.imread(image_path)
        # gray-map
    
        image_x = resize_and_pad(image_x_temp, (SIZE, SIZE))
        # transform to binary mask --> threshold = 0 
        if flag == True:
            # val_map_x_temp = cv2.imread(val_map_path)
            # temp = resize_and_pad(val_map_x_temp, (32, 32))
            # temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            # #np.where(temp < 1, temp, 1)
            # val_map_x = temp
            val_map_x = None
            return image_x, val_map_x
        else:
            return image_x

