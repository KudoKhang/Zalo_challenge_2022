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
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa
from pathlib import Path
from math import sqrt


#frames_total = 8    # each video 8 uniform samples
face_scale = 1.3  #default for test, for training , can be set from [1.2 to 1.5]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)), # GammaContrast with a gamma of 0.5 to 1.5
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
    iaa.pillike.EnhanceSharpness(),
    iaa.pillike.EnhanceColor(),
    #iaa.GaussianBlur(sigma=(0.0, 1.0)),
    iaa.MotionBlur(k=11, angle=[-45, 45]),
    iaa.Rotate((-25, 25))
])

def crop_face_from_scene(image, bbox, scale):

    x1, y1, w1, h1 = bbox
    real_h, real_w,_ = image.shape

    llength = sqrt(w1 * h1)
    center_x = w1 / 2 + x1
    center_y = h1 / 2 + y1
    size = int(llength * 1.4)
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    h = real_h
    w = real_w

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(image.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    #region = image[y1:h1+y1, x1:x1+w1, :]
    res[dsy:dey, dsx:dex] = image[sy:ey, sx:ex]


    return res


def read_bbox(image_path, face_name_full):

    image = cv2.imread(image_path)

    f = open(face_name_full,"r")
    f_rl = f.readlines()
    tmp  = f_rl
    if tmp == []:
        print(tmp, face_name_full)
    tmp_ = tmp[0]
    boxes = tmp_.split(" ")
    real_h, real_w,_ = image.shape

    x1 = int(int(boxes[0])*(real_w / 224))
    y1 = int(int(boxes[1])*(real_h / 224))
    w1 = int(int(boxes[2])*(real_w / 224))
    h1 = int(int(boxes[3])*(real_h / 224))
    bbox = [x1, y1, w1, h1]

    return bbox




# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_map_x = map_x/255.0                 # [0,1]
        return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_map_x = cv2.flip(map_x, 1)

                
            return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        map_x = np.array(map_x)
        
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'map_x': torch.from_numpy(map_x.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()}


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir, map_dir,  transform=None):

        #self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)

        char_lst = []

        with open(info_list) as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.rstrip()
                tmp_ = tmp.split(" ")
                char_lst.append(tmp_)

        self.landmarks_frame = char_lst
        self.info_list = info_list
        self.root_dir = root_dir
        self.map_dir = map_dir
        self.transform = transform
        self.root_fld = "/mnt/datadrive/dataset_faces/antispoof/celeba/CelebA-Spoof-zips/CelebA_Spoof/"

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        file_imgs = self.landmarks_frame[idx][0].split("/")
        file_img = file_imgs[len(file_imgs)-1]
        kind_img = file_imgs[len(file_imgs)-2]
        num_img = file_imgs[len(file_imgs)-3]
        
        self.landmarks_frame[idx][0] = self.landmarks_frame[idx][0].replace("train","train")
        image_path = os.path.join(self.root_fld, self.landmarks_frame[idx][0])
        #print(image_path)
        #bbox_path = image_path.replace(".jpg","_BB.txt")

        #bbox = read_bbox(image_path, bbox_path)
        bbox = [0, 0, 0, 0]

        spoofing_label = 0

        if "live" in str(kind_img):
            map_path = os.path.join(self.map_dir,num_img, kind_img, file_img)
            image_x, map_x = self.get_single_image_x(image_path, map_path, bbox, flag=True)
            spoofing_label = 1
        else:
            map_path = " "
            image_x = self.get_single_image_x(image_path, map_path, bbox, flag=False)
            spoofing_label = 0
            map_x = np.zeros((32, 32))
		    
        sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}


        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path, map_path, bbox, flag=True):

        # random scale from [1.2 to 1.5]
        #face_scale = np.random.randint(12, 15)
        #face_scale = face_scale/10.0
 
        # RGB
        image_x_temp = cv2.imread(image_path)

        # gray-map

        #image_x = cv2.resize(crop_face_from_scene(image_x_temp, bbox, face_scale), (256, 256))
        image_x = cv2.resize(image_x_temp, (256, 256))
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        image_x_aug = seq.augment_image(image_x) 
        if flag == True:
            map_x_temp = cv2.imread(map_path)
            #map_x = cv2.resize(crop_face_from_scene(map_x_temp, bbox, face_scale), (32, 32))
            map_x = cv2.resize(map_x_temp, (32, 32))
            map_x = cv2.cvtColor(map_x, cv2.COLOR_BGR2GRAY)
        
            return image_x_aug, map_x
        else:
            map_x = np.zeros((32, 32))

            return image_x_aug




