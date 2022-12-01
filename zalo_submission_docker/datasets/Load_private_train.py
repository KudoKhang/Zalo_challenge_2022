import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import math
import os
from glob import glob


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

    if len(image.shape) == 3 and not isinstance(pad_color,(list, tuple, np.ndarray)):  # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img


class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir, transform=None, scale_up=1.5, scale_down=1.0, img_size=256, map_size=32, UUID=-1):
        char_lst = []
        with open(info_list) as f:
            lines = f.readlines()
            for line in lines:
                tmp_ = line.replace('\n', '')
                char_lst.append(tmp_)

        self.landmarks_frame = char_lst

        self.root_dir = root_dir
        self.transform = transform
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.img_size = img_size
        self.map_size = map_size
        self.UUID = UUID

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_path = self.landmarks_frame[idx]
        if "live" in image_path:
            spoofing_label = 1
            # map_path = image_path.replace('crop', 'depth')
            map_path = " "
            image_x, _ = self.get_single_image_x(image_path, map_path, spoofing_label)
        # real
        else:
            spoofing_label = 0          # fake
            map_path = " "
            # image_x, map_x = self.get_single_image_x(image_path, map_path, spoofing_label)
            image_x, _ = self.get_single_image_x(image_path, map_path, spoofing_label)

        sample = {"image_x": image_x, "label": spoofing_label, "UUID": self.UUID}
        # sample = {"image_x": image_x, "label": spoofing_label, "map_x": map_x, "UUID": self.UUID}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_dir, map_path, spoofing_label):
        image_x_temp = cv2.imread(image_dir)
        map_x = None
        try:
            image_x = resize_and_pad(image_x_temp, (self.img_size, self.img_size))
            # image_x = cv2.resize(image_x_temp, (self.img_size, self.img_size))
        except:
            print(image_dir)
        # if image_x_temp is None:
        #     print(image_dir)
        # if spoofing_label:
        #     map_x_temp = cv2.imread(map_path, 0)
        #     map_x = resize_and_pad(map_x_temp, (32, 32))
        # else:
        #     map_x = np.zeros((32, 32))

        return image_x, map_x
