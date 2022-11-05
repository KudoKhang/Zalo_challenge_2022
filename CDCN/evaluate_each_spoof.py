from __future__ import print_function, division
import torch
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import tqdm
from models.CDCNs import Conv2d_cd, CDCN, CDCNpp
from dataloader.Load_test_private_data import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest
from functions.utils import AvgrageMeter, accuracy, performances, make_weights_for_balanced_classes
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pdb
from datetime import datetime


def performances_wild_img(map_score_test_filename, threshold):
    # val
    with open(map_score_test_filename, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        name = str(tokens[3])
        #data.append({'map_score': score, 'label': label})
        #count += 1
        if label == 1 and args.type_device in name:
            count += 1
            data.append({'map_score': score, 'label': 1})
            num_real += 1
        elif label == 0 and args.type_device in name and args.type_attack in name:
        #else:
            count += 1
            data.append({'map_score': score, 'label': 0})
            num_fake += 1

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count
    val_APCER = type2 / num_fake if num_fake != 0 else 0
    val_BPCER = type1 / num_real if num_real !=0 else 0
    val_ACER = (val_APCER + val_BPCER) / 2.0
    print(f'all img: {count}  ; num fake: {num_fake}')
    return val_ACC, val_APCER, val_BPCER, val_ACER

# main function
def train_test():
    map_score_test_filename = os.path.join(args.log,f'test_score.txt')
    #test_ACC, test_APCER, test_BPCER, test_ACER,test_best_threshold = performances(map_score_test_filename)
    test_ACC, test_APCER, test_BPCER, test_ACER = performances_wild_img(map_score_test_filename,args.threshold)

    print('Test:  ACC = %.4f, APCER = %.4f, BPCER = %.4f, ACER = %.4f' % (test_ACC, test_APCER, test_BPCER, test_ACER))
    #print(f'Test_best_threshold: {test_best_threshold}')
    print('Finished Evaluation')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
#     parser.add_argument('--test_image_dir', type=str, default="/mnt/datadrive/thanhnc/FAS_data/private_test_ver2_thanhnc", help='Image root')
#     parser.add_argument('--test_list', default='/mnt/datadrive/thanhnc/FAS_data/private_test_ver2_thanhnc/private_test_ver2_Iphone_12_raw.txt', help='Image list text')
    parser.add_argument('--log', type=str, default="CDCN_test_private_v4(oulu+gplx)_all_epoch4_2022_08_19", help='Save log')
    parser.add_argument('--threshold', type=float, default=0.1615, help='Initialize a threshold')
    parser.add_argument('--device', default='cuda', help='cpu / cuda')
    parser.add_argument('--type_attack',type=str, help='type attack')
    parser.add_argument('--type_device',type=str, help='type device')
    args = parser.parse_args()
    train_test()
