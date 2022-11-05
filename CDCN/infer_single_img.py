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
from dataloader.Load_test_private_data import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest,resize_and_pad

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pdb
from datetime import datetime
from functions.utils import performances_wild_img,performances


# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)


    device = args.gpu
    t_start = time.time()
    model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
    t_load_model = time.time()

    model = model.to(args.device)

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    weights_root = args.weights
    model.load_state_dict(torch.load(weights_root, map_location=map_location))

    t_load_weight = time.time()

    model.eval()
    transform = transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()])
    with torch.no_grad():
        image_x_temp = cv2.imread('/home/thanhnc/face-anti-spoofing/face_det_openvino/thanh_crop_crop.png')
        image_x = resize_and_pad(image_x_temp, (256,256))
        new_image_x = (image_x - 127.5)/128 
        new_image_x = new_image_x[:,:,::-1].transpose((2, 0, 1))
        new_image_x = np.array(new_image_x)
        new_image_x = torch.from_numpy(new_image_x.astype(np.float)).float()

        img = new_image_x.unsqueeze(0).cuda()
        map_x, _, _ ,_, _, _  =  model(img)
        score_norm = torch.mean(map_x)
        print(f'score: {score_norm}')



      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--weights', type=str, default="CDCNpp_train_oulu_npu_gplx_replay_print_2022_8_20/CDCNpp_train_oulu_npu_gplx_replay_print_2022_8_20_17.pkl", help='weight root')  
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--test_image_dir', type=str, default="/mnt/datadrive/thanhnc/FAS_data/private_test_ver2_thanhnc", help='Image root')
    parser.add_argument('--test_list', default='/mnt/datadrive/thanhnc/FAS_data/test_private_all_clean.txt', help='Image list text')
    parser.add_argument('--log', type=str, default="CDCN_test_private_v4(oulu+gplx+replay+print)_all_epoch45", help='Save log')
    parser.add_argument('--device', default='cuda', help='cpu / cuda')

    args = parser.parse_args()
    train_test()

