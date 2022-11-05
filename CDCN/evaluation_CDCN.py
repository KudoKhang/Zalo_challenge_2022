from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import tqdm

from models.CDCNs import Conv2d_cd, CDCN, CDCNpp

from Load_OULUNPU_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_OULUNPU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import copy
import pdb

from functions.utils import AvgrageMeter, accuracy, performances, test_threshold_based

# Dataset root
test_image_dir = '/mnt/datadrive/dataset_faces/antispoof/celeba/CelebA-Spoof-zips/CelebA_Spoof/Data/train'
test_map_dir = '/mnt/datadrive/dataset_faces/antispoof/celeba/CelebA-Spoof-zips/CelebA_Spoof/Data/3ddfa_v2_depth/test_fb'
test_list =  './live_spoofing_label_test.txt'


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)


        criterion_MSE = nn.MSELoss().cuda()

        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)

        return loss

# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log_CA.txt', 'w')
    
    echo_batches = args.echo_batches

    print("CelebA-Spoofing, P1:\n ")

    log_file.write('CelebA-Spoofing, P1:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    print('finetune!\n')
    log_file.write('finetune!\n')
    log_file.flush()

    device = args.gpu
    t_start = time.time()
    model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
    t_load_model = time.time()
    model = model.cuda()

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    model.load_state_dict(torch.load('./CDCNpp_2021_11_4/CDCNpp_2021_11_4_5_2021_11_4.pkl', map_location=map_location))
    t_load_weight = time.time()

    model.eval()

    time_process_filename = args.log+'/'+args.log+'_time_process.txt'
    time_lst = []

    with torch.no_grad():

        #t_data1 = time.time()
        test_data = Spoofing_valtest(test_list, test_image_dir, test_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
        #t_data3 = time.time()

        map_score_list = []
        for  sample_batched in dataloader_test:
           # get the inputs
            inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
            test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet

            map_score = 0.0
            t_pr1 = time.time()
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,:,:,:])
            t_pr2 = time.time()
            t_pr = t_pr2 - t_pr1
            #score_norm = torch.mean(map_x)
            score_norm = torch.sum(map_x)/torch.sum(test_maps)
            map_score += score_norm

            is_NaN = math.isnan(map_score)
            is_inf = math.isinf(map_score)

            if is_NaN ==True or is_inf == True:
                map_score = 0

            map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))

            map_score_test_filename = args.log+'/'+ args.log+'_eval_map_score_test.txt'

            with open(map_score_test_filename, 'w') as file:
                file.writelines(map_score_list)

            time_lst.append(t_pr)

    t_mid = time.time()
    test_ACC, test_APCER, test_BPCER, test_ACER, threshold = performances(map_score_test_filename)
    t_end = time.time()

    with open(time_process_filename, 'w') as file:
        file.writelines("Loading CDCN model {}, loading weight from pretrained model {}, Total: {}".format(t_load_model-t_start, t_load_weight - t_load_model, t_load_weight - t_start))
        file.writelines("Average processing time for each image: {}".format(sum(time_lst) / len(time_lst)))

    print('Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, Threshold= %.4f' % (test_ACC, test_APCER, test_BPCER, test_ACER, threshold))
    log_file.write('Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, Threshold= %.4f' % (test_ACC, test_APCER, test_BPCER, test_ACER, threshold))
    log_file.flush()

    print('Finished Training')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=1, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=1024, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=1400, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp_P1", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test()
