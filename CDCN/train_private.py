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
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader, distributed
from torchvision import transforms
# from torchsummary import summary
from tqdm import tqdm
from alive_progress import alive_bar

from models.CDCNs import Conv2d_cd, CDCN, CDCNpp
from dataloader.Load_training_private_data import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from dataloader.Load_val_private_data import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from functions.utils import AvgrageMeter, accuracy, performances, make_weights_for_balanced_classes
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# feature  -->   [ batch, channel, height, width ]
def FeatureMap2Heatmap( x, feature1, feature2, feature3, map_x):
    ## initial images 
    feature_first_frame = x[0,:,:,:].cpu()    ## the middle frame 

    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_visual.jpg')
    plt.close()

    ## first feature
    feature_first_frame = feature1[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_Block1_visual.jpg')
    plt.close()
    
    ## second feature
    feature_first_frame = feature2[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_Block2_visual.jpg')
    plt.close()
    
    ## third feature
    feature_first_frame = feature3[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_Block3_visual.jpg')
    plt.close()
    
    ## third feature
    heatmap2 = torch.pow(map_x[0,:,:],2)    ## the middle frame 

    heatmap2 = heatmap2.data.cpu().numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap2)
    plt.colorbar()
    plt.savefig(args.log+'/'+args.log + '_x_DepthMap_visual.jpg')
    plt.close()

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
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().to(args.device)
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

        criterion_MSE = nn.MSELoss().to(args.device)

        loss = criterion_MSE(contrast_out, contrast_label)
        return loss

# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
    # wandb.init(project=args.wandb_name, sync_tensorboard=True)
    temp = datetime.now()

    year = temp.year
    month = temp.month
    day = temp.day

    args.log = args.log+"_{}_{}_{}".format(year, month, day)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    writer = SummaryWriter(args.log+"/"+args.log+"_cdcn_tb_{:04d}_{:02d}_{:02d}".format(year, month, day))
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    
    echo_batches = args.echo_batches

    print("Training on Private Data, {}_{}_{}:\n ".format(year, month, day))

    log_file.write('Private Data:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')
        log_file.write('finetune!\n')
        log_file.flush()
            
        model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
        model = model.to(args.device)

        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        model.load_state_dict(torch.load(args.weights, map_location=map_location))

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        

    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()

        model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
        model = model.to(args.device)
        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    criterion_absolute_loss = nn.MSELoss().to(args.device)
    criterion_contrastive_loss = Contrast_depth_loss().to(args.device)

    ACER_save = 1.0 
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma
        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()

        # ----------------------------------------- TRAIN -----------------------------------------
        model.train()
        
        # load random data every epoch
        train_data = Spoofing_train(args.train_list, args.image_dir, args.size, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=args.num_worker)

        i=0
        running_loss = 0.0
        abs_loss = 0.0
        cts_loss = 0.0

        with alive_bar(total=len(dataloader_train), theme="musical", length=200) as bar:
            for sample_batched in dataloader_train:
                # get the inputs
                inputs, map_label, spoof_label = sample_batched['image_x'].to(args.device), sample_batched['map_x'].to(args.device), sample_batched['spoofing_label'].to(args.device)
                optimizer.zero_grad()
                #pdb.set_trace()

                # forward + backward + optimize
                map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)

                absolute_loss = criterion_absolute_loss(map_x, map_label)
                contrastive_loss = criterion_contrastive_loss(map_x, map_label)

                loss =  absolute_loss + contrastive_loss

                loss.backward()
                optimizer.step()
                n = inputs.size(0)
                loss_absolute.update(absolute_loss.data, n)
                loss_contra.update(contrastive_loss.data, n)

                if i % echo_batches == echo_batches-1:    # print every 50 mini-batches

                    # visualization
                    FeatureMap2Heatmap(x_input, x_Block1, x_Block2, x_Block3, map_x)

                    # log written
                    log_file.write('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, i + 1, lr, loss_absolute.avg, loss_contra.avg))
                    log_file.flush()

                    writer.add_scalar('total loss',
                                loss_absolute.avg+loss_contra.avg,
                                epoch * len(dataloader_train) + i)

                    writer.add_scalar('absolute loss',
                                loss_absolute.avg,
                                epoch * len(dataloader_train) + i)

                    writer.add_scalar('contrastive loss',
                                loss_contra.avg,
                                epoch * len(dataloader_train) + i)
                i+=1
                bar()
        
        log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.flush()
           
        model.eval()


        # ----------------------------------------- TEST -----------------------------------------
        with torch.no_grad():
            # test for ACC
            test_data = Spoofing_valtest(args.test_list, args.image_dir, args.size, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
            dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_worker)
            
            map_score_list = []

            with alive_bar(total=len(dataloader_test), theme="musical", length=200) as bar:
                for sample_batched in dataloader_test:
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].to(args.device), sample_batched['spoofing_label'].to(args.device)

                    optimizer.zero_grad()

                    map_score = 0.0
                    map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,:,:,:])

                    score_norm = torch.mean(map_x)
                    map_score += score_norm
                    map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))

                    bar()

                map_score_test_filename = args.log+'/'+ args.log+'_map_score_test.txt'
                with open(map_score_test_filename, 'w') as file:
                    file.writelines(map_score_list)

                #############################################################
                #       performance measurement both val and test
                #############################################################
                test_ACC, test_APCER, test_BPCER, test_ACER, test_best_threshold = performances(map_score_test_filename)
                print('Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, test_threshold= %.4f' % (test_ACC, test_APCER, test_BPCER, test_ACER, test_best_threshold))
                log_file.write('Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, test_threshold= %.4f \n' % (test_ACC, test_APCER, test_BPCER, test_ACER, test_best_threshold))
                log_file.flush()
                if test_ACER<ACER_save:
                    print('-----------YAHOO !!! UPDATE SOTA----------------------')
                    torch.save(model.state_dict(), args.log+'/'+args.log+'_{}.pkl'.format((epoch + 1)))
                    ACER_save = test_ACER
                if epoch%2 == 0:
                    torch.save(model.state_dict(), args.log + '/' + args.log + '_{}.pkl'.format((epoch + 1)))
 

    print('Finished Training')
    writer.close()
    log_file.close()
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=1e-4
                            , help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=8, help='initial batchsize')
    parser.add_argument('--step_size', type=int, default=300, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--size', type=int, default=256, help='Image size')  # 256x256 or 112x112
    parser.add_argument('--epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp_train_oulu_npu_ftech", help='log and save model name')
    parser.add_argument('--image_dir', type=str, default="/root/", help='Train Image root') # khong can

    parser.add_argument('--train_list', default='../dataset/train_list.txt', help='Train Image list text')
    parser.add_argument('--test_list', default='../test_list.txt', help='Test Image list text')

    parser.add_argument('--num_worker', type=int, default=8, help='Num woker for dataloader')
    parser.add_argument('--device', default='cpu', help='cpu / cuda')
    parser.add_argument('--weights', type=str, default="/root/CDCNpp_train_oulu_npu_gplx_replay_print_2022_8_20_17.pkl", help='weight root')
    parser.add_argument('--finetune', action='store_true', help='whether finetune other models')
    parser.add_argument('--wandb_name',type=str,default='training_CDCN_ftech_fas_data',help='name wandb project')

    args = parser.parse_args()
    train_test()
  
