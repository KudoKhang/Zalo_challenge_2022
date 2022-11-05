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
from torchsummary import summary
from tqdm import tqdm


from models.CDCNs import Conv2d_cd, CDCN, CDCNpp

from dataloader.Load_OULUNPU_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from dataloader.Load_OULUNPU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from functions.utils import AvgrageMeter, accuracy, performances, make_weights_for_balanced_classes
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Dataset root
train_image_dir = '/mnt/datadrive/dataset_faces/antispoof/celeba/CelebA-Spoof-zips/CelebA_Spoof/Data/crop/train_2crop'        
test_image_dir = '/mnt/datadrive/dataset_faces/antispoof/celeba/CelebA-Spoof-zips/CelebA_Spoof/Data/crop/test_2crop'   
   
map_dir = '/mnt/datadrive/dataset_faces/antispoof/celeba/CelebA-Spoof-zips/CelebA_Spoof/Data/3ddfa_v2_depth/train_2fb'   
test_map_dir = '/mnt/datadrive/dataset_faces/antispoof/celeba/CelebA-Spoof-zips/CelebA_Spoof/Data/3ddfa_v2_depth/test_2fb' 

train_list = './data_train/ls_label_train_v2.txt'
test_list =  './data_train/ls_label_test_v2.txt'


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
    temp = datetime.now()

    year = temp.year
    month = temp.month
    day = temp.day

    args.log = args.log+"_{}_{}_{}".format(year, month, day)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    writer = SummaryWriter(args.log+"/"+args.log+"_cdcn_tb_{:04d}_{:02d}_{:02d}".format(year, month, day))
    log_file = open(args.log+'/'+ args.log+'_log_{}_{}_{}.txt'.format(year, month, day), 'w')
    
    
    echo_batches = args.echo_batches

    print("CelebA-Spoofing, {}_{}_{}:\n ".format(year, month, day))

    log_file.write('CelebA-Spoofing:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')
        log_file.write('finetune!\n')
        log_file.flush()
            
        model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
        model = model.cuda()

        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        model.load_state_dict(torch.load('./CDCNpp_112x112_v1_2022_3_3/CDCNpp_112x112_v1_2022_3_3_32.pkl',map_location=map_location))

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        

    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()

        model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
        model = model.cuda()

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model)
    
    
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    
    train_data = Spoofing_train(train_list, train_image_dir, map_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
    dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

    #bandpass_filter_numpy = build_bandpass_filter_numpy(30, 30)  # fs, order  # 61, 64 

    ACER_save = 1.0
    running_loss = 0.0
    abs_loss = 0.0
    cts_loss = 0.0
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        #top5 = utils.AvgrageMeter()
        
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        # load random 16-frame clip data every epoch
        i=0
        j=0
        for sample_batched in tqdm(dataloader_train):
             # get the inputs
            inputs, map_label, spoof_label = sample_batched['image_x'].cuda(), sample_batched['map_x'].cuda(), sample_batched['spoofing_label'].cuda() 

            optimizer.zero_grad()
            
                #pdb.set_trace()
            
                # forward + backward + optimize
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
           
            
            absolute_loss = criterion_absolute_loss(map_x, map_label)
            contrastive_loss = criterion_contrastive_loss(map_x, map_label)
            
            loss =  absolute_loss + contrastive_loss
            #loss =  absolute_loss 
            running_loss += loss 
            abs_loss += absolute_loss
            cts_loss += contrastive_loss

            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
        

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
                
                # visualization
                FeatureMap2Heatmap(x_input, x_Block1, x_Block2, x_Block3, map_x)

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
                log_file.write('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, i + 1, lr, loss_absolute.avg, loss_contra.avg))
                log_file.flush()
                
            #break

            if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
                writer.add_scalar('total loss',
                            running_loss / 1000,
                            epoch * len(dataloader_train) + i)

                writer.add_scalar('absolute loss',
                            abs_loss / 1000,
                            epoch * len(dataloader_train) + i)

                writer.add_scalar('contrastive loss',
                            cts_loss / 1000,
                            epoch * len(dataloader_train) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
                running_loss = 0.0
                abs_loss = 0.0
                cts_loss = 0.0
            i+=1
        
        # whole epoch average:
        if j%100==0:
            print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        j+=1

        log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.flush()
           
    
                    
        #### validation/test
        if epoch <300:
            epoch_test = 300   
        else:
            epoch_test = 20  

        if (epoch+1) % 2 == 0:
            model.eval()
            
            with torch.no_grad():
                ###########################################
                '''                test             '''
                ##########################################
                # test for ACC
                test_data = Spoofing_valtest(test_list, test_image_dir, test_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)
                
                map_score_list = []
                
                for  sample_batched in tqdm(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    map_score = 0.0
                    map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,:,:,:])
                    
                    score_norm = torch.mean(map_x)
                    map_score += score_norm
                    map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))
                                
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
                
        torch.save(model.state_dict(), args.log+'/'+args.log+'_{}.pkl'.format((epoch + 1)))


    print('Finished Training')
    writer.close()
    log_file.close()
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=1, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001
                            , help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=32, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=4, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=3, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp_256s_v8.5", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=True, help='whether finetune other models')

    args = parser.parse_args()
    train_test()
