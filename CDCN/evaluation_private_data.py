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

    temp = datetime.now()
    year = temp.year
    month = temp.month
    day = temp.day

    args.log = args.log+'_{:04d}_{:02d}_{:02d}'.format(year, month, day)
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/log_private_data.txt', 'w')

    # load the network, load the pre-trained model in UCF101?
    print('Do an evaluation on Private Data!\n')
    log_file.write('Do an evaluation on Private Data!\n')
    log_file.flush()

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

    time_process_filename = args.log+'/time_process.txt'
    time_lst = []

    with torch.no_grad():

        t_data1 = time.time()
        test_data = Spoofing_valtest(args.test_list, args.test_image_dir, args.size, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_test = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=4)
        t_data3 = time.time()

        map_score_list = []
        import tqdm
        for  sample_batched in tqdm.tqdm(dataloader_test):
           # get the inputs
            inputs, spoof_label, spoofing_name, image_path = sample_batched['image_x'].to(args.device), sample_batched['spoofing_label'].to(args.device), sample_batched["spoofing_name"][0], sample_batched["path"]
            map_score = 0.0
            t_pr1 = time.time()
            map_x, _, _ ,_, _, _  =  model(inputs[:,:,:,:])
            t_pr2 = time.time()
            t_pr = t_pr2 - t_pr1
#            print(t_pr, map_x.shape)
            score_norm = torch.mean(map_x)
            map_score += score_norm
            
            map_score_list.append('{} {} {} {}\n'.format(map_score, spoof_label[0][0], spoofing_name, image_path[0]))
            map_score_test_filename = args.log+'/test_score.txt'

            with open(map_score_test_filename, 'w') as file:
                file.writelines(map_score_list)
            time_lst.append(t_pr)
    t_mid = time.time()

    test_ACC, test_APCER, test_BPCER, test_ACER,test_best_threshold = performances(map_score_test_filename)
    t_end = time.time()

    avg_time_img = sum(time_lst[1:]) / len(time_lst[1:])

    with open(time_process_filename, 'w') as file:
        file.writelines("Loading CDCN model {}, loading weight from pretrained model {}, Total: {} \n".format(t_load_model-t_start, t_load_weight - t_load_model, t_load_weight - t_start))
        file.writelines("Average processing time for each image: {} \n".format(avg_time_img))

    print('Test:  ACC = %.4f, APCER = %.4f, BPCER = %.4f, ACER = %.4f time = %.4f' % (test_ACC, test_APCER, test_BPCER, test_ACER, avg_time_img))
    log_file.write('Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, threshold= %.4f, time= %.4f' % (test_ACC, test_APCER, test_BPCER, test_ACER,test_best_threshold, avg_time_img))
    log_file.flush()

    print('Finished Evaluation')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=1, help='the gpu id used for predict')
    parser.add_argument('--weights', type=str, default="CDCNpp_train_oulu_npu_gplx_replay_print_2022_8_20/CDCNpp_train_oulu_npu_gplx_replay_print_2022_8_20_45.pkl", help='weight root')  
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--test_image_dir', type=str, default="/mnt/datadrive/thanhnc/FAS_data/private_test_ver2_thanhnc", help='Image root')
    parser.add_argument('--test_list', default='/mnt/datadrive/thanhnc/FAS_data/test_private_all_clean.txt', help='Image list text')
    parser.add_argument('--log', type=str, default="CDCN_test_private_v4(oulu+gplx+replay+print)_all_epoch45", help='Save log')
    parser.add_argument('--device', default='cuda', help='cpu / cuda')

    args = parser.parse_args()
    train_test()

