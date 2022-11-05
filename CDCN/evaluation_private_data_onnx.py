from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
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
from functions.utils import  performances_wild_img

import torch.onnx
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
    device = args.gpu

    temp = datetime.now()
    year = temp.year
    month = temp.month
    day = temp.day

    args.log = args.log+'_{:04d}_{:02d}_{:02d}'.format(year, month, day)
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'.txt', 'w')

    print("Evaluation on Private Data (ONNX):\n ")

    log_file.write('Evaluation on Private Data (ONNX):\n ')
    log_file.flush()

    ort_session = onnxruntime.InferenceSession(args.weights)
    time_process_filename = args.log+'/'+args.log+'_time_process.txt'

    time_lst = []

    test_data = Spoofing_valtest(args.test_list, args.test_image_dir, args.size, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
    dataloader_test = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=4)
    map_score_list = []

    import tqdm

    for  sample_batched in dataloader_test:
        # get the inputs
        inputs, spoof_label, spoofing_name, image_path = sample_batched['image_x'].to(args.device), sample_batched['spoofing_label'].to(args.device), sample_batched["spoofing_name"][0], sample_batched["path"]
        map_score = 0.0
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}

        t_pr1 = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        t_pr2 = time.time()
        map_x = ort_outs[0]
        score_norm = np.mean(map_x)
        map_score += score_norm
            
        map_score_list.append('{} {} {} {}\n'.format(map_score, spoof_label[0][0], spoofing_name, image_path[0]))
        map_score_test_filename = args.log+'/'+ args.log+'_test_score.txt'

        with open(map_score_test_filename, 'w') as file:
            file.writelines(map_score_list)
        time_lst.append(t_pr2 - t_pr1)

    test_ACC, test_APCER, test_BPCER, test_ACER = performances_wild_img(map_score_test_filename, args.threshold)
    t_end = time.time()
    avg_time_img = sum(time_lst[1:]) / len(time_lst[1:])

    with open(time_process_filename, 'w') as file:
        file.writelines("Average processing time for each image: {} \n".format(avg_time_img))

    print('Test:  ACC = %.4f, APCER = %.4f, BPCER = %.4f, ACER = %.4f time = %.4f' % (test_ACC, test_APCER, test_BPCER, test_ACER, avg_time_img))
    log_file.write('Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f, threshold= %.4f, time= %.4f' % (test_ACC, test_APCER, test_BPCER, test_ACER, args.threshold, avg_time_img))
    log_file.flush()

    print('Finished Evaluation')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using CDCN model")
    parser.add_argument('--gpu', type=int, default=1, help='the gpu id used for predict')
    parser.add_argument('--weights', type=str, default="./zip/onnx/256/FAS_256s_v11.4.onnx", help='weight root')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--log', type=str, default="CDCNpp_Private_data_256s_v11.4_Mi10T", help='Save log')
    parser.add_argument('--threshold', type=float, default=0.1869, help='Initialize a threshold')
    parser.add_argument('--test_image_dir', type=str, default="./private/data_test_fixed_size", help='Image root')
    parser.add_argument('--test_list', default='./private/data_test_fixed_size/private_data_full_Mi10T_raw.txt', help='Image list text')
    parser.add_argument('--device', default='cuda', help='cpu / cuda')

    args = parser.parse_args()
    train_test()
