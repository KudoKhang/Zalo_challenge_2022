import torch
import torch.nn as nn
import os
from networks import get_model
from datasets import data_merge
from optimizers import get_optimizer
from torch.utils.data import Dataset, DataLoader
from transformers import *
from utils import *
from configs import parse_args
import time
import numpy as np
import random
from loss import *
import tqdm
from datetime import datetime
import torch.onnx
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

torch.manual_seed(16)
np.random.seed(16)
random.seed(16)


def main(args):
    temp = datetime.now()
    year = temp.year
    month = temp.month
    day = temp.day
    args.log = args.log + "_{}_{}_{}".format(year, month, day)
    if not os.path.exists(args.log):
        os.makedirs(args.log)

    log_file = open(args.log + '/' + args.log + '_log_evaluate_private_data_ONNX.txt', 'w')
    log_file.write('Evaluate private ONNX data:\n ')
    log_file.flush()

    data_bank = data_merge(args.data_dir)

    model = onnxruntime.InferenceSession("/root/SSAN/FAS_256s_SSAN.onnx", None, providers=['CUDAExecutionProvider'])


    # metrics
    eva = {
        "best_epoch": -1,
        "best_HTER": 100,
        "best_auc": -100,
        "best_APCER": 100,
        "best_BPCER": 100,
    }

    if args.trans in ["o", "p"]:
        test_data_dic = data_bank.get_datasets(train=False, protocol=args.protocol, img_size=args.img_size,
                                               transform=transformer_test_video(),
                                               debug_subset_size=args.debug_subset_size)
    elif args.trans in ["I"]:
        test_data_dic = data_bank.get_datasets(train=False, protocol=args.protocol, img_size=args.img_size,
                                               transform=transformer_test_video_ImageNet(),
                                               debug_subset_size=args.debug_subset_size)
    else:
        raise Exception

    score_root_path = os.path.join(args.result_path, args.result_name, "score_ONNX")
    check_folder(score_root_path)
    map_score_name_list = []
    score_path = os.path.join(score_root_path, "evaluate_epoch_ONNX")
    check_folder(score_root_path)
    for i, test_name in enumerate(test_data_dic.keys()):
        print("[{}/{}]Testing {}...".format(i + 1, len(test_data_dic), test_name))
        test_set = test_data_dic[test_name]
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=32)
        HTER, auc_test, APCER, BPCER, val_threshold = test_video(model, args, test_loader, score_path,
                                                                 name=test_name)
        print(f'---------Evaluate: APCER= {APCER:0.4f}, BPCER= {BPCER:0.4f}, '
              f'val_threshold={val_threshold:0.4f}---------')
        log_file.write(f'---------Evaluate: APCER= {APCER:0.4f}, BPCER= {BPCER:0.4f}, '
                       f'val_threshold={val_threshold:0.4f}---------\n')
        log_file.flush()



def test_video(model, args, test_loader, score_root_path, name=""):

    start_time = time.time()
    scores_list = []
    for sample_batched in tqdm.tqdm(test_loader):
        image_x, label = sample_batched["image_x"].cuda(), sample_batched["label"].cuda()
        map_score = 0
        for frame_i in range(image_x.shape[1]):
            if args.model_type in ["SSAN_R"]:
                str_time = time.time()
                ort_outs = model.run(None,
                                     {model.get_inputs()[0].name: to_numpy(image_x),
                                      model.get_inputs()[1].name: to_numpy(image_x)})
                print(f"Time process: {time.time() - str_time}")

                score_norm = torch.softmax(torch.from_numpy(ort_outs[0]), dim=1)[:, 1]
            print(score_norm)

            map_score += score_norm
        map_score = map_score / image_x.shape[1]
        for ii in range(image_x.shape[0]):
            scores_list.append("{} {} {}\n".format(map_score[ii], label[ii][0], map_score[3]))

    map_score_val_filename = os.path.join(score_root_path, "{}_score.txt".format(name))
    print("score: write test scores to {}".format(map_score_val_filename))
    with open(map_score_val_filename, 'w') as file:
        file.writelines(scores_list)

    test_ACC, fpr, FRR, HTER, APCER, BPCER, ACER, auc_test, test_err, val_threshold =\
        performances_val(map_score_val_filename)
    print("## {} score:".format(name))
    print("test:  val_ACC={:.4f}, HTER={:.4f}, APCER={:.4f}, BPCER={:.4f}, ACER={:.4f},"
          "AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, val_threshold={:.4f}".format(
                                                          test_ACC, HTER, APCER, BPCER, ACER, auc_test, test_err,
                                                          test_ACC, val_threshold))
    print("test phase cost {:.4f}s".format(time.time() - start_time))

    return HTER, auc_test, APCER, BPCER, val_threshold


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args=args)