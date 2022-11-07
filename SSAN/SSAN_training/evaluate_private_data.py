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
import onnx

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

    log_file = open(args.log + '/' + args.log + '_log_evaluate_private_data.txt', 'w')
    log_file.write('Evaluate private data:\n ')
    log_file.flush()


    data_bank = data_merge(args.data_dir)
    # define train loader
    # define model
    model = get_model(args.model_type, max_iter=4114800).cuda()

    # def optimizer
    optimizer = get_optimizer(
        args.optimizer, model,
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    # def scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # model = nn.DataParallel(model).cuda()

    model_ = torch.load("/root/SSAN/results/demo/model/SSAN_R_pprivate_best.pth")
    model.load_state_dict(model_["state_dict"])
    model.eval()
    """
    size = 256
    output = "FAS_{}s_SSAN.onnx".format(size, size)
    model.eval()

    inputs = torch.zeros(1, 3, size, size).cuda()
    torch.onnx.export(model,
                      (inputs, inputs),
                      output,
                      export_params=True,
                      verbose=False,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input', 'input_2'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size', 2: 'width', 3: 'height'},
                                    'input_2': {0: 'batch_size', 2: 'width', 3: 'height'},
                                    'output': {0: 'batch_size', 2: 'width', 3: 'height'}}
                      )


    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model, full_check=True)
    """
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

    score_root_path = os.path.join(args.result_path, args.result_name, "score")
    check_folder(score_root_path)
    map_score_name_list = []
    score_path = os.path.join(score_root_path, "evaluate_epoch")
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
    with torch.no_grad():
        start_time = time.time()
        scores_list = []
        for sample_batched in tqdm.tqdm(test_loader):
            image_x, label = sample_batched["image_x"].cuda(), sample_batched["label"].cuda()
            map_score = 0
            for frame_i in range(image_x.shape[1]):
                if args.model_type in ["SSAN_R"]:
                    cls_x1_x1, fea_x1_x1, fea_x1_x2, _ = model(image_x[:, :, :, :], image_x[:, :, :, :])
                    score_norm = torch.softmax(cls_x1_x1, dim=1)[:, 1]
                elif args.model_type in ["SSAN_M"]:
                    pred_map, fea_x1_x1, fea_x1_x2, _ = model(image_x[:, :, :, :],
                                                              image_x[:, :, :, :])
                    score_norm = torch.sum(pred_map, dim=(1, 2)) / (args.map_size * args.map_size)
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