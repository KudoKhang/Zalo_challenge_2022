import torch
import torch.nn as nn
import os
from networks import get_model
from utils import *
import argparse
import time
import numpy as np
import random
from loss import *
import tqdm
from datetime import datetime
import cv2


from face_detection.OpenVino import OpenVinoModel
face_detection = OpenVinoModel("./face_detection/models/320x320_25.xml", input_size=(320, 320))

torch.manual_seed(16)
np.random.seed(16)
random.seed(16)

def parse_args():
    parser = argparse.ArgumentParser()
    # inference settings
    parser.add_argument('--model_path', type=str,
                        default="./saved_model/SSAN_R_pprivate_epochs_29.pth", help='model_path')
    parser.add_argument('--device', type=str, default='0', help='device: 0, 1')
    parser.add_argument('--test_case_path', type=str, default="./data", help='test_case_path')
    parser.add_argument('--result_file', type=str, default="./result/submission.csv", help='result_file_path')
    return parser.parse_args()



def detect_face(img):
    bboxes = face_detection.predict(img)
    face_img = img[bboxes[0][1]:bboxes[0][3], bboxes[0][0]:bboxes[0][2]].copy()
    return face_img

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

    # set pad color
    if len(image.shape) is 3 and not isinstance(pad_color,
                                                (list, tuple, np.ndarray)):  # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img


def normalize(image):
    image = image.astype(np.float32)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image /= 255.0
    image -= mean
    image /= std
    return image


def preproces(image):
    face = resize_and_pad(image, (256, 256))
    new_img = normalize(face)
    new_img = new_img[:, :, ::-1].transpose((2, 0, 1))
    new_img = np.array(new_img)
    img2tens = torch.from_numpy(new_img.astype(np.float32)).float()
    img2tens = img2tens.cpu().unsqueeze(0)
    return img2tens


class Inference:
    def __init__(self, model):
        self.model = get_model("SSAN_R", max_iter=685000).cpu()
        model_ = torch.load(model)
        self.model.load_state_dict(model_["state_dict"])
        self.model.eval()

    def predict(self, img_x):
        image_x = preproces(img_x)
        cls_x1_x1, _, _, _ = self.model(image_x[:, :, :, :], image_x[:, :, :, :])
        score_norm = torch.softmax(cls_x1_x1, dim=1)[:, 1]
        return score_norm.item()


def main(args_, FAS):

    #define variables
    video_name = []

    # define the result file
    result = open(args_.result_file, "w")
    result.writerow(['fname,liveness_score'])

    #load video
    for video in glob.glob(args_.test_case_path + "/.mp4"):
        video_name.append(video)

    for video in tqdm.tqdm(video_name):
        score = []
        count = 0
    
        cap = cv2.VideoCapture(video)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_number):
            ret, frame = cap.read()
            count += 1
            try:
                face = detect_face(frame)
                score.append(FAS.predict(face))
            except:
                if count == frame_number:
                    video_score = [f'{video.split("/")[-1]}', 0]
                    result.writerow(video_score)
                else:
                    continue
        cap.release()
        video_score = [f'{video.split("/")[-1]}', np.mean(score)]
        result.writerow(video_score)
    print("Done..!")


if __name__ == '__main__':
    args = parse_args()
    # model_path = "./saved_model/SSAN_R_pprivate_epochs_62.pth"
    FAS_loader = Inference(args.model_path)
    # img_path = "/mnt/datadrive/thonglv/SSAN/train_data/crop_live_180/817_4_180.png"
    # img = cv2.imread(img_path)
    # print(FAS.predict(img))

    main(args, FAS_loader)

