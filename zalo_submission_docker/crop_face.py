"""

"""
import time
import cv2
import sys
import yaml
import os
import tqdm
import cv2
import numpy as np
import glob
import tqdm

# sys.path.insert(0, ".")
# sys.path.insert(0, "./face_detection")
from face_detection.OpenVino import OpenVinoModel
face_detection = OpenVinoModel("./face_detection/models/320x320_25.xml", input_size=(320, 320))

def detect_face(img):
    bboxes = face_detection.predict(img)
    face_img = img[bboxes[0][1]:bboxes[0][3], bboxes[0][0]:bboxes[0][2]].copy()
    return face_img

def crop_face_Zalo(path, save_path):
    for video in tqdm.tqdm(glob.glob(path +"/*")):

        count = 0
        name_video = video.split("/")[-1].split(".")[0]
        cap = cv2.VideoCapture(video)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Original Frames
        frames = []
        for i in range(frame_number):
            ret, frame = cap.read()
            count += 1
            if count % 1 == 0:
                try:
                    face = detect_face(frame)

                    if not os.path.exists(f"{save_path}/{name_video}"):
                        os.makedirs(f"{save_path}/{name_video}")
                    cv2.imwrite(f"{save_path}/{name_video}/{name_video}_{count}.png", face)

                except:
                    continue
            else:
                continue
        cap.release()


if __name__ == '__main__':
    crop_face_Zalo("/mnt/sdb2/data/auto_test/public_test_2/videos_crop_3ddfa")







