import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import cv2
import glob
import argparse
from ThreeDDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
# from functions import (
#             crop_img, parse_roi_box_from_bbox,
#             )
#from equalizeHist import process_gamma_correction, process_sigmoid_contrast, process_histogram, power_law_transformation\
#    , change_contrast_brightness
#from process_filter import process_bilatering_filter, process_gaussian_blur
#from sharpening import unsharp_mask, sharp_filter
from matplotlib import pyplot as plt




def crop_img(img, roi_box):
    h, w = img.shape[:2]
    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx

    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]

    #res = img[sy:ey+sy, sx:ex+sx, :]

    return res

def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]

    left = bbox[0]
    top = bbox[1]
    right = bbox[2]
    bottom = bbox[3]



    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.0)

    roi_box = [0] * 4
    roi_box[0] = center_x - size /2
    roi_box[1] = center_y - size
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size * 1.4


  
    return roi_box

parser = argparse.ArgumentParser(description="save quality using landmarkpose model")

parser.add_argument('--device', type=str, help='device')

args = parser.parse_args()


face_boxes = FaceBoxes_ONNX()


# path = '/mnt/datadrive/thanhnc/FAS_data/raw_datatest/live/31/2022-01-14_695.jpg'

# imgx = cv2.imread(path)

# bboxs = face_boxes(imgx)

# roi_box = parse_roi_box_from_bbox(bboxs[0])
# img_crop = crop_img(imgx, roi_box)

# cv2.imwrite('input_thonglv.jpg',imgx)
# cv2.imwrite('crop_face_thonglv.jpg',img_crop)
import tqdm
from tqdm import trange

same_dv = ['Iphone_8','Iphone_Xs','Xiaomi_Mi_10T']


root_device = f'/mnt/datadrive2/dataset/dataset_faces/antispoof/Private_test_v2/{args.device}'

root_save = f'/mnt/datadrive/thanhnc/FAS_data/private_test_ver2_thanhnc/{args.device}_30fps'


total_img = 0
all_class = os.listdir(root_device)
type_attack_atent = ['spoof_cccd2','spoof_at2','spoof_gplx2','spoof_cmnd2','spoof_laptop2','spoof_paper2','spoof_phone2','spoof_tnv2']
for class_name in tqdm.tqdm(all_class):
    print(f'-------------- {class_name} ---------------')
    all_video = os.listdir(f'{root_device}/{class_name}') if class_name in type_attack_atent else []
#     if 'spoof_cccd2' not in class_name:
#         all_video = []
    for video_name in all_video:
        print(f'Process: {video_name}\n')
        video_path = os.path.join(root_device,class_name,video_name)
        capture = cv2.VideoCapture(video_path)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
#         print(f'{class_name}  {video_name}')
#         print(f'fps video: {fps}')
#         #print(f'frame widht: {frame_width} ; frame height : {frame_height}')
#         print(f'num frame: {num_frame}')
        #print(f'-------------------------------')
  
        for idx in trange(num_frame):
            _, frame = capture.read()
            bboxs = face_boxes(frame)
            if len(bboxs)>0:
                roi_box = parse_roi_box_from_bbox(bboxs[0])
                img_crop = crop_img(frame, roi_box)
                video_name_rm_tag = video_name
                video_name_rm_tag = video_name_rm_tag.replace('.MOV','')
                video_name_rm_tag = video_name_rm_tag.replace('.mp4','')
                name_pix = f'{video_name_rm_tag}_{idx}.jpg'
                if not os.path.exists(os.path.join(root_save,class_name)):
                    os.makedirs(os.path.join(root_save,class_name))
                path_save = os.path.join(root_save,class_name,name_pix)
                cv2.imwrite(path_save,img_crop)
