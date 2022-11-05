import os
import numpy as np
import cv2
import glob

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




face_boxes = FaceBoxes_ONNX()


path = '/mnt/datadrive/thanhnc/FAS_data/raw_datatest/live/31/2022-01-14_695.jpg'

imgx = cv2.imread(path)

bboxs = face_boxes(imgx)

roi_box = parse_roi_box_from_bbox(bboxs[0])
img_crop = crop_img(imgx, roi_box)

cv2.imwrite('input_thonglv.jpg',imgx)
cv2.imwrite('crop_face_thonglv.jpg',img_crop)