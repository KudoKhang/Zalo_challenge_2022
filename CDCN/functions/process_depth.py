import glob
import sys
import cv2
import numpy as np
import os

from TDDFA import TDDFA
from utils.depth import  generate_depth_image, depth
import yaml
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)


def process_celebA(root, tddfa, kind):

    root_dst = "/home/quangtn/projects/FAS/CDCN/CVPR2020_paper_codes/dataset/preprocess/train/crop_{}/".format(kind)
    root_dst_depth = "/home/quangtn/projects/FAS/CDCN/CVPR2020_paper_codes/dataset/preprocess/train/depth_{}".format(kind)

    f = open("/home/quangtn/projects/FAS/CDCN/CVPR2020_paper_codes/dataset/preprocess/train/label_train_{}.txt".format(kind), "w")
    face_boxes = FaceBoxes_ONNX()
    
    for folders in root:

        tps = glob.glob(folders+"/*")
        for tp in tps:
            imgs = glob.glob(tp+"/*.jpg") # train
            for img in imgs:

                names = img.split("/")
                name_img = names[len(names)-1]
                name_img = name_img.replace(" ", "")

                #print(names[len(names)-2], name_img)

                if "live" in img:
                    imgx = cv2.imread(img)
                    bboxs = face_boxes(imgx)
                    if bboxs == []:
                        continue


                    param_lst, roi_box_lst, img_crop = tddfa(imgx, bboxs)
                    dense_flag = True
                    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
                    depth_lst = ver_lst
                    name_fr = names[len(names)-2]

                    root_dst_depth_ = os.path.join(root_dst_depth, name_fr,"live")
                    file_img = str(os.path.join("Private/train", name_fr, "live", name_img))
                    file_img = "{}\n".format(file_img)
                    f.write(file_img)

                    if not os.path.exists(root_dst_depth_):
                        os.makedirs(root_dst_depth_)
                    root_dst_crop_ = os.path.join(root_dst, name_fr, "live")
                    if not os.path.exists(root_dst_crop_):
                        os.makedirs(root_dst_crop_)
                    img_depth = depth(imgx, depth_lst, tddfa.tri, show_flag=False, wfp=None, with_bg_flag=False)
                    img_depth_ = crop_img(img_depth, roi_box_lst[0])

                    cv2.imwrite(os.path.join(root_dst_depth_,name_img), img_depth_)
                    cv2.imwrite(os.path.join(root_dst_crop_, name_img), img_crop)

                else:
                    imgx = cv2.imread(img)
                    bboxs = face_boxes(imgx)
                    if bboxs == []:
                        continue
                    roi_box = parse_roi_box_from_bbox(bboxs[0])
                    name_fr = names[len(names)-2]
                    name_spoof = "{}".format(names[len(names)-3])

                    file_img = str(os.path.join("Private/train/", name_fr, name_spoof, name_img))
                    file_img = "{}\n".format(file_img)
                    f.write(file_img)
                    
                    root_dst_crop_ = os.path.join(root_dst, name_fr, name_spoof)

                    if not os.path.exists(root_dst_crop_):
                        os.makedirs(root_dst_crop_)

                    img_crop = crop_img(imgx, roi_box)

                    cv2.imwrite(os.path.join(root_dst_crop_, name_img), img_crop)



    f.close()


if __name__ == '__main__':

    kind = "oak2"
    root = glob.glob("/home/quangtn/projects/FAS/CDCN/CVPR2020_paper_codes/dataset/train/{}/*".format(kind))

    cfg = yaml.load(open("./configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    tddfa = TDDFA(**cfg)
    process_celebA(root, tddfa, kind)
