import os
import numpy as np
import cv2
import glob

from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from functions import (
            crop_img, parse_roi_box_from_bbox,
            )
#from equalizeHist import process_gamma_correction, process_sigmoid_contrast, process_histogram, power_law_transformation\
#    , change_contrast_brightness
#from process_filter import process_bilatering_filter, process_gaussian_blur
#from sharpening import unsharp_mask, sharp_filter
from matplotlib import pyplot as plt


face_boxes = FaceBoxes_ONNX()

def plot_histogram(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def detect_private_test():

    kind = "private_data_full_oak2"
    root = glob.glob("./dataset/train/OAK_v2/{}_raw/*".format(kind))
    root_dst = "./dataset/preprocess/test/{}".format(kind)
    if not os.path.exists(root_dst):
        os.makedirs(root_dst)

    f = open("./dataset/preprocess/test/{}_raw.txt".format(kind), "w")

    for kinds in root:
        folders = glob.glob(kinds+"/*")
        for folder in folders:
            imgs = glob.glob(folder+"/*.jpg")
            for img in imgs:

                # Generate a image file list for private data
                name_txt = img.split("{}_raw/".format(kind))[1]
                name_txt = name_txt.replace(" ", "")
                x = "{}/".format(kind)
                file_img = "{}\n".format(x+name_txt)

                # Find file name of an image.
                names = name_txt.split("/")
                name_img = names[len(names)-1]

                # Detect face with Facebox.
                imgx = cv2.imread(img)

                bboxs = face_boxes(imgx)
                if bboxs == []:
                    continue

                roi_box = parse_roi_box_from_bbox(bboxs[0])
                img_crop = crop_img(imgx, roi_box)

                f.write(file_img)
                root_dst_ = os.path.join(root_dst, names[0], names[1])

                if not os.path.exists(root_dst_):
                    os.makedirs(root_dst_)
                cv2.imwrite(os.path.join(root_dst_, name_img), img_crop)

    f.close()

def detect_private_ekyc():

    kind = "face_test"
    root = "./dataset/private_dataset_2.0.1/{}_raw".format(kind)
    root_dst = "./dataset/private_dataset_2.0.1/{}".format(kind)
    if not os.path.exists(root_dst):
        os.makedirs(root_dst)

    f = open("./dataset/private_dataset_2.0.1/{}_raw.txt".format(kind), "w")


    imgs = glob.glob(root+"/*.jpg")
    for img in imgs:
    # Generate a image file list for private data
        name_txt = img.split("{}_raw/".format(kind))[1]
        name_txt = name_txt.replace(" ", "")
        x = "{}/".format(kind)
        file_img = "{}\n".format(x+name_txt)

        names = name_txt.split("/")

        # Detect face with Facebox.
        imgx = cv2.imread(img)
        bboxs = face_boxes(imgx)
        if bboxs == []:
            print(img)
            continue

        roi_box = parse_roi_box_from_bbox(bboxs[0])
        img_crop = crop_img(imgx, roi_box)

        f.write(file_img)

        #if not os.path.exists(root_dst):
        #    os.makedirs(root_dst)
        #cv2.imwrite(os.path.join(root_dst, names[0]), img_crop)

    f.close()

def detect_face(img):

    bboxs = face_boxes(img)
    if bboxs == []:
        return 0

    roi_box = parse_roi_box_from_bbox(bboxs[0])
    img_crop = crop_img(img, roi_box)
    cv2.imshow("output", img_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #plot_histogram(img_crop)
    #plot_histogram(img_gm)


if __name__ == "__main__":
    detect_private_test()
    #detect_private_ekyc()

    #root = "./dataset/train/high/live/31/2022-01-21_362.jpg"
    #img = cv2.imread(root)

    #detect_face(img)
