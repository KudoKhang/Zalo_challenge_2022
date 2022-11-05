import cv2
import numpy as np

from equalizeHist import process_gamma_correction
from  sharpening import unsharp_mask, sharp_filter, color_quantization, Gaussian_smooth

def process_median_blur(img):
    dst = cv2.medianBlur(img,5)

    return dst

def process_gaussian_blur(img):
    dst = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)

    return dst

def process_img_filter(img):
    kernel = np.ones((3, 3), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)

    return dst

def process_averaging(img):
    blur = cv2.blur(img, (3, 3))

    return blur

def process_bilatering_filter(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)

    return blur




if __name__ == "__main__":

    root = "./dataset/check/img_52/pixel_distribution/2021-12-16_210.jpg"

    img = cv2.imread(root)

    #img = process_gamma_correction(img)
    img_gs_sm = Gaussian_smooth(img)
    cv2.imwrite("dataset/check/img_52/gaussian_filter/live_gc_sm.jpg", img_gs_sm)