import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_histogram(img):
  color = ('b', 'g', 'r')
  for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
  plt.show()

def process_sigmoid_contrast(img):

  norm_img1 = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  # scale to uint8
  norm_img1 = (255 * norm_img1).astype(np.uint8)
  norm_img2 = np.clip(norm_img2, 0, 1)
  norm_img2 = (255 * norm_img2).astype(np.uint8)

  return norm_img2

def process_gamma_correction(img):
  import math

  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  hue, sat, val = cv2.split(hsv)

  # compute gamma = log(mid*255)/log(mean)
  mid = 0.58
  mean = np.mean(val)
  gamma = math.log(mid * 255) / math.log(mean)

  # do gamma correction on value channel
  val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

  # combine new value channel with original hue and sat channels
  hsv_gamma = cv2.merge([hue, sat, val_gamma])
  img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

  return img_gamma2

def process_gamma_correction_v2(img, gamma):
  lookUpTable = np.empty((1, 256), np.uint8)
  for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
  res = cv2.LUT(img, lookUpTable)

  return res

def process_histogram(img):


  img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
  img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

  return img_output

  #color = ('b','g','r')
  #for i,col in enumerate(color):
  #  histr = cv2.calcHist([img_output],[i],None,[256],[0,256])
  #  plt.plot(histr,color = col)
  #  plt.xlim([0,256])
  #plt.show()

def power_law_transformation(img):
  img2 = np.log2(1 + img.astype(np.float)).astype(np.uint8)

  # Back to your code
  img2 = 38 * img2  # Edit from before

  img3 = img2
  B = np.int(img3.max())
  A = np.int(img3.min())

  c = (img2.max()) / (img2.max() ** (0.5))
  img2 = (c * img.astype(np.float) ** (0.5)).astype(np.uint8)

  return img2

def change_contrast_brightness(image, alpha, beta):

  new_image = np.zeros(image.shape, image.dtype)

  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      for c in range(image.shape[2]):
        new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

  return new_image

if __name__ == "__main__":

  root = "./dataset/check/img_crop.jpg"

  img = cv2.imread(root)
  #img_hist = power_law_transformation(img)
  res = process_gamma_correction_v2(img, gamma=0.5)
  plot_histogram(res)

  cv2.imshow("output", res)
  cv2.waitKey(0)
  cv2.destroyAllWindows()