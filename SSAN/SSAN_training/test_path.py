import cv2
import numpy as np
import os
import tqdm
from tqdm import trange
import random
import glob
def check_path_list(dir):
    count = 0
    name_array = []
    with open(dir) as f:
        lines = f.readlines()
        for i in tqdm.tqdm(lines):
            count += 1
            name = i.split("\n")[0]
            try:
                img = cv2.imread(name)
                size = img.shape
                if "live" in name:
                    img = cv2.imread(name.replace("crop", "depth"))
                    size = img.shape
            except:
                print(name)
    print(f"count: {count}")

if __name__ == '__main__':

    check_path_list("/root/data_train.txt")