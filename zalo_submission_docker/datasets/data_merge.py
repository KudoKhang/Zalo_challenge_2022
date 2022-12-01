import os
import torch
import cv2

from .Load_private_train import Spoofing_train as Spoofing_train_private
from .Load_private_valtest import Spoofing_valtest as Spoofing_valtest_private


class dataset_info(object):

    def __init__(self):
        self.root_dir = ""


class data_merge(object):

    def __init__(self, image_dir):
        self.dic = {}
        self.image_dir = image_dir

        # Private
        private_info = dataset_info()
        private_info.root_dir = os.path.join(self.image_dir, "private_images_crop")
        self.dic["private"] = private_info

    def get_single_dataset(self, data_name="", train=True, img_size=256, map_size=32, transform=None, debug_subset_size=None, UUID=-1):
        if train:
            data_dir = self.dic[data_name].root_dir
            data_set = Spoofing_train_private(os.path.join("/mnt/datadrive/thonglv/SSAN/train_rotate_txt.txt"),
                                           os.path.join(data_dir, "Train_files"), transform=transform,
                                           img_size=img_size, map_size=map_size, UUID=UUID)

            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        else:

            data_set = Spoofing_valtest_private(os.path.join("/mnt/datadrive/thonglv/SSAN/public_v2_one.txt"),  # test: public_v2.txt
                                           os.path.join(data_dir, "Train_files"), transform=transform,
                                           img_size=img_size, map_size=map_size, UUID=UUID)
            
            if debug_subset_size is not None:
                data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
        print("Loading {}, number: {}".format(data_name, len(data_set)))
        
        return data_set

    def get_datasets(self, train=True, protocol="1", img_size=256, map_size=32, transform=None, debug_subset_size=None):
        if protocol == "private":
            data_name_list_train = ["private"]
            data_name_list_test = ["private"]
        sum_n = 0
        if train:
            data_set_sum = self.get_single_dataset(data_name=data_name_list_train[0], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0)
            sum_n = len(data_set_sum)
            for i in range(1, len(data_name_list_train)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_train[i], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i)
                data_set_sum += data_tmp
                sum_n += len(data_tmp)
        else:
            data_set_sum = {}
            for i in range(len(data_name_list_test)):
                data_tmp = self.get_single_dataset(data_name=data_name_list_test[i], train=False, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i)
                data_set_sum[data_name_list_test[i]] = data_tmp
                sum_n += len(data_tmp)
        print("Total number: {}".format(sum_n))
        return data_set_sum
