import os
import numpy as np 
import cv2



# all_class = ['live','spoof_at','spoof_cccd','spoof_cmnd','spoof_gplx','spoof_laptop','spoof_print','spoof_phone','spoof_tablet','spoof_tnv']

# for vl in all_class:
#     os.mkdir(f'/mnt/datadrive/thanhnc/FAS_data/private_data_train/high_crop/{vl}')


# all_type = os.listdir('/mnt/datadrive/thanhnc/FAS_data/private_data_train/high')
# for vl in all_type:
#     all_child = os.listdir(f'/mnt/datadrive/thanhnc/FAS_data/private_data_train/high/{vl}')
#     for c in all_child:
#         os.mkdir(f'/mnt/datadrive/thanhnc/FAS_data/private_data_train/high_crop/{vl}/{c}')



all_child = os.listdir(f'/mnt/datadrive/thanhnc/FAS_data/private_data_train/high_crop/live')
for c in all_child:
    os.mkdir(f'/mnt/datadrive/thanhnc/FAS_data/private_data_train/high_depth/live/{c}')


