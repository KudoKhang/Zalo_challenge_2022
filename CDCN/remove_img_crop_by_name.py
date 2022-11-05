import os
from glob import glob
import argparse


parser = argparse.ArgumentParser(description="save quality using landmarkpose model")

parser.add_argument('--device', type=str, help='device')

args = parser.parse_args()

root_dir = f'/mnt/datadrive/thanhnc/FAS_data/private_test_ver2_thanhnc/{args.device}/spoof_phone2/*'
dic_video = {'Iphone_11': ['IMG_1735', 'IMG_1734', 'IMG_1733'],
             'Iphone_8': ['IMG_9728', 'IMG_9729', 'IMG_9730'],
             'Iphone_12': ['IMG_1436', 'IMG_1435', 'IMG_1434'],
             'Iphone_Xs': ['IMG_7393', 'IMG_7392', 'IMG_7391'],
             'SS_Note_8': ['20220714_101411', '20220714_101312', '20220714_101230'],
             'Vsmart': ['VID_20220714_105326_246', 'VID_20220714_105250_735', 'VID_20220714_105356_575'],
             'Xiaomi_Mi_10T': ['VID_20220714_104302', 'VID_20220714_104228', 'VID_20220714_104153'],
             'Iphone_Xs_30fps':['IMG_7459', 'IMG_7458', 'IMG_7457']
             }


all_file = glob(root_dir)

c = 0
for vl in all_file:
    t = 0
    for k in dic_video[args.device]:
        if k in vl:
            t +=1
    if t==0:
        os.remove(vl)
#         c +=1
# print(f'num file will be remove: {c}')