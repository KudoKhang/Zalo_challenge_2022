import os.path
import numpy as np
import json
import matplotlib.pyplot as plt



def write_new_data():
    root2 = "dataset/lb_training/ls_label_test_v2.txt"
    file_mid = open("./middle_test.txt", "w")

    char_lst = []
    with open(root2) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.rstrip()
            tmp_ = tmp.split(" ")
            char_lst.append(tmp_)

    for key, value in data.items():
        key_imgs = key.split("/")
        key_img = (key_imgs[len(key_imgs)-1])

        for i in range(len(char_lst)):
            if key_img in char_lst[i][0]:
                str_txt = " "
                if "live" in char_lst[i][0]:
                      str_txt = char_lst[i][0]+" "+"1\n"
                else:
                      str_txt = char_lst[i][0]+" "+"0\n"
                file_mid.write(str_txt)

    file_mid.close()

def cal_distribution(data):
    total = 0
    a4 = 0
    dig = 0
    mask = 0
    live = 0
    for key, value in data.items():
        key_imgs = key.split("/")
        key_img = (key_imgs[len(key_imgs)-1])
        if value[40] == 1 or value[40] == 2 or value[40] == 3:
            a4+=1
        elif value[40] == 4 or value[40] == 5 or value[40] == 6:
            mask+=1
        elif value[40] == 7 or value[40] == 8 or value[40] == 9:
            dig+=1
        else:
            live+=1
        total+=1

    print("A4 ", a4/total)
    print("Mask ", mask / total)
    print("Digital ", dig / total)
    print("Live ", live / total)

def process_score_distribution(root):
    with open(root, 'r') as file:
        lines = file.readlines()

    live = []
    spoof = []

    for line in lines:
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        #name = str(tokens[2])

        if label == 1:
            live.append(score)
        #elif label == 0 and "print" in name:
        else:
            spoof.append(score)

    mean = 0.5
    std = 0.2
    array_live = np.array(live)
    count, bins, ignored = plt.hist(array_live, 30, stacked=True)
    plt.plot(bins, 1 / (std * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mean) ** 2 / (2 * std ** 2)),
             linewidth=2, color='r')

    #mean = 1
    #std = 0.2
    #array_spoof = np.array(spoof)
    #count, bins, ignored = plt.hist(array_spoof, 30, stacked=True)
    #plt.plot(bins, 1 / (std * np.sqrt(2 * np.pi)) *
    #         np.exp(- (bins - mean) ** 2 / (2 * std ** 2)),
    #         linewidth=2, color='r')

    plt.show()


if __name__ == "__main__":

################## Calculating the existence of each "live" or "spoof" ##################
    #root = "/home/quangtn/projects/FAS/CDCN/data/protocol/protocol2/test_on_middle_quality_device/train_label.json"
    #f = open(root, "r")

    #data = json.load(f)
    #data_ = data.items

    #cal_distribution(data)

################## Calculating the score distribution of each "live" or "spoof" ##################

    root = "../result/v6/v6.3/CDCNpp_Private_data_aris_v6.3_2022_02_14/" \
           "CDCNpp_Private_data_aris_v6.3_2022_02_14_test_score.txt"
    process_score_distribution(root)