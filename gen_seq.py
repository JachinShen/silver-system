import os
import json
import numpy as np


def gen_seq(seq):

    if seq != '':
        # generate config from a sequence name

        seq_home = './dataset/OTB'
        # save_home = './result_fig_labgpu_test'
        # result_home = './result_labgpu_test'

        seq_name = seq
        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]
        if seq_name == "david":
            img_list = img_list[299:]
            print(len(img_list))
        for line in open(gt_path):
            temprect = line
            break
        if ',' in temprect:
            gt = np.loadtxt(gt_path, delimiter=',')
        else:
            gt = np.loadtxt(gt_path)

        init_bbox = gt[0]

        # savefig_dir = os.path.join(save_home, seq_name)
        # result_dir = os.path.join(result_home, seq_name)
        # if not os.path.exists(result_dir):
            # os.makedirs(result_dir)
        # result_path = os.path.join(result_dir, 'result.json')
    return img_list, init_bbox, gt
