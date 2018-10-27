import os
import json
import numpy as np

def gen_config(args):

    if args.seq != '':
        # generate config from a sequence name

        seq_home = 'NIPS2018/DAT/dataset/OTB'
        save_home = 'NIPS2018/DAT/result_fig_labgpu_test'
        result_home = 'NIPS2018/DAT/result_labgpu_test'
        
        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir,x) for x in img_list]
        if seq_name=="david":
           img_list=img_list[299:]
           print(len(img_list))
        for line in open(gt_path):
            temprect = line
            break
        if ',' in temprect:
            gt = np.loadtxt(gt_path,delimiter=',')
        else:
            gt = np.loadtxt(gt_path)

        init_bbox = gt[0]
        
        savefig_dir = os.path.join(save_home,seq_name)
        result_dir = os.path.join(result_home,seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir,'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json,'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None
        
    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path
