from DaSiamRPN.net import SiamRPNBIG
from DaSiamRPN.run_SiamRPN import SiamRPN_init, SiamRPN_track
from modules.utils import overlap_ratio
import cv2 as cv
import matplotlib.pyplot as plt
import json
import random
import torch
import numpy as np

from gen_seq import gen_seq
# from DAT.tracking.run_DAT import run_mdnet


class Tracker:
    def __init__(self):
        pass

    def load_model(self):
        pass

    def init_bbox(self, image, bbox):
        self.image = image
        self.bbox = bbox

    def update(self, image):
        self.image = image
        return self.bbox


class DaSiamRPN():
    def __init__(self):
        pass

    def load_model(self, net_file):
        self.net = SiamRPNBIG()
        self.net.load_state_dict(torch.load(net_file, map_location="cpu"))
        self.net.eval().cpu()

        for i in range(10):
            self.net.temple(torch.autograd.Variable(
                torch.FloatTensor(1, 3, 127, 127)).cpu())
            self.net(torch.autograd.Variable(
                torch.FloatTensor(1, 3, 255, 255)).cpu())

    def init_bbox(self, image, bbox):
        target_pos = np.array(bbox[0:2])
        target_size = np.array(bbox[2:4])
        self.state = SiamRPN_init(image, target_pos, target_size, self.net)

    def update(self, image):
        self.state = SiamRPN_track(self.state, image)
        target_pos = self.state['target_pos']
        target_size = self.state['target_sz']
        bbox = [target_pos[0], target_pos[1], target_size[0], target_size[1]]
        return bbox


def draw_rect(image, rect, color):
    x = int(rect[0])
    y = int(rect[1])
    width = int(rect[2])
    height = int(rect[3])

    if color == "green":
        color = (0, 255, 0)
    elif color == "red":
        color = (255, 0, 0)
    elif color == "blue":
        color = (0, 0, 255)

    cv.rectangle(image, (x, y), (x+width, y+height), color, 3)
    return image


if __name__ == "__main__":
    # Generate sequence config
    img_list, init_bbox, gt = gen_seq(seq='Football')

    # Tracker
    tracker = DaSiamRPN()
    tracker.load_model("./DaSiamRPN/SiamRPNBIG.model")

    # init
    image_init = cv.imread(img_list[0])
    image_init = cv.cvtColor(image_init, cv.COLOR_BGR2RGB)
    tracker.init_bbox(image_init, gt[0])
    overlap_ratio_list = []

    # Run tracker
    for img_file, ground_truth in zip(img_list[1:4], gt[1:4]):
        image = cv.imread(img_file)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # draw init
        image_draw = image.copy()
        image_draw = draw_rect(image_draw, ground_truth, "red")

        # update
        result_bbox = tracker.update(image)
        result_bbox = np.array(result_bbox)
        overlap = overlap_ratio(result_bbox, ground_truth)
        overlap_ratio_list.append(overlap)
        print("Overlap{}".format(overlap))

        image_draw = draw_rect(image_draw, result_bbox, "blue")

        image_draw = cv.cvtColor(image_draw, cv.COLOR_RGB2BGR)
        cv.imshow("image", image_draw)
        cv.waitKey(0)
        # plt.imshow(image)
        # plt.show()

    plt.plot(overlap_ratio_list)
    # Save result
    # res = {}
    # res['res'] = result_bb.round().tolist()
    # res['type'] = 'rect'
    # res['fps'] = fps
