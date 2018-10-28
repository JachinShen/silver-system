import cv2 as cv
import matplotlib.pyplot as plt
import json
import random

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
        self.bbox[0] += random.random()
        self.bbox[3] += random.random()
        return self.bbox

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

    cv.rectangle(image, (x, y), (x+width, y+height), color)
    return image


if __name__ == "__main__":
    # Generate sequence config
    img_list, init_bbox, gt = gen_seq( seq='Football')

    # Tracker
    tracker = Tracker()

    # init
    image_init = cv.imread(img_list[0])
    image_init = cv.cvtColor(image_init, cv.COLOR_BGR2RGB)
    tracker.init_bbox(image_init, gt[0])


    # Run tracker
    for img_file, ground_truth in zip(img_list, gt):
        image = cv.imread(img_file)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # draw init
        image_draw = image.copy()
        image_draw = draw_rect(image_draw, ground_truth, "red")

        # update
        result_bbox = tracker.update(image)
        print(result_bbox)

        image_draw = draw_rect(image_draw, result_bbox, "blue")

        image_draw = cv.cvtColor(image_draw, cv.COLOR_RGB2BGR)
        cv.imshow("image", image_draw)
        cv.waitKey(0)
        # plt.imshow(image)
        # plt.show()

    # Save result
    # res = {}
    # res['res'] = result_bb.round().tolist()
    # res['type'] = 'rect'
    # res['fps'] = fps