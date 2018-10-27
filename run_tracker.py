from PIL import Image
import matplotlib.pyplot as plt
import json

from gen_config import gen_config
from DAT.tracking.run_DAT import run_mdnet

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

if __name__ == "__main__":
    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, result_path = gen_config(
        seq='football', json='', savefig=True)

    # Run tracker
    result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=True)

    # Save result
    res = {}
    res['res'] = result_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(result_path, 'w'), indent=2)