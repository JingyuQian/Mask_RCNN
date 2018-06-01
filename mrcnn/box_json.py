import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, lines
import argparse
import cv2
import json


def loadVideo(path_to_video):
    handle = cv2.VideoCapture(path_to_video)
    if not handle.isOpened():
        sys.exit('Opencv cannot open video {}'.format(path_to_video))
    else:
        return handle


def loadJson(path_to_json):
    with open(path_to_json, 'r') as fp:
        data = json.load(fp)
    ids = set([i['image_id'] for i in data])
    numFrames = len(ids)
    list_divide = [[]] * numFrames
    for i in data:
        list_divide[i['image_id']].append(i)
    return list_divide, numFrames


def draw_bbox(vh, bbox_list, num_to_draw, out_dir=None):
    colors = [0.5, 0, 0]
    width = vh.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vh.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    assert num_to_draw <= vh.get(cv2.CAP_PROP_FRAME_COUNT)
    for index in range(num_to_draw):
        success, image = vh.read()
        assert success, "Read {}-th frame failed".format(index)
        frame_instances = bbox_list[index]

        fig, ax = plt.subplots(figsize=(width / 220.0, height / 220.0), dpi=220)
        ax.axis('off')

        masked_image = image.astype(np.uint32).copy()

        for instance in frame_instances:
            x1 = instance['bbox'][0]
            y1 = instance['bbox'][1]
            width = instance['bbox'][2]
            height = instance['bbox'][3]
            p = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                  alpha=0.7, linestyle="dashed", edgecolor=colors, facecolor='none')
            ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))
        plt.show()
        wait = input()
        plt.close()
        if out_dir is not None:
            plt.savefig(str(index) + '.jpg', dpi=220)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, help="path to the video")
    ap.add_argument("-j", "--json", type=str, required=False,
                    help="json file containing all the bouding boxes")
    ap.add_argument("-n", "--num", type=int, required=False,
                    help="Number of frames to draw starting from 0")
    args = ap.parse_args()

    print('Loading video {}'.format(os.path.abspath(args.video)))
    vid_reader = loadVideo(args.video)
    print('Loading coco json {}'.format(os.path.abspath(args.json)))
    box_list, N = loadJson(args.json)
    draw_bbox(vid_reader, box_list, N)
