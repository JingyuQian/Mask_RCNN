#!/usr/env/bin python

# Sample code to detect instances in video
# Author: Jingyu Qian
# Last modified: May 30, 2018

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import os
import sys
import time
import json
import numpy as np

# github root directory
ROOT_DIR = os.path.abspath("../")

# add search path
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
from mrcnn import vu
from mrcnn import model_lite
from mrcnn import utils
import coco

# Path to the save logs and trained model. Used for training.
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Path to the trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to the saved video file
VIDEO_DIR = ROOT_DIR
VIDEO_NAME = os.path.join(ROOT_DIR, 'wanda_20180325_ch07_part5.mp4')


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # The number of frames to run on shortcut model after an rpn_roi extraction
    SHORTCUT_FRAMES = 4


config = InferenceConfig()
config.display()

# Create model object in inference mode
model_inf = model_lite.MaskRCNN(model_dir=MODEL_DIR, config=config)
model_inf.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names. A total of 81 classes including background.
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Create a video_reader
video_reader = vu.VideoReader(VIDEO_NAME)
num_frames = video_reader.getTotalNumberOfFrames()
print('Total number of frames: {}'.format(num_frames))
cycle = config.SHORTCUT_FRAMES + 1
roi_touse = None
final_result = []
for frame_index in range(num_frames):
    print('*' * 20)
    print(frame_index)
    _, image, _ = video_reader.nextFrame()
    temp_result = {}
    # Run full model. Result is a list: [dict, proposals]
    if not frame_index % cycle:
        result = model_inf.detect([image])
        rois = result[0][0]['rois']
        class_ids = result[0][0]['class_ids']
        scores = result[0][0]['scores']
        masks = result[0][0]['masks']
        roi_touse = result[1]
    else:  # Run shortcut model. Result is a dictionary
        assert roi_touse is not None
        result = model_inf.detect_shortcut([image], roi_touse)
        rois = result[0]['rois']
        class_ids = result[0]['class_ids']
        scores = result[0]['scores']
        masks = result[0]['masks']
    for i, j, k in zip(class_ids, rois, scores):
        if i == 1:
            temp_result["image_id"] = frame_index
            temp_result["category_id"] = 0
            x = np.asscalar(j[1])
            y = np.asscalar(j[0])
            width = np.asscalar(j[3] - j[1])
            height = np.asscalar(j[2] - j[0])
            temp_result["bbox"] = [x, y, width, height]
            temp_result["score"] = np.asscalar(k)
            final_result.append(temp_result.copy())
video_reader.release()
with open('new_model.json', 'w') as file:
    json.dump(final_result, file)

sys.exit('Exiting the program.')

# r = result[0]
# visualize.display_instances(image, r['rois'], r['masks'], r[
#     'class_ids'], class_names, str(frame_index) + '.jpg', r['scores'])
