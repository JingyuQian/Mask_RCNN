#!/usr/env/bin python

# Sample code to detect instances in video
# Author: Jingyu Qian


from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import os
import sys

ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))

from mrcnn import vu
from mrcnn import model
from mrcnn import utils
from mrcnn import visualize

import coco

# Path to the save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Path to the trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to the saved video file
VIDEO_DIR = ROOT_DIR


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode
model_inf = model.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained with MS-COCO
model_inf.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
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

VIDEO_NAME = './cat.mp4'

# Create a video_reader
video_reader = vu.VideoReader(VIDEO_NAME)
num_frames = video_reader.getTotalNumberOfFrames()

for frame_index in range(num_frames):
    print('*' * 20)
    print(frame_index)
    _, image, _ = video_reader.nextFrame()
    result = model_inf.detect([image], verbose=1)
    r = result[0]
    visualize.display_instances(image, r['rois'], r['masks'], r[
        'class_ids'], class_names, str(frame_index) + '.jpg', r['scores'])