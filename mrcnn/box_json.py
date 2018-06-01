# Author: Jingyu Qian
# Visualization from COCO formatted json files

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import argparse
import cv2
import json
import glob


def loadVideo(path_to_video):
    """
    Load a video a return the cv2 Video object.
    :param path_to_video: path to the video

    :return: a cv2 video object
    """
    handle = cv2.VideoCapture(path_to_video)
    if not handle.isOpened():
        sys.exit('Opencv cannot open video {}'.format(path_to_video))
    else:
        return handle


def loadJson_det(path_to_json):
    """
    Load a COCO formatted json file containing detection results.
    :param path_to_json: path to the json file

    :return: a dictionary in the form of {frame_id: list dict}
    :return: number of unique frames from the json file
    """
    with open(path_to_json, 'r') as fp:
        data = json.load(fp)
    ids = set([i.get('image_id') for i in data])
    numFrames = len(ids)
    list_divide = {}
    for i in data:
        frame_id = i['image_id']
        if not list_divide.get(frame_id, 0):
            list_divide[frame_id] = []
        list_divide[frame_id].append(i.copy())
    return list_divide, numFrames


def loadJson_gt(path_to_json):
    with open(path_to_json, 'r') as fp:
        data = json.load(fp)
    assert 'annotations' in data
    data = data['annotations']
    ids = set([i.get('image_id') for i in data])
    numFrames = len(ids)
    list_divide = {}
    for i in data:
        frame_id = i['image_id']
        if not list_divide.get(frame_id, 0):
            list_divide[frame_id] = []
        list_divide[frame_id].append(i.copy())
    return list_divide, numFrames


def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    :param box: a single detection result in the form of {'image_id': int, 'category_id': int, 'bbox': list int, 'score': float}
    :param boxes: a list of detection results from a frame loaded from loadJson function.
    """
    # Convert [x,y,w,h] into [x1,y1,x2,y2]
    box_x1, box_y1 = box['bbox'][:2]
    box_x2 = box_x1 + box['bbox'][2]
    box_y2 = box_y1 + box['bbox'][3]

    boxes_x1 = np.array([i['bbox'][0] for i in boxes])
    boxes_y1 = np.array([i['bbox'][1] for i in boxes])

    boxes_x2 = np.array([i['bbox'][2] for i in boxes]) + boxes_x1
    boxes_y2 = np.array([i['bbox'][3] for i in boxes]) + boxes_y1

    # Compute intersection
    x1 = np.maximum(box_x1, boxes_x1)
    x2 = np.minimum(box_x2, boxes_x2)
    y1 = np.maximum(box_y1, boxes_y1)
    y2 = np.minimum(box_y2, boxes_y2)

    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
    boxes_area = (boxes_x2 - boxes_x1) * (boxes_y2 - boxes_y1)

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """
    Compute IoUs for 2 sets of bounding boxes. Typically detection results and ground truths.

    :param boxes1: a list of dictionaries. Each dictionary is a detection result.
    :param boxes2: a list of dictionaries. Each dictionary is a detection result.

    :return: A numpy matrix of overlap
    """
    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for index, box in enumerate(boxes1):
        overlaps[index, :] = compute_iou(box, boxes2)
    return overlaps


def bbox_match(det_boxes, gt_boxes, threshold):
    """
    Match a set of detection boxes to ground truth boxes.

    :param det_boxes: a list of detection results from loadJson function
    :param gt_boxes: a list of ground truths from loadJson function
    :param threshold: threshold IoU to match the boxes

    :return: a dictionary {det_id: gt_id}
    """
    match = {}
    overlaps = compute_overlaps(det_boxes, gt_boxes)
    matched_gt_index = []

    # Loop over all detection boxes and find each corresponding ground truth
    # box.
    for i, _ in enumerate(det_boxes):
        gt_match = np.argmax(overlaps[i, :])

        # Check to prevent if a gt box is matched twice

        # If the overlap is bigger than threshold and the gt hasn't been matched
        # to another detectoin box, match it.
        if overlaps[i][gt_match] >= threshold and gt_match not in matched_gt_index:
            match[i] = gt_match
            matched_gt_index.append(gt_match)
        # If the gt has been matched to a previous detection,
        # choose the one with bigger IoU
        elif overlaps[i][gt_match] >= threshold and gt_match in matched_gt_index:
            key = [k for k, v in match.items if v == gt_match][0]
            if overlaps[i][gt_match] < overlaps[key][gt_match]:
                match[key] = None
                match[i] = gt_match
            else:
                match[i] = None
        else:
            match[i] = None
    return match


def draw_match_to_sequence(vh, det_list, gt_list,
                           threshold=0.7, num_to_draw=None, auto=True, out_dir=None):
    """
    Match each detection to a ground truth, and draw each pair of match, and optionally saves them.

    :param vh: cv2 video handle
    :param det_list: a list of detection bounding boxes. Loaded from loadJson function.
    :param gt_list: a list of ground truth boxes. Loaded from loadJson function.
    :param threshold: IoU threshold
    :param num_to_draw: number of frames to draw bboxes on. If None, defaults to number of freams in the video object
    :param auto: if False, show every picture and wait, and press any key to continue; Otherwise don't show picture
    :param out_dir: picture saving directory
    """
    assert det_list and gt_list, "Must have both detections and ground truths"
    det_color = [1, 0, 0]
    gt_color = [0, 1, 0]
    arrow_color = [0, 0, 1]

    width = vh.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vh.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if not num_to_draw:
        num_to_draw = vh.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        assert num_to_draw <= vh.get(cv2.CAP_PROP_FRAME_COUNT)

    for index in range(num_to_draw):
        print('Drawing {}-th frame...'.format(index))
        fig, ax = plt.subplots(
            figsize=(width / 192.0, height / 192.0), dpi=192)
        ax.axis('off')

        success, image = vh.read()
        assert success, "Read {}-frame failed".format(index)
        image = image[:, :, ::-1]
        masked_image = image.astype(np.uint32).copy()
        # convert BGR to RGB

        dets_frame = det_list[index]  # a list of dictionaries
        gts_frame = gt_list[index]

        match_results = bbox_match(dets_frame, gts_frame, threshold=threshold)

        for det, gt in match_results.items():
            if gt is None:
                continue
            xd, yd, wd, hd = dets_frame[det]['bbox']
            xt, yt, wt, ht = gts_frame[gt]['bbox']
            p = patches.Rectangle((xd, yd), wd, hd, linewidth=0.5,
                                  alpha=0.9, linestyle='solid', edgecolor=det_color, facecolor='none')
            ax.add_patch(p)
            p = patches.Rectangle((xt, yt), wt, ht, linewidth=0.5,
                                  alpha=0.9, linestyle='dashed', edgecolor=gt_color, facecolor='none')
            ax.add_patch(p)
            p = patches.Arrow(xd, yd, xt - xd, yt - yd,
                              width=0.5, color=arrow_color)
            ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))

        if not auto:
            plt.show()
            cv2.waitKey(0)
        if out_dir:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            save_name = os.path.join(
                os.path.abspath(out_dir), str(index) + '.jpg')
            plt.savefig(save_name, dpi=192)
        plt.close()


def draw_bbox_to_sequence(vh, det_list, gt_list, num_to_draw=None, auto=True, out_dir=None):
    """
    Draw detection bounding boxes to a sequence, and optionally saves them.

    :param vh: cv2 video handle
    :param det_list: a list of detection bounding boxes. Loaded from loadJson function.
    :param gt_list: a list of ground truth boxes. Loaded from loadJson function.
    :param num_to_draw: number of frames to draw bboxes on. If None, defaults to number of freams in the video object
    :param auto: if False, show every picture and wait, and press any key to continue; Otherwise don't show picture
    :param out_dir: picture saving directory
    """

    assert det_list or gt_list, "No boxes to draw"
    det_color = [1.0, 0, 0]
    gt_color = [0, 1.0, 0]

    width = vh.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vh.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if not num_to_draw:
        num_to_draw = vh.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        assert num_to_draw <= vh.get(cv2.CAP_PROP_FRAME_COUNT)

    for index in range(num_to_draw):

        print('Drawing {}-th frame...'.format(index))
        # dpi is based on Macbook Pro retina resolution
        fig, ax = plt.subplots(
            figsize=(width / 192.0, height / 192.0), dpi=192)
        ax.axis('off')

        success, image = vh.read()
        assert success, "Read {}-th frame failed".format(index)

        # convert BGR to RGB
        image = image[:, :, ::-1]
        masked_image = image.astype(np.uint32).copy()

        # detection results
        if det_list:
            frame_instances = det_list[index]

            for instance in frame_instances:
                x1, y1, w, h = instance['bbox']
                p = patches.Rectangle((x1, y1), w, h, linewidth=0.5,
                                      alpha=0.9, linestyle="dashed", edgecolor=det_color, facecolor='none')
                ax.add_patch(p)
        if gt_list:
            frame_instances = gt_list[index]

            for instance in frame_instances:
                x1, y1, w, h = instance['bbox']
                p = patches.Rectangle((x1, y1), w, h, linewidth=0.5,
                                      alpha=0.9, linestyle="dashed", edgecolor=gt_color, facecolor='none')
                ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))

        if not auto:
            plt.show()
            cv2.waitKey(0)
        if out_dir:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            save_name = os.path.join(
                os.path.abspath(out_dir), str(index) + '.jpg')
            plt.savefig(save_name, dpi=192)
        plt.close()


def draw_false_positive(vh, det_list, gt_list, threshold=0.7, num_to_draw=None, auto=True, out_dir=None):
    """
    Match each detection to a ground truth, and draw each pair of match, and optionally saves them.

    :param vh: cv2 video handle
    :param det_list: a list of detection bounding boxes. Loaded from loadJson function.
    :param gt_list: a list of ground truth boxes. Loaded from loadJson function.
    :param threshold: IoU threshold
    :param num_to_draw: number of frames to draw bboxes on. If None, defaults to number of freams in the video object
    :param auto: if False, show every picture and wait, and press any key to continue; Otherwise don't show picture
    :param out_dir: picture saving directory
    """
    assert det_list and gt_list, "Must have both detections and ground truths"
    det_color = [1, 0, 0]

    width = vh.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vh.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if not num_to_draw:
        num_to_draw = vh.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        assert num_to_draw <= vh.get(cv2.CAP_PROP_FRAME_COUNT)

    for index in range(num_to_draw):
        print('Drawing {}-th frame...'.format(index))
        fig, ax = plt.subplots(
            figsize=(width / 192.0, height / 192.0), dpi=192)
        ax.axis('off')

        success, image = vh.read()
        assert success, "Read {}-frame failed".format(index)
        image = image[:, :, ::-1]
        masked_image = image.astype(np.uint32).copy()
        # convert BGR to RGB

        dets_frame = det_list[index]  # a list of dictionaries
        gts_frame = gt_list[index]

        match_results = bbox_match(dets_frame, gts_frame, threshold=threshold)

        for det, gt in match_results.items():
            if gt is None:
                xd, yd, wd, hd = dets_frame[det]['bbox']
                p = patches.Rectangle((xd, yd), wd, hd, linewidth=0.5,
                                      alpha=0.9, linestyle='solid', edgecolor=det_color, facecolor='none')
                ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))

        if not auto:
            plt.show()
            cv2.waitKey(0)
        if out_dir:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            save_name = os.path.join(
                os.path.abspath(out_dir), str(index) + '.jpg')
            plt.savefig(save_name, dpi=192)
        plt.close()


def stitch_to_video(path, shape):
    now_path = os.path.abspath('.')
    pic_path = os.path.abspath(path)
    os.chdir(pic_path)
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    video_writer = cv2.VideoWriter('stitched.mp4', fourcc, 24, (int(shape[0]), int(shape[1])))

    pic_list = sorted(glob.glob('*.jpg'), key=lambda x: int(x.split('.')[0]))

    for i in pic_list:
        image = cv2.imread(i)
        video_writer.write(image)
    video_writer.release()

    os.chdir(now_path)


def run_args(args):
    """
    Run program according to parsed args.
    """
    print('Loading video {} \n'.format(os.path.abspath(args.video)))
    vid_reader = loadVideo(args.video)
    if args.det:
        print('Loading coco json {} \n'.format(os.path.abspath(args.det)))
        det_dict, N1 = loadJson_det(args.det)
    else:
        det_dict = None
        N1 = None
    if args.gt:
        print('Loading coco json {} \n'.format(os.path.abspath(args.gt)))
        gt_dict, N2 = loadJson_gt(args.gt)
    else:
        gt_dict = None
        N2 = None
    if N1 and N2:
        assert N1 == N2, "Detected different number of frames in two json files"

    if args.num == 0:
        args.num = N1

    if args.mode == 'default':
        draw_bbox_to_sequence(vid_reader, det_dict, gt_dict,
                              args.num, args.auto, args.dir)
    elif args.mode == 'match':
        draw_match_to_sequence(vid_reader, det_dict,
                               gt_dict, num_to_draw=args.num, auto=args.auto, out_dir=args.dir)
    elif args.mode == 'fp':
        draw_false_positive(vid_reader, det_dict, gt_dict,
                            num_to_draw=args.num, auto=args.auto, out_dir=args.dir)
    if args.stitch:
        stitch_to_video(args.dir, (vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH), vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, help="path to the video")
    ap.add_argument("mode", choices=['default', 'match', 'fp'],
                    help="'default' simply draws bboxes, 'match' matches each det-gt pair,"
                         " 'fp' draw false positives only")
    ap.add_argument("-d", "--det", type=str, required=False,
                    help="detection results, COCO formatted")
    ap.add_argument("-g", "--gt", type=str, required=False,
                    help="ground truth, COCO formatted")
    ap.add_argument("-n", "--num", type=int, required=False, default=0,
                    help="number of frames to draw starting from 0")
    ap.add_argument("--dir", type=str, required=False, default=None,
                    help="directory to save drawn pictures")
    ap.add_argument("-a", "--auto", type=bool, required=False, default=True,
                    help="whether to auto process or manually inspect each picture")
    ap.add_argument("-s", "--stitch", type=bool, required=False, default=True,
                    help="whether to stitch pictures to a video")

    args = ap.parse_args()
    run_args(args)
