
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import pdb

ap = argparse.ArgumentParser()
ap.add_argument("gt", type=str, help="ground-truth coco")
ap.add_argument("-d", "--det", type=str, required=False,
                help="detection results coco")
ap.add_argument("-i", "--input", type=str, required=False, help="video input")
ap.add_argument("-v", "--vis", type=str, required=False,
                help="visualize results")
args = ap.parse_args()

# initialize COCO ground truth api
cocoGt = COCO(args.gt)

if args.det:

    cocoDt = cocoGt.loadRes(args.det)
    imgIds = sorted(cocoGt.getImgIds())

    annType = ['segm', 'bbox', 'keypoints']

    cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
    cocoEval.params.imgIds = imgIds

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    cocoEval.params.areaRngLbl = ['all']
    cocoEval.params.iouThrs = [0.5]
    cocoEval.params.maxDets = [100]

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
