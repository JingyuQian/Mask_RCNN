import os
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import tensorflow as tf
import numpy as np

from mrcnn import utils


class MaskRCNN():

    def __init__(self, mode, config, model_dir):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.model = self.build(mode=mode, config=config)
        self.model_shortcut = self.build_shortcut(mode=mode, config=config)

    def build(self, mode, config):
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != (h / 2 ** 6) or w / 2 ** 6 != int(w / w ** 6):
            raise Exception("Image size must be dividable by 2 at least"
                            " 6 times to avoid fractions"
                            " when downscaling and upscaling.")

        input_image = KL.Input(
            shape=[None, None, 3], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")

        # Discard training mode for the moment. Focus on inference.

        input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        _, C2, C3, C4, C5 = renet_graph(input_image, config.BACKBONE,
                                        stage5=True, train_bn=config.TRAIN_BN)
        P5 = KL.Conv2D(256, (1, 1), name="fpn_c5p5")(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(256, (1, 1), name="fpn_c4p4")(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        #
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        anchors = input_anchors

        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), 256)

        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))

        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        proposal_count = config.POST_NMS_ROIS_INFERENCE

        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, align = \
            fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                 config.POOL_SIZE, config.NUM_CLASSES, train_bn=config.TRAIN_BN)

        detections = DetectionLayer(config, name="mrcnn_detection")(
            [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
        mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                          input_image_meta, config.MASK_POOL_SIZE, config.NUM_CLASSES, train_bn=config.TRAIN_BN)

        model = KM.Model(inputs=[input_image, input_image_meta, input_anchors],
                         outputs=[detections, mrcnn_class, mrcnn_mask,
                                  mrcnn_class_logits, rpn_rois, align],
                         name="mask_rcnn")
