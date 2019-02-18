from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from collections import defaultdict
import numpy as np
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import sys
import time
from config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.CAFFE2_ROOT)
sys.path.append(cfg_priv.GLOBAL.DETECTRON_ROOT)

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
utils.logging.setup_logging(__name__)

cfg.TEST.WEIGHTS = '/home/priv/workspace/detectron/models/priv/e2e_mask-keypoint_rcnn_R-50-FPN_s1x' \
                   '/model_final.pkl'
cfg.NUM_GPUS = 1
merge_cfg_from_file('/home/priv/workspace/detectron/models/priv/e2e_mask-keypoint_rcnn_R-50-FPN_s1x'
                    '/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml')
assert_and_infer_cfg(cache_urls=False)
MODEL = infer_engine.initialize_model_from_cfg(gpu_id=0)

DUMMY = dummy_datasets.get_coco_dataset()


def draw_bbox_mask_pose(im, boxes, segms=None, keypoints=None, thresh=0.7, kp_thresh=2, show_box=False, dataset=None,
                        show_class=False):
    vis_result = vis_utils.vis_one_image_opencv(
        im[:, :, ::-1],  # BGR -> RGB for visualization
        boxes,
        segms=segms,
        keypoints=keypoints,
        thresh=thresh,
        kp_thresh=kp_thresh,
        show_box=show_box,
        dataset=dataset,
        show_class=show_class
    )

    return vis_result


class C2MaskPose:
    def __init__(self, gpu_id=0):
        pass

        # self.model = infer_engine.initialize_model_from_cfg(gpu_id=gpu_id)

    def __call__(self, img):
        timers = defaultdict(Timer)
        im = cv2.resize(img, (540, 400))
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                MODEL, im, None, timers=timers
            )
            # print cls_boxes, cls_segms, cls_keyps

        return cls_boxes, cls_segms, cls_keyps

    def __del__(self):
        print(self.__class__.__name__)

