from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import time
import pprint
from collections import defaultdict
from priv_config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT)

import cv2
import torch
import torch.backends.cudnn as cudnn

import pet1.maskrcnn.nn as mynn
import pet1.maskrcnn.datasets.dummy_datasets as datasets
import pet1.maskrcnn.utils.net as net_utils
import pet1.maskrcnn.utils.vis as vis_utils
from pet1.maskrcnn.core.config import cfg, merge_cfg_from_file, assert_and_infer_cfg
from pet1.maskrcnn.core.test import im_detect_all
from pet1.maskrcnn.modeling.model_builder import Generalized_RCNN
from pet1.maskrcnn.utils.detectron_weight_helper import load_detectron_weight
from pet1.maskrcnn.utils.timer import Timer

cv2.ocl.setUseOpenCL(False)

merge_cfg_from_file(osp.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT, cfg_priv.MODULES.PPA.CFG))
pprint.pprint(cfg)

dataset = datasets.get_dataset(cfg.TEST.DATASETS[0])
cudnn.benchmark = True


class PersonParsing:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)

        assert_and_infer_cfg()

        self.maskRCNN = Generalized_RCNN()
        self.maskRCNN.cuda()

        # Load trained model
        weights = get_weights()
        print("==> Using trained model: '{}'".format(weights))

        _, ext = os.path.splitext(weights)
        if ext == '.pkl':
            load_detectron_weight(self.maskRCNN, weights)
        elif ext == '.pth':
            checkpoint = torch.load(weights, map_location=lambda storage, loc: storage)
            net_utils.load_ckpt(self.maskRCNN, checkpoint['model'])

        self.maskRCNN = mynn.DataParallel(self.maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                          minibatch=True, device_ids=[self.gpu_id])  # only support single GPU
        self.maskRCNN.eval()

    def __call__(self, img, show_box=True, show_kpts=True, show_mask=True):
        timers = defaultdict(Timer)

        t = time.time()
        cls_boxes_i, cls_segms_i, cls_keyps_i, cls_parss_i, cls_uvs_i = im_detect_all(self.maskRCNN, img, timers=timers)
        t1 = time.time()
        for k, v in timers.items():
            print(' | {}: {:.3f}s'.format(k, v.average_time))
        print('-Inference time: {:.3f}s'.format(time.time() - t))

        vis_im = vis_utils.vis_one_image_opencv(
            img,
            cls_boxes_i,
            segms=cls_segms_i,
            keypoints=cls_keyps_i,
            parsing=cls_parss_i,
            uv=cls_uvs_i,
            dataset=dataset
        )

        print('-Visulization time: {:.3f}s'.format(time.time() - t1))
        print('Total time: {:.3f}s'.format(time.time() - t))
        print('--------------------------')

        return vis_im

    def __del__(self):
        print(self.__class__.__name__)


def get_weights(mode='final'):
    if os.path.exists(cfg.TEST.WEIGHTS):
        weights = cfg.TEST.WEIGHTS
    else:
        model_root = os.path.dirname(cfg_priv.MODULES.PPA.CFG)
        # pkl first
        if os.path.exists(os.path.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT, model_root, 'model_{}.pkl'.
                          format(mode))):
            weights = os.path.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT, model_root, 'model_{}.pkl'.format(mode))
        else:
            final_step = int(cfg.SOLVER.MAX_ITER - 1)
            weights = os.path.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT, model_root, 'model_step{}.pth'.
                                   format(final_step))

    return weights
