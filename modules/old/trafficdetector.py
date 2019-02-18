# -*- coding:utf8 -*-
import sys
from config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.CAFFE_ROOT)

import caffe
from libs.utils import objs_sort_by_center
from pypriv.nnutils.caffeutils import Detector
from pypriv import variable as V


class TrafficDetector:
    def __init__(self, gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        det_net = caffe.Net(cfg_priv.Traffic.DEPLOY, cfg_priv.Traffic.WEIGHTS, caffe.TEST)

        self.D = Detector(det_net, mean=cfg_priv.Traffic.PIXEL_MEANS, std=cfg_priv.Traffic.PIXEL_STDS,
                          scales=(540,), max_sizes=(960,), preN=6000, postN=300, conf_thresh=0.6,
                          color_map=V.COLORMAP21, class_map=V.SHANXI7_CLASS)

    def __call__(self, img):
        objs = self.D.det_im(img)
        if objs is None:
            return None
        else:
            return objs_sort_by_center(objs)

    def __del__(self):
        print self.__class__.__name__
