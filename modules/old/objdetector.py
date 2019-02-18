# -*- coding:utf8 -*-
import sys
from config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.CAFFE_ROOT)

import caffe
from libs.utils import objs_sort_by_center
from pypriv.nnutils.caffeutils import Detector
from pypriv import variable as V


class ObjDetector:
    def __init__(self, gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        det_net = caffe.Net(cfg_priv.ObjDet.DEPLOY, cfg_priv.ObjDet.WEIGHTS, caffe.TEST)

        self.D = Detector(det_net, mean=cfg_priv.ObjDet.PIXEL_MEANS, std=cfg_priv.ObjDet.PIXEL_STDS,
                          scales=(480,), max_sizes=(960,), preN=1000, postN=100, conf_thresh=0.7,
                          color_map=V.COLORMAP81, class_map=V.COCO_CLASS81)

    def __call__(self, img):
        objs = self.D.det_im(img)
        if objs is None:
            return None
        else:
            return objs_sort_by_center(objs)

    def __del__(self):
        print self.__class__.__name__
