# -*- coding:utf8 -*-
import sys
from config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.CAFFE_ROOT)

import caffe
from libs.utils import objs_sort_by_center
from pypriv.nnutils.caffeutils import Detector
from pypriv import variable as V


class PersonDetector:
    def __init__(self, gpu_id=0, fastface=False):
        self.fastface = fastface
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        det_net = caffe.Net(cfg_priv.PersonDet.DEPLOY, cfg_priv.PersonDet.WEIGHTS, caffe.TEST)

        self.D = Detector(det_net, mean=cfg_priv.PersonDet.PIXEL_MEANS, std=cfg_priv.PersonDet.PIXEL_STDS,
                          scales=(480,), max_sizes=(800,), preN=500, postN=50, conf_thresh=0.7,
                          color_map={0: [192, 0, 192], 1: [64, 64, 255]}, class_map=V.PERSON_CLASS)

        if self.fastface:
            face_net = caffe.Net(cfg_priv.FastFace.DEPLOY, cfg_priv.FastFace.WEIGHTS, caffe.TEST)
            self.FastFace = Detector(face_net, mean=cfg_priv.FastFace.PIXEL_MEANS, std=cfg_priv.FastFace.PIXEL_STDS,
                                     scales=(480,), max_sizes=(800,), preN=500, postN=50, conf_thresh=0.5,
                                     color_map={0: [192, 0, 192], 1: [255, 64, 64]}, class_map=V.FACE_CLASS)

    def __call__(self, img):
        objs = self.D.det_im(img)
        if self.fastface:
            face_objs = self.FastFace.det_im(img)
            objs.extend(face_objs)
        if objs is None:
            return None
        else:
            return objs_sort_by_center(objs)

    def __del__(self):
        print self.__class__.__name__
