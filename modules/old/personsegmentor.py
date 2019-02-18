# -*- coding:utf8 -*-
import sys
from config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.CAFFE_ROOT)

import caffe
from pypriv.nnutils.caffeutils import Segmentor
from pypriv import variable as V


class PersonSegmentor:
    def __init__(self, gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        seg_net = caffe.Net(cfg_priv.PersonSeg.DEPLOY, cfg_priv.PersonSeg.WEIGHTS, caffe.TEST)

        self.S = Segmentor(seg_net, class_num=7, mean=cfg_priv.PersonSeg.PIXEL_MEANS, std=cfg_priv.PersonSeg.PIXEL_STDS,
                           scales=(384,), crop_size=384)

    def __call__(self, img):
        mask = self.S.seg_im(img)

        return mask

    def __del__(self):
        print self.__class__.__name__
