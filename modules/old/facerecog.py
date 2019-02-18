# -*- coding:utf8 -*-
import sys
from config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.CAFFE_ROOT)

import caffe

import numpy as np
from libs.utils import objs_sort_by_center
from pypriv.nnutils.caffeutils import Detector, Identity
from pypriv import variable as V
from pypriv import transforms as T


class FaceRecog:
    def __init__(self, gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        det_net = caffe.Net(cfg_priv.FastFace.DEPLOY, cfg_priv.FastFace.WEIGHTS, caffe.TEST)
        self.FastFace = Detector(det_net, mean=cfg_priv.FastFace.PIXEL_MEANS, std=cfg_priv.FastFace.PIXEL_STDS,
                                 scales=(300,), max_sizes=(400,), preN=500, postN=50, conf_thresh=0.5,
                                 color_map={0: [192, 0, 192], 1: [255, 64, 64]}, class_map=V.FACE_CLASS)

        id_net = caffe.Net(cfg_priv.FaceID.DEPLOY, cfg_priv.FaceID.WEIGHTS, caffe.TEST)
        self.I = Identity(id_net, mean=cfg_priv.FaceID.PIXEL_MEANS, std=cfg_priv.FaceID.PIXEL_STDS,
                          base_size=256, crop_size=224, crop_type='center', prob_layer='classifier',
                          gallery=np.load(cfg_priv.FaceID.GALLERY))
        self.name_list = open(cfg_priv.FaceID.GALLERY_NAMES).readlines()
        print self.name_list

    def __call__(self, img):
        img_det = img.copy()
        objs = self.FastFace.det_im(img_det)
        if objs is None:
            return None
        else:
            faces = []
            for i in objs:
                face_box = T.extend_bbox(img, i['bbox'])
                crop_face = T.bbox_crop(img, face_box)
                faces.append(crop_face)
            if len(faces) == 0:
                return None
            ids = self.I.cls_batch(faces)
            for _ in xrange(len(ids)):
                sims = self.I.one2n_identity(ids[_])
                idx = np.asarray(sims).argsort()[-1]
                score = sims[idx]
                if score >= 0.5:
                    objs[_]['attri'] = dict(name=self.name_list[idx].strip())
                    # score='{:.3f}'.format(score),
                    # age='unknown')
                else:
                    objs[_]['attri'] = dict(name='--------')  # , score=str(0.0), age='unknown')
            return objs

    def __del__(self):
        print self.__class__.__name__
