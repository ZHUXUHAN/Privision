import os
import sys
from config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.CAFFE_ROOT)

import PyOpenPose as OP
import caffe
import numpy as np
import cv2
from pypriv import transforms as T
from pypriv import variable as V


class PoseEstimator:
    def __init__(self):
        self.input_size = (320, 240)  # w, h
        self.output_size = (320, 240)  # w, h
        self.op = OP.OpenPose(self.input_size, (240, 240), self.output_size,
                              "COCO", cfg_priv.GLOBAL.OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                              False)

    def __call__(self, img):
        scale_im, scale_factor = T.scale_by_target(img, target_size=self.input_size[::-1])
        pad_im = T.padding_im(scale_im, target_size=self.input_size[::-1])

        self.op.detectPose(pad_im)
        pose_kpts = self.op.getKeypoints(self.op.KeypointType.POSE)[0]
        if pose_kpts is not None:
            a, b, c = pose_kpts.shape
            kpts = np.zeros((a, b + 1, c), np.float)
            pose_kpts[:, :, 0:2] /= scale_factor
            kpts[:, 0:b, :] = pose_kpts
            for i in xrange(a):
                if kpts[i][8][2] * kpts[i][11][2] == 0:
                    continue
                kpts[i][b] = (kpts[i][8] + kpts[i][11]) / 2
            return kpts
        else:
            return None

    def __del__(self):
        print self.__class__.__name__
