from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import pprint
import datetime

this_dir = osp.dirname(__file__)
sys.path.append(this_dir + '/..')

from priv_config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT)

# Numerical libs
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image

import pet.models.imagenet as models
from pet.utils.misc import weight_filler
from pet.utils.transforms import bgr2rgb, normalize, pil_scale, center_crop
from pet.cls.core.config import cfg, merge_cfg_from_file

merge_cfg_from_file(osp.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT, cfg_priv.MODULES.OBJCLS.CFG))
# pprint.pprint(cfg)

cudnn.benchmark = True
_GRAY = [218, 227, 218]
_GREEN = [18, 127, 15]
_WHITE = [255, 255, 255]


class ObjCls:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)

        # Create model
        self.model = models.__dict__[cfg.MODEL.CONV_BODY]()
        print(self.model)

        # Load pre-train model
        model_weights = get_weights()
        pretrained_dict = torch.load(model_weights)
        # pretrained_dict = torch.load(model_weights, map_location='cpu')
        model_dict = self.model.state_dict()
        updated_dict, match_layers, mismatch_layers = weight_filler(pretrained_dict, model_dict)
        model_dict.update(updated_dict)
        self.model.load_state_dict(model_dict)
        print('==> Mismatch: ', mismatch_layers)

        self.model.cuda()
        self.model.eval()

    def __call__(self, img, show_cls=True, topN=max(cfg.TEST.TOP_K)):
        # img = bgr2rgb(img)
        scale_im = pil_scale(Image.fromarray(bgr2rgb(img)), cfg.TRAIN.AUG.BASE_SIZE)
        normalized_im = normalize(np.asarray(scale_im) / 255.0, mean=cfg.PIXEL_MEANS, std=cfg.PIXEL_STDS)

        input_data = [center_crop(normalized_im, crop_size=cfg.TRAIN.AUG.CROP_SIZE)]
        input_data = np.asarray(input_data, dtype=np.float32)
        input_data = input_data.transpose(0, 3, 1, 2)

        vis = cv2.resize(img, cfg_priv.GLOBAL.IM_SHOW_SIZE)
        with torch.no_grad():
            input_data = torch.from_numpy(input_data)
            input_data = input_data.cuda()
            outputs = self.model(input_data)
            scores = nn.functional.softmax(outputs, dim=1)
            scores = scores.data.cpu().numpy()
            scores = np.sum(scores, axis=0)
            idx = np.argsort(-scores)
            if show_cls:
                for i in range(topN):
                    pos = (30, i * 30 + 60)
                    cls_str = '{}: {:.6f}'.format(idx[i], scores[idx[i]])
                    vis = vis_class(vis, pos, cls_str)

        return vis


def get_weights(mode='best'):
    if os.path.exists(cfg.TEST.WEIGHTS):
        weights = cfg.TEST.WEIGHTS
    else:
        weights = os.path.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT,
                               cfg.CKPT,
                               'model_{}.pth'.format(mode))
    return weights


def vis_class(img, pos, class_str, font_scale=0.75):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img
