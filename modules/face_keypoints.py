from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
from priv_config import cfg_priv

sys.path.append(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT)

# Numerical libs
import cv2
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import pet4.models.imagenet as models
from pet4.utils.misc import weight_filler
from pet4.utils.transforms import bgr2rgb, normalize
from pet4.cls.core.config import cfg, merge_cfg_from_file

merge_cfg_from_file(osp.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT, cfg_priv.MODULES.FKP.CFG))
# pprint.pprint(cfg)

cudnn.benchmark = True
_GRAY = [218, 227, 218]
_GREEN = [18, 127, 15]
_WHITE = [255, 255, 255]

extend_scale = 0.85
bbox_adapt_R = np.array([[0.355067475559597, 0.119262390148535, -0.355067475560677, -0.119262390148450],
                         [0.332986547478306, 0.246449822265788, -0.332986547479770, -0.246449822265689],
                         [-0.331071700057979, -0.146350441272887, 0.331071700059095, 0.146350441272787],
                         [-0.145853845175441, -0.399418765180059, 0.145853845176905, 0.399418765179917]])


class FaceKpts:
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

    def __call__(self, img, cls_boxes_i, face_idx=1, min_thresh=0.8, min_abso_area=50*50, min_rela_area=0.01,
                 max_aspect_ratio=2.0):
        # img = bgr2rgb(img)
        normalized_im = normalize(np.asarray(img) / 255.0, mean=cfg.PIXEL_MEANS, std=cfg.PIXEL_STDS)
        img_area = img.shape[0] * img.shape[1]

        # boxes processing
        crop_imgs = []
        ldmk_bboxes = []
        bboxes = cls_boxes_i[face_idx]
        bboxes = boxes_filter(bboxes, img_area=img_area, min_thresh=min_thresh, min_abso_area=min_abso_area,
                              min_rela_area=min_rela_area, max_aspect_ratio=max_aspect_ratio)
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            ldmk_bbox = bbox_adapt(np.asarray(bbox))
            ldmk_bbox = crop_bbox(ldmk_bbox)
            ldmk_bboxes.append(ldmk_bbox)
            crop_img = sample_img_ignore_bound(ldmk_bbox, normalized_im)
            crop_img = cv2.resize(crop_img, (120, 120), interpolation=cv2.INTER_CUBIC)
            crop_img = crop_img.astype(float)
            crop_imgs.append(crop_img)

        if len(crop_imgs) == 0:
            return None
        input_data = np.asarray(crop_imgs, dtype=np.float32)
        input_data = input_data.transpose(0, 3, 1, 2)

        with torch.no_grad():
            input_data = torch.from_numpy(input_data).float()
            input_data = input_data.cuda()
            outputs = self.model(input_data)
            feature = outputs.data.cpu().numpy()
            norm_pts84 = feature.reshape([-1, 2, cfg.MODEL.NUM_CLASSES // 2])

            all_pts = []
            for i in range(len(norm_pts84)):
                norm_pts = de_norm_kpt(ldmk_bboxes[i], norm_pts84[i])
                all_pts.append(norm_pts)

        return np.asarray(all_pts)


def get_weights(mode='ldmk10_84'):
    if os.path.exists(cfg.TEST.WEIGHTS):
        weights = cfg.TEST.WEIGHTS
    else:
        weights = os.path.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT,
                               cfg.CKPT,
                               '{}.pth'.format(mode))
    return weights


def boxes_filter(boxes, img_area=800*1000, min_thresh=0.8, min_abso_area=50*50, min_rela_area=0.01,
                 max_aspect_ratio=2.0):
    boxes_fl = []
    for i in range(len(boxes)):
        box = boxes[i][:4]
        score = boxes[i][-1]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        box_aspect_ratio = max((box[2] - box[0]) / float(box[3] - box[1]), (box[3] - box[1]) / float(box[2] - box[0]))
        if score < min_thresh:
            print('min_thresh', score)
            continue
        if 0 < min_abso_area and box_area < min_abso_area:
            print('min_abso_area', box_area)
            continue
        if 0 < min_rela_area <= 1 and box_area / float(img_area) < min_rela_area:
            print('min_rela_area', box_area / float(img_area))
            continue
        if 1 <= max_aspect_ratio and box_aspect_ratio > max_aspect_ratio:
            print('max_aspect_ratio', box_aspect_ratio)
            continue
        boxes_fl.append(box)

    return np.asarray(boxes_fl)


def bbox_adapt(bbox_dt):
    # bbox norm
    centers = np.array([(bbox_dt[0] + bbox_dt[2]) / 2, (bbox_dt[1] + bbox_dt[3]) / 2])
    llengthes = ((bbox_dt[2] - bbox_dt[0]) + (bbox_dt[3] - bbox_dt[1])) / 4
    bbox_norm = np.array([bbox_dt[0] - centers[0], bbox_dt[1] - centers[1],
                          bbox_dt[2] - centers[0], bbox_dt[3] - centers[1]])
    bbox_norm = bbox_norm / llengthes

    # bbox adapt
    bbox_dt = np.transpose(np.dot(bbox_adapt_R, np.transpose(bbox_norm)))
    # bbox denorm
    bbox_dt = bbox_dt * llengthes
    bbox_dt = np.array([bbox_dt[0] + centers[0], bbox_dt[1] + centers[1],
                        bbox_dt[2] + centers[0], bbox_dt[3] + centers[1]])

    return bbox_dt


def crop_bbox(bbox_dt):
    half_width = bbox_dt[2] - bbox_dt[0]
    half_height = bbox_dt[3] - bbox_dt[1]
    llength = math.sqrt(half_width * half_width + half_height * half_height)
    llength = round(llength * extend_scale)
    centers = np.array([(bbox_dt[0] + bbox_dt[2]) / 2, (bbox_dt[1] + bbox_dt[3]) / 2])
    x1 = centers[0] - llength * 0.5
    y1 = centers[1] - llength * 0.43
    x2 = x1 + llength
    y2 = y1 + llength
    roi_box = np.array([x1, y1, x2, y2])

    return roi_box


# denorm pts5 via ldmk bbox
def de_norm_kpt(bbox, pt):
    pts_num = pt.shape[1]
    for i in range(pts_num):
        pt[0, i] = pt[0, i] * (bbox[2] - bbox[0]) + bbox[0]
        pt[1, i] = pt[1, i] * (bbox[3] - bbox[1]) + bbox[1]

    return pt


def sample_img_ignore_bound(roi, img):
    x1 = int(round(roi[0]))
    y1 = int(round(roi[1]))
    x2 = int(round(roi[2]))
    y2 = int(round(roi[3]))

    if (y2 - y1) > (x2 - x1):
        x2 += 1
    elif (y2 - y1) < (x2 - x1):
        y2 += 1

    img1 = np.zeros([y2 - y1, x2 - x1, 3])
    x1_ = max(x1, 0)
    y1_ = max(y1, 0)
    x2_ = min(x2, img.shape[1])
    y2_ = min(y2, img.shape[0])

    assert (y2 - y1) == (x2 - x1)

    img1[(y1_ - y1):(y2_ - y1), (x1_ - x1):(x2_ - x1), :] = img[y1_:y2_, x1_:x2_, :]
    return img1


def vis_kpts(img, kpts, color=(255, 255, 255), size=2):
    face_num, _, kpt_num = kpts.shape
    for i in range(face_num):
        for j in range(kpt_num):
            cv2.circle(img, (int(round(kpts[i, 0, j])), int(round(kpts[i, 1, j]))), size, color, -1)
