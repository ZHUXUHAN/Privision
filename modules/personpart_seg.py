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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms

from pet.models.sync_batchnorm import user_scattered_collate, async_copy_to
from pet.utils import variable as V
from pet.semseg.core.config import cfg, merge_cfg_from_file
from pet.semseg.modeling.model_builder import ModelBuilder, SegmentationModule
from pet.semseg.utils.misc import colorEncode
from pet.semseg.utils import as_numpy

merge_cfg_from_file(osp.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT, cfg_priv.MODULES.PPS.CFG))
# pprint.pprint(cfg)

cudnn.benchmark = True


class PersonPartSeg:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)

        # Network Builders
        weights_encoder, weights_decoder = get_weights()
        builder = ModelBuilder()
        net_encoder = builder.build_encoder(arch=cfg.MODEL.ENCODER,
                                            weights=weights_encoder)
        net_decoder = builder.build_decoder(arch=cfg.MODEL.DECODER,
                                            fc_dim=cfg.MODEL.FC_DIM,
                                            num_class=cfg.MODEL.NUM_CLASSES,
                                            weights=weights_decoder,
                                            use_softmax=True)
        real_ignore_label = cfg.DATASET.IGNORE_LABEL + cfg.DATASET.LABEL_SHIFT
        crit = nn.NLLLoss(ignore_index=real_ignore_label)

        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        self.segmentation_module.cuda()
        self.segmentation_module.eval()

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=cfg.PIXEL_MEANS, std=cfg.PIXEL_STDS)
        ])

    def __call__(self, img):
        batch_data = self.data_preproc(img)

        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            segSize = (batch_data['img_ori'].shape[0], batch_data['img_ori'].shape[1])
            pred = torch.zeros(1, cfg.MODEL.NUM_CLASSES, segSize[0], segSize[1])
            pred = Variable(pred).cuda()

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, self.gpu_id)

                # forward pass
                pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
                pred += pred_tmp / len(cfg.TEST.AUG.INPUT_SIZE)
            pred = as_numpy(pred)[0].transpose((1, 2, 0))

            if cfg.TEST.AUG.H_FLIP:
                _pred = torch.zeros(1, cfg.MODEL.NUM_CLASSES, segSize[0], segSize[1])
                _pred = Variable(_pred).cuda()
                for _img in img_resized_list:
                    _img = as_numpy(_img)
                    _img = _img[0].transpose((1, 2, 0))
                    _img = cv2.flip(_img, 1)
                    _img = _img.transpose((2, 0, 1))
                    _img = torch.from_numpy(np.asarray([_img]))
                    _feed_dict = batch_data.copy()
                    _feed_dict['img_data'] = _img
                    del _feed_dict['img_ori']
                    del _feed_dict['info']
                    _feed_dict = async_copy_to(_feed_dict, self.gpu_id)

                    # forward pass
                    _pred_tmp = self.segmentation_module(_feed_dict, segSize=segSize)
                    _pred += _pred_tmp / len(cfg.TEST.AUG.INPUT_SIZE)
                _pred = as_numpy(_pred)[0].transpose((1, 2, 0))
                _pred = cv2.flip(_pred, 1)
                pred = (pred + _pred) * 0.5

            preds = np.argmax(pred, axis=2)

        print('[{}]'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return visualize_result(batch_data['img_ori'], preds)
        # return batch_data['img_ori']

    def data_preproc(self, img):
        # img = img[:, :, ::-1]  # BGR to RGB!!!
        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in cfg.TEST.AUG.INPUT_SIZE:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        cfg.TEST.AUG.MAX_SIZE / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, cfg.MODEL.PADDING_CONSTANT)
            target_width = round2nearest_multiple(target_width, cfg.MODEL.PADDING_CONSTANT)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = ''
        return output

    def __del__(self):
        print(self.__class__.__name__)


# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


def get_weights():
    if os.path.exists(cfg.TEST.WEIGHTS_ENCODER) and os.path.exists(cfg.TEST.WEIGHTS_DECODER):
        weights_encoder, weights_decoder = cfg.TEST.WEIGHTS_ENCODER, cfg.TEST.WEIGHTS_DECODER
    else:
        weights_encoder = os.path.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT,
                                       cfg.CKPT,
                                       'model_encoder_latest.pth')
        weights_decoder = os.path.join(cfg_priv.GLOBAL.PYTORCH_EVERYTHING_ROOT,
                                       cfg.CKPT,
                                       'model_decoder_latest.pth')
    return weights_encoder, weights_decoder


def visualize_result(img, preds):
    colors = V.COLORMAP20

    label_shift = 0
    color_mode = 'RGB'
    ignore = 255

    # prediction
    pred_color = colorEncode(preds, colors, label_shift=label_shift, mode=color_mode, ignore=ignore)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1).astype(np.uint8)
    return im_vis
