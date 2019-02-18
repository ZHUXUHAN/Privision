from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
import copy
import os
import os.path as osp
import numpy as np
import yaml

"""config system.
This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.
"""

cur_pth = os.getcwd()


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


__C = AttrDict()

cfg_priv = __C

# ---------------------------------------------------------------------------- #
# Global options
# ---------------------------------------------------------------------------- #
__C.GLOBAL = AttrDict()

__C.GLOBAL.BANNER = 'PriVision System'

__C.GLOBAL.CAFFE2_ROOT = '/home/user/workspace/caffe2/build'

__C.GLOBAL.DETECTRON_ROOT = '/home/user/workspace/Detectron'

__C.GLOBAL.PYTORCH_EVERYTHING_ROOT = '/home/user/workspace/PytorchEveryThing'

__C.GLOBAL.IM_SHOW_SIZE = (1296, 960)

__C.GLOBAL.VIDEO1 = './files/news.avi'

__C.GLOBAL.VIDEO2 = './files/shanxi_traffic/0500320.avi'

__C.GLOBAL.VIDEO3 = './files/shanxi_traffic/0500320.avi'

__C.GLOBAL.F_MODEL1 = False

__C.GLOBAL.F_MODEL2 = False

__C.GLOBAL.F_MODEL3 = False

__C.GLOBAL.F_MODEL4 = False

__C.GLOBAL.F_MODEL5 = False

__C.GLOBAL.SAVE_VIDEO_PATH = './files/save'

__C.GLOBAL.SAVE_VIDEO_MAX_SECOND = 1800 * 20

__C.GLOBAL.SAVE_VIDEO_FPS = 20

__C.GLOBAL.SAVE_VIDEO_SIZE = (1296, 960)

# ---------------------------------------------------------------------------- #
# Icons options
# ---------------------------------------------------------------------------- #
__C.ICONS = AttrDict()

__C.ICONS.LOGO = cur_pth + '/icons/pose_icon.png'

__C.ICONS.BACKGROUND = cur_pth + '/icons/back_large.jpg'

# top
__C.ICONS.TOP = AttrDict()

__C.ICONS.TOP.SIZE = (110, 20)

__C.ICONS.TOP.LEFT1 = cur_pth + '/icons/icon-top/realtime-mode-bright.png'

__C.ICONS.TOP.LEFT2 = cur_pth + '/icons/icon-top/playback1-bright.png'

__C.ICONS.TOP.LEFT3 = cur_pth + '/icons/icon-top/playback2-bright.png'

__C.ICONS.TOP.LEFT4 = cur_pth + '/icons/icon-top/picture-mode-bright.png'

__C.ICONS.TOP.LEFT5 = cur_pth + '/icons/icon-top/loading-mode-bright.png'

# left
__C.ICONS.LEFT = AttrDict()

__C.ICONS.LEFT.SIZE = (110, 70)

__C.ICONS.LEFT.TOP1 = cur_pth + '/icons/icon-left/play-bright.png'

__C.ICONS.LEFT.TOP2 = cur_pth + '/icons/icon-left/pause-bright.png'

__C.ICONS.LEFT.TOP3 = cur_pth + '/icons/icon-left/record-bright.png'

__C.ICONS.LEFT.TOP4 = cur_pth + '/icons/icon-left/empty-bright.png'

__C.ICONS.LEFT.TOP5 = cur_pth + '/icons/icon-left/setting-bright.png'

__C.ICONS.LEFT.TOP6 = cur_pth + '/icons/icon-left/exit.png'

# ---------------------------------------------------------------------------- #
# Right button function type
# ---------------------------------------------------------------------------- #

__C.FUNC_OPT = AttrDict()

# func1
__C.FUNC_OPT.FUNC1 = AttrDict()
__C.FUNC_OPT.FUNC1.MODULE = 'person_ins'
__C.FUNC_OPT.FUNC1.CLASS = 'PersonIns'
__C.FUNC_OPT.FUNC1.GPU_ID = 0
__C.FUNC_OPT.FUNC1.ICON = cur_pth + '/icons/icon-right/human-pose-bright.png'
__C.FUNC_OPT.FUNC1.ICON_SIZE = (110, 70)

# func2
__C.FUNC_OPT.FUNC2 = AttrDict()
__C.FUNC_OPT.FUNC2.MODULE = 'personpart_seg'
__C.FUNC_OPT.FUNC2.CLASS = 'PersonPartSeg'
__C.FUNC_OPT.FUNC2.GPU_ID = 1
__C.FUNC_OPT.FUNC2.ICON = cur_pth + '/icons/icon-right/human-seg-bright.png'
__C.FUNC_OPT.FUNC2.ICON_SIZE = (110, 70)

# func3
__C.FUNC_OPT.FUNC3 = AttrDict()
__C.FUNC_OPT.FUNC3.MODULE = 'personpart_det'
__C.FUNC_OPT.FUNC3.CLASS = 'PersonPartDet'
__C.FUNC_OPT.FUNC3.GPU_ID = 2
__C.FUNC_OPT.FUNC3.ICON = cur_pth + '/icons/icon-right/human-detection-bright.png'
__C.FUNC_OPT.FUNC3.ICON_SIZE = (110, 70)

# func4
__C.FUNC_OPT.FUNC4 = AttrDict()
__C.FUNC_OPT.FUNC4.MODULE = 'object_cls'
__C.FUNC_OPT.FUNC4.CLASS = 'ObjCls'
__C.FUNC_OPT.FUNC4.GPU_ID = 3
__C.FUNC_OPT.FUNC4.ICON = cur_pth + '/icons/icon-right/road-analysis-bright.png'
__C.FUNC_OPT.FUNC4.ICON_SIZE = (110, 70)

# func5
__C.FUNC_OPT.FUNC5 = AttrDict()
__C.FUNC_OPT.FUNC5.MODULE = ''
__C.FUNC_OPT.FUNC5.CLASS = ''
__C.FUNC_OPT.FUNC5.GPU_ID = 0
__C.FUNC_OPT.FUNC5.ICON = cur_pth + '/icons/icon-right/face-recog-bright.png'
__C.FUNC_OPT.FUNC5.ICON_SIZE = (110, 70)

# ---------------------------------------------------------------------------- #
# Modules options
# ---------------------------------------------------------------------------- #

__C.MODULES = AttrDict()

# imcls: image_classification
__C.MODULES.OBJCLS = AttrDict()

__C.MODULES.OBJCLS.CFG = 'ckpts/cls/imagenet/resnet18_1x64d/resnet18_1x64d.yaml'

# ppo: person_pose
__C.MODULES.PPO = AttrDict()

__C.MODULES.PPO.CFG = 'ckpts/mscoco_person/mscoco/e2e_mask-keypoint_rcnn_R-50-FPN_s1x_ms6/' \
                      'e2e_mask-keypoint_rcnn_R-50-FPN_s1x_ms6.yaml'

# ppa: person_parsing
__C.MODULES.PPA = AttrDict()

__C.MODULES.PPA.CFG = 'ckpts/maskrcnn/CIHP/e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPP-PBD-2xLW_3x_COCO_ms/' \
                      'e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPP-PBD-2xLW_3x_COCO_ms.yaml'

# pdp: person_densepose
__C.MODULES.PDP = AttrDict()

__C.MODULES.PDP.CFG = 'ckpts/maskrcnn/DensePose_COCO/e2e_parsing_rcnn_R-50-FPN-PSS-ERR_s1x_ms/' \
                      'e2e_parsing_rcnn_R-50-FPN-PSS-ERR_s1x_ms.yaml'

# ppd: personpart_det
__C.MODULES.PPD = AttrDict()

__C.MODULES.PPD.CFG = 'ckpts/wider_face/mscoco/e2e_faster_rcnn_R-50-FPN_2x_ms/' \
                      'e2e_faster_rcnn_R-50-FPN_2x_ms.yaml'


# pfd: personface_det
__C.MODULES.PFD = AttrDict()

__C.MODULES.PFD.CFG = 'ckpts/wider_face/mscoco/e2e_faster_rcnn_R-50-FPN_2x_ms/' \
                      'e2e_faster_rcnn_R-50-FPN_2x_ms.yaml'


# pps: personpart_seg
__C.MODULES.PPS = AttrDict()

__C.MODULES.PPS.CFG = 'ckpts/semseg/lip_mlhp/resnet26_1x64d_dilated16-ppm_bilinear_deepsup-crop-2x/' \
                      'resnet26_1x64d_dilated16-ppm_bilinear_deepsup-crop-2x.yml'

# fkp: personface_kpts
__C.MODULES.FKP = AttrDict()

__C.MODULES.FKP.CFG = 'ckpts/cls/face_keypoints/ldmk10_84/ldmk10_84.yaml'

# cs: crack_seg
__C.MODULES.CS = AttrDict()

__C.MODULES.CS.CFG = 'ckpts/semseg/crack/resnet18_1x64d_dilated8-ppm_bilinear_deepsup-crop/' \
                     'resnet18_1x64d_dilated8-ppm_bilinear_deepsup-crop.yaml'

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set()

# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'PIXEL_MEAN': 'PIXEL_MEANS',
    'PIXEL_STD': 'PIXEL_STDS',
}


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def merge_priv_cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)
    # update_cfg()


def merge_priv_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_priv_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
            format(full_key, new_key, msg)
    )
