#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import cv2
import math
import random
import functools

try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import paddle
from PIL import Image
import logging

logger = logging.getLogger(__name__)
python_ver = sys.version_info


class KineticsReader(object):
    """
    Data reader for kinetics dataset of two format mp4 and pkl.
    1. mp4, the original format of kinetics400
    2. pkl, the mp4 was decoded previously and stored as pkl
    In both case, load the data, and then get the frame data in the form of numpy and label as an integer.
     dataset cfg: format
                  num_classes
                  seg_num
                  short_size
                  target_size
                  num_reader_threads
                  buf_size
                  image_mean
                  image_std
                  batch_size
                  list
    """

    def __init__(self, name, mode, cfg):
        self.cfg = cfg
        self.mode = mode
        self.name = name
        self.format = cfg.MODEL.format
        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        self.seg_num = self.get_config_from_sec('model', 'seg_num')
        self.seglen = self.get_config_from_sec('model', 'seglen')
        self.step = self.get_config_from_sec('model', 'step')

        self.seg_num = self.get_config_from_sec(mode, 'seg_num', self.seg_num)
        self.random_scale = self.get_config_from_sec(mode, 'random_scale')
        self.short_size = self.get_config_from_sec(mode, 'short_size')
        self.target_size = self.get_config_from_sec(mode, 'target_size')
        self.num_reader_threads = self.get_config_from_sec(mode,
                                                           'num_reader_threads')
        self.buf_size = self.get_config_from_sec(mode, 'buf_size')
        self.enable_ce = self.get_config_from_sec(mode, 'enable_ce')
        self.color_jitter = self.get_config_from_sec(mode, 'color_jitter')
        self.color_space_aug = self.get_config_from_sec(mode, 'color_space_aug')
        self.temporal_jitter = self.get_config_from_sec(mode, 'temporal_jitter')
        self.random_shift = self.get_config_from_sec(mode, 'random_shift')

        self.img_mean = np.array(cfg.MODEL.image_mean).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg.MODEL.image_std).reshape(
            [3, 1, 1]).astype(np.float32)
        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']
        if self.enable_ce:
            random.seed(0)
            np.random.seed(0)

    def get_config_from_sec(self, sec, item, default=None):
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)

    def create_reader(self):
        _reader = self._reader_creator(self.filelist, self.mode, seg_num=self.seg_num, seglen=self.seglen,
                                       step = self.step,
                                       short_size=self.short_size, target_size=self.target_size,
                                       img_mean=self.img_mean, img_std=self.img_std,
                                       shuffle=(self.mode == 'train'),
                                       num_threads=self.num_reader_threads,
                                       buf_size=self.buf_size, format=self.format)

        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return _batch_reader

    def _reader_creator(self,
                        pickle_list,
                        mode,
                        seg_num,
                        seglen,
                        step,
                        short_size,
                        target_size,
                        img_mean,
                        img_std,
                        shuffle=False,
                        num_threads=1,
                        buf_size=1024,
                        format='pkl'):
        def decode_mp4(sample, mode, seg_num, seglen, step, short_size, target_size,
                       img_mean, img_std):
            sample = sample[0].split(' ')
            mp4_path = sample[0]
            # when infer, we store vid as label
            label = int(sample[1])
            try:
                imgs = mp4_loader(mp4_path, seg_num, seglen, step, mode)
                if len(imgs) < 1:
                    logger.error('{} frame length {} less than 1.'.format(
                        mp4_path, len(imgs)))
                    return None, None
            except:
                logger.error('Error when loading {}'.format(mp4_path))
                return None, None

            return imgs_transform(imgs, label, mode, seg_num, seglen, step, \
                                  short_size, target_size, img_mean, img_std)

        def decode_pickle(sample, mode, seg_num, seglen, step, short_size,
                          target_size, img_mean, img_std):
            pickle_path = sample[0]
            #print(pickle_path)
            old_length = seg_num * step
            try:
                if python_ver < (3, 0):
                    data_loaded = pickle.load(open(pickle_path, 'rb'))
                else:
                    data_loaded = pickle.load(
                        open(pickle_path, 'rb'), encoding='bytes')

                vid, label, frames = data_loaded
                if len(frames) < 1:
                    logger.error('{} frame length {} less than 1.'.format(
                        pickle_path, len(frames)))
                    return None, None
                if len(frames) < old_length:
                    logger.error('{} frame length {} less than {}.'.format(
                        pickle_path, len(frames), old_length))
                    return None, None
            except:
                logger.info('Error when loading {}'.format(pickle_path))
                return None, None

            if mode == 'train' or mode == 'valid' or mode == 'test':
                ret_label = label
            elif mode == 'infer':
                ret_label = vid

            imgs = video_loader(frames, seg_num, seglen, step, mode)
            return imgs_transform(imgs, ret_label, mode, seg_num, seglen, step, \
                                  short_size, target_size, img_mean, img_std)

        def imgs_transform(imgs, label, mode, seg_num, seglen, step, short_size,
                           target_size, img_mean, img_std):
            if self.random_scale:
                random_size = random.randint(target_size, short_size)
                imgs = group_scale(imgs, random_size)
            else:
                imgs = group_scale(imgs, short_size)

            if mode == 'train':
                if self.name == "TSM":
                    imgs = group_multi_scale_crop(imgs, short_size)
                imgs = group_random_crop(imgs, target_size)
                imgs = group_random_flip(imgs)
                #添加数据增强部分，提升分类精度
                if self.color_jitter:
                    imgs = group_color_jitter(imgs, self.color_space_aug)
                

            else:
                imgs = group_center_crop(imgs, target_size)

            np_imgs = (np.array(imgs[0]).astype('float32').transpose(
                (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
            for i in range(len(imgs) - 1):
                img = (np.array(imgs[i + 1]).astype('float32').transpose(
                    (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
                np_imgs = np.concatenate((np_imgs, img))
            imgs = np_imgs
            imgs -= img_mean
            imgs /= img_std
            # imgs shape: N C H W
            imgs = np.reshape(imgs, (seglen, seg_num, 3, target_size, target_size))
            # imgs shape: N T C H W
            imgs = imgs.transpose(0, 2, 1, 3, 4)
            # imgs shape: N C T H W
            imgs = np.reshape(imgs,
                              (seglen, 3, seg_num, target_size, target_size))

            return imgs, label

        def reader(dir_flag=True):
            if dir_flag:
                dir_path = os.path.dirname(pickle_list) + '/'
            else:
                dir_path = ''

            with open(pickle_list) as flist:
                lines = [line.strip() for line in flist]
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    pickle_path = dir_path + line.strip()
                    #print(pickle_path)
                    yield [pickle_path]

        if format == 'pkl':
            decode_func = decode_pickle
        elif format == 'mp4':
            decode_func = decode_mp4
        else:
            raise "Not implemented format {}".format(format)

        mapper = functools.partial(
            decode_func,
            mode=mode,
            seg_num=seg_num,
            seglen=seglen,
            step=step,
            short_size=short_size,
            target_size=target_size,
            img_mean=img_mean,
            img_std=img_std)

        return paddle.reader.xmap_readers(mapper, reader, num_threads, buf_size)


def group_multi_scale_crop(img_group, target_size, scales=None, \
                           max_distort=1, fix_crop=True, more_fix_crop=True):
    scales = scales if scales is not None else [1, .875, .75, .66]
    input_size = [target_size, target_size]

    im_size = img_group[0].size

    # get random crop offset
    def _sample_crop_size(im_size):
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in scales]
        crop_h = [
            input_size[1] if abs(x - input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            input_size[0] if abs(x - input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_step = (image_w - crop_pair[0]) / 4
            h_step = (image_h - crop_pair[1]) / 4

            ret = list()
            ret.append((0, 0))  # upper left
            if w_step != 0:
                ret.append((4 * w_step, 0))  # upper right
            if h_step != 0:
                ret.append((0, 4 * h_step))  # lower left
            if h_step != 0 and w_step != 0:
                ret.append((4 * w_step, 4 * h_step))  # lower right
            if h_step != 0 or w_step != 0:
                ret.append((2 * w_step, 2 * h_step))  # center

            if more_fix_crop:
                ret.append((0, 2 * h_step))  # center left
                ret.append((4 * w_step, 2 * h_step))  # center right
                ret.append((2 * w_step, 4 * h_step))  # lower center
                ret.append((2 * w_step, 0 * h_step))  # upper center

                ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

            w_offset, h_offset = random.choice(ret)

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
    crop_img_group = [
        img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        for img in img_group
    ]
    ret_img_group = [
        img.resize((input_size[0], input_size[1]), Image.BILINEAR)
        for img in crop_img_group
    ]

    return ret_img_group


def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
        "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images


def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group


def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size), \
            "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop


def group_scale(imgs, target_size):
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

    return resized_imgs

def group_color_jitter(img_group, color_space_aug=False, alphastd=0.1, eigval=None, eigvec=None):
    if eigval is None:
        # note that the data range should be [0, 255]
        eigval = np.array([55.46, 4.794, 1.148])
    if eigvec is None:
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203]])
    if color_space_aug:
        bright_delta = np.random.uniform(-32, 32)
        contrast_alpha = np.random.uniform(0.6, 1.4)
        saturation_alpha = np.random.uniform(0.6, 1.4)
        hue_alpha = random.uniform(-18, 18)
        out = []
        for img in img_group:
            img = brightnetss(img, delta=bright_delta)
            if random.uniform(0, 1) > 0.5:
                img = contrast(img, alpha=contrast_alpha)
                img = saturation(img, alpha=saturation_alpha)
                img = hue(img, alpha=hue_alpha)
            else:
                img = saturation(img, alpha=saturation_alpha)
                img = hue(img, alpha=hue_alpha)
                img = contrast(img, alpha=contrast_alpha)
            out.append(img)
        img_group = out

    alpha = np.random.normal(0, alphastd, size=(3,))
    rgb = np.array(np.dot(eigvec * alpha, eigval)).astype(np.float32)
    bgr = np.expand_dims(np.expand_dims(rgb[::-1], 0), 0)
    return [img + bgr for img in img_group]

def brightnetss(img, delta):
    if random.uniform(0, 1) > 0.5:
        # delta = np.random.uniform(-32, 32)
        delta = np.array(delta).astype(np.float32)
        img = img + delta
        # img_group = [img + delta for img in img_group]
    return img

def contrast(img, alpha):
    if random.uniform(0, 1) > 0.5:
        # alpha = np.random.uniform(0.6,1.4)
        alpha = np.array(alpha).astype(np.float32)
        img = img * alpha
        # img_group = [img * alpha for img in img_group]
    return img

def saturation(img, alpha):
    if random.uniform(0, 1) > 0.5:
        # alpha = np.random.uniform(0.6,1.4)
        alpha = np.array(alpha).astype(np.float32)
        gray = img * np.array([0.299, 0.587, 0.114]).astype(np.float32)
        gray = np.sum(gray, 2, keepdims=True)
        gray *= (1.0 - alpha)
        img = img * alpha
        img = img + gray
    return img

def hue(img, alpha):
    if random.uniform(0, 1) > 0.5:
        # alpha = random.uniform(-18, 18)
        u = np.cos(alpha * np.pi)
        w = np.sin(alpha * np.pi)
        bt = np.array([[1.0, 0.0, 0.0],
                        [0.0, u, -w],
                        [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114],
                            [0.596, -0.274, -0.321],
                            [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621],
                            [1.0, -0.272, -0.647],
                            [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        t = np.array(t).astype(np.float32)
        img = np.dot(img, t)
        # img_group = [np.dot(img, t) for img in img_group]
    return img

def imageloader(buf):
    if isinstance(buf, str):
        img = Image.open(buf)
    else:
        img = Image.open(BytesIO(buf))

    return img.convert('RGB')


def video_loader(frames, nsample, seglen, step, mode):
    videolen = len(frames)
    #print(videolen)
    old_length = nsample * step

    imgs = []
    for j in range(seglen):
        begin = random.randint(0, videolen - old_length)

        for i in range(nsample):
            idx = begin + i * step

            imgbuf = frames[int(idx)]
            img = imageloader(imgbuf)
            imgs.append(img)

    return imgs


def mp4_loader(filepath, nsample, seglen, step, mode):  # 改为模板的样子
    cap = cv2.VideoCapture(filepath)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    old_length = nsample * step
    if old_length > videolen:
        return []
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)
    imgs = []
    for j in range(seglen):
        begin = random.randint(0, videolen - old_length)

        for i in range(nsample):
            idx = begin + i * step

            imgbuf = sampledFrames[int(idx)]
            #img = Image.fromarray(imgbuf, mode='RGB')
            img = Image.fromarray(imgbuf)
            imgs.append(img)
    # 内存爆了，不知是否关事
    del cap 
    del sampledFrames
    return imgs
