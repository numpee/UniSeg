"""
# Code borrowded from:
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
#
#
# MIT License
#
# Copyright (c) 2017 ZijunDeng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""

"""
Joint Transform
"""

import math
import numbers

import random
from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

# class WindowCrop(object):
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#
#     def __call__(self, img, mask):
#         assert img.size == mask.size
#         w, h = img.size
#         th, tw = self.size
#         num_horizontals = (w // tw) + 1
#         overlap_w = num_horizontals * tw - w
#         overlap_w_per_crop = overlap_w // (num_horizontals - 1)
#
#         num_verticals = (h // th) + 1
#         overlap_h = num_verticals * th - h
#         overlap_h_per_crop = overlap_h // (num_verticals - 1)
#
#         start_x, start_y = 0, 0
#         for
#
#         x1 = int(round((w - tw) / 2.))
#         y1 = int(round((h - th) / 2.))
#         return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomCrop(object):
    """
    Take a random crop from the image.

    First the image or crop size may need to be adjusted if the incoming image
    is too small...

    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image

    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """

    def __init__(self, size, ignore_index=0, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size
        w, h = img.size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            return img, mask

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border, fill=self.ignore_index)
                w, h = img.size

        if w == tw:
            x1 = 0
        else:
            x1 = random.randint(0, w - tw)
        if h == th:
            y1 = 0
        else:
            y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class ResizeHeight(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.target_h = size
        self.interpolation = interpolation

    def __call__(self, img, mask):
        w, h = img.size
        target_w = int(w / h * self.target_h)
        return (img.resize((target_w, self.target_h), self.interpolation),
                mask.resize((target_w, self.target_h), Image.NEAREST))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCropPad(object):
    def __init__(self, size, ignore_index=-1):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index

    def set_size(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):

        assert img.size == mask.size
        w, h = img.size
        if isinstance(self.size, tuple):
            tw, th = self.size[0], self.size[1]
        else:
            th, tw = self.size, self.size

        if w < tw:
            pad_x = tw - w
        else:
            pad_x = 0
        if h < th:
            pad_y = th - h
        else:
            pad_y = 0

        if pad_x or pad_y:
            # left, top, right, bottom
            img = ImageOps.expand(img, border=(pad_x, pad_y, pad_x, pad_y), fill=0)
            mask = ImageOps.expand(mask, border=(pad_x, pad_y, pad_x, pad_y),
                                   fill=self.ignore_index)

        x1 = int(abs(round((w - tw) / 2.)))
        y1 = int(abs(round((h - th) / 2.)))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class PadImage(object):
    def __init__(self, size, ignore_index):
        self.size = size
        self.ignore_index = ignore_index

    def __call__(self, img, mask):
        assert img.size == mask.size
        th, tw = self.size, self.size

        w, h = img.size

        if w > tw or h > th:
            wpercent = (tw / float(w))
            target_h = int((float(img.size[1]) * float(wpercent)))
            img, mask = img.resize((tw, target_h), Image.BILINEAR), mask.resize((tw, target_h), Image.NEAREST)

        w, h = img.size
        ##Pad
        img = ImageOps.expand(img, border=(0, 0, tw - w, th - h), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, tw - w, th - h), fill=self.ignore_index)

        return img, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(
                Image.FLIP_LEFT_RIGHT)
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    """
    Scale image such that longer side is == size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILIENAR), mask.resize(
                (ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize(
                (ow, oh), Image.NEAREST)


class ScaleMin(object):
    """
    Scale image such that shorter side is == size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize(
                (ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize(
                (ow, oh), Image.NEAREST)


class Resize(object):
    """
    Resize image to exact size of crop
    """

    def __init__(self, size):
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w == h and w == self.size):
            return img, mask
        return (img.resize(self.size, Image.BILINEAR),
                mask.resize(self.size, Image.NEAREST))


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), \
                       mask.resize((self.size, self.size), Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(
            rotate_degree, Image.NEAREST)


class RandomSizeAndCrop(object):
    def __init__(self, size, scale_min=0.5, scale_max=2.0, ignore_index=-1, pre_size=None):
        self.size = size
        self.crop = RandomCrop(self.size, ignore_index=ignore_index)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.pre_size = pre_size

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size

        # first, resize such that shorter edge is pre_size
        if self.pre_size is None:
            scale_amt = 1.
        elif img.size[1] < img.size[0]:
            scale_amt = self.pre_size / img.size[1]
        else:
            scale_amt = self.pre_size / img.size[0]
        scale_amt *= random.uniform(self.scale_min, self.scale_max)
        w, h = [int(i * scale_amt) for i in img.size]

        if centroid is not None:
            centroid = [int(c * scale_amt) for c in centroid]

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(img, mask, centroid)


class Rotate(object):
    def __init__(self, rotate_min=-10, rotate_max=10):
        self.rotate_min = rotate_min
        self.rotate_max = rotate_max

    def __call__(self, img, mask):
        deg = random.uniform(self.rotate_min, self.rotate_max)
        return img.rotate(deg, resample=Image.BILINEAR), mask.rotate(deg, resample=Image.NEAREST)
