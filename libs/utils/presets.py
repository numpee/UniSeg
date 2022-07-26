"""Preset Transforms for Demos"""
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transform

__all__ = ['load_image', 'subtract_imagenet_mean_batch']

import models.segformer_utils

input_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize([.485, .456, .406], [.229, .224, .225])])

def load_image(filename, size=None, scale=None, keep_asp=True):
    """Load the image for demos"""
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = models.segformer_utils.resize((size, size2), Image.ANTIALIAS)
        else:
            img = models.segformer_utils.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = models.segformer_utils.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)

    img = input_transform(img)
    return img
