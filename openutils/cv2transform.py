'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-11-20 21:50:14
'''

import cv2
import numpy as np
import random
from .image_utli import remove_padding
import torch


class Rotate():
    def __init__(self, flipcode, p):
        self.flipcode = flipcode
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = cv2.flip(img, self.flipcode)
        return img


class Remove_padding():
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, img):
        img = remove_padding(img, self.padding)
        return img


class Resize():
    '''
    size = (x,y) or (h,w)
    '''
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h, w = self.size
        img = cv2.resize(img, (w, h))
        return img


class ToTensor():
    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        tonsor = torch.from_numpy(img)
        return tonsor
