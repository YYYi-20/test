'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: Please set LastEditors
LastEditTime: 2020-11-22 23:45:28
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from imageio import imsave
import cv2
import torch
from pathlib import Path
import logging
from scipy import ndimage


def pil_to_np(pil_img):
    rgb = np.asarray(pil_img, dtype='uint8')
    return rgb


def np_to_pil(np_img):
    if np_img.dtype == 'bool':
        np_img = np_img.astype('uint8') * 255
    elif np_img.max() <= 1 and np_img.min() >= 0:
        np_img = (np_img * 255).astype('uint8')
    else:
        np_img = np_img.astype('uint8')
    return Image.fromarray(np_img)


def pltshow(img):
    """[summary]

    Args:
        img ([type]): shape is (h,w,3),(3,h,w) or (h, w)

    Raises:
        ValueError: [description]
        ValueError: [description]
    """
    plt.figure()
    if isinstance(img, torch.Tensor):
        img = img.cpu().data.numpy()
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            plt.imshow(img)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                plt.imshow(img)
            elif img.shape[0] == 3:
                plt.imshow(img.transpose(1, 2, 0))
            else:
                raise ValueError('showing error, array dim is wrong')
        else:
            raise ValueError('showing error, array dim is wrong')
    else:
        plt.imshow(pil_to_np(img))


def preds_to_classmap(pred_w_h, padding=1):
    '''
    ndarray with shape [preds,w,h]
    '''
    value, w_index, h_index = pred_w_h[:, 0], pred_w_h[:, 1], pred_w_h[:, 2]
    w_max, h_max = np.max(w_index), np.max(h_index)
    cls_map = np.zeros((h_max + padding, w_max + padding),
                       dtype='uint8')  # 右侧 下侧 加白边

    cls_map[h_index, w_index] = value
    return cls_map


def img_zoom(array, factor):
    if len(array.shape) == 2:
        array = ndimage.zoom(array, factor, order=0)
        return array
    elif len(array.shape) == 3:
        tmp = []
        for i in range(array.shape[2]):
            tmp.append(ndimage.zoom(array[:, :, i], factor, order=0))
        return np.stack(tmp, axis=2)
    '''
    def fun(array2d):
        row = []
        for i in range(array2d.shape[0]):
            values = array2d[i, :]
            temp = np.concatenate(
                [np.ones((x, y), dtype='uint8') * v for v in values], axis=1)
            row.append(temp)
        return np.concatenate(row, axis=0)

    if len(array.shape) == 2:
        return fun(array)
    elif len(array.shape) == 3:
        tmp = []
        for i in range(array.shape[2]):
            tmp.append(fun(array[:, :, i]))
        return np.concatenate(tmp, axis=0)
    '''


def remove_padding(img, padding_value):
    """[summary]

    Args:
        array ([type]): 2D/3D img array
    Returns:
        [type]: [description]
    """
    def column_mask(array):
        mask = (array == padding_value).all(0)
        return mask

    if len(img.shape) == 2:
        c_mask = column_mask(img)
        r_mask = column_mask(img.T)
        img = img[~r_mask, ~c_mask]
        return img
    elif len(img.shape) == 3:
        img = img.transpose(2, 0, 1)  # channel first
        summed = np.sum(img, axis=0)
        c_mask = column_mask(summed)
        r_mask = column_mask(summed.T)
        img = img[:, :, ~c_mask]
        img = img[:, ~r_mask, :]
        img = img.transpose(1, 2, 0)  # channel last
        return img


def color_transform(value):
    """Convert hex string color to dec tuple color

    Args:
        value (string ot tuple): color

    Returns:
        string or tuple: transformed color
    """
    digit = "0123456789ABCDEF"
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string

    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)


# 十进制的color map
colormap_dec = {
    'ADI': (114, 64, 70),
    'BACK': (225, 225, 225),
    'DEB': (105, 105, 105),
    'LYM': (195, 100, 197),
    'MUC': (245, 222, 179),
    'MUS': (205, 92, 92),
    'NORM': (244, 164, 96),
    'STR': (70, 130, 180),
    'TUM': (60, 179, 113),
    ##
    'ConneTissue': (105, 105, 105),
    'SquamoEpithe': (244, 164, 96),
    'Gland': (245, 222, 179),
    'SmooMus': (205, 92, 92),
    'CanAssoStro': (70, 130, 180),
}

# 字符串color map
colormap_hex = {
    'White': '#FFFFFF',
}
