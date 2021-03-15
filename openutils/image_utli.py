'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: Please set LastEditors
LastEditTime: 2021-03-12 21:12:53
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import ndimage


def pil_to_np(pil_img):
    rgb = np.asarray(pil_img, dtype='uint8')
    return rgb


def np_to_pil(np_img):
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(np_img)


def normal_to_uint8(array):
    img = np.asarray(array, dtype='float32')
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype('uint8')
    return img


def split_by_strides_4D(X, kh, kw, s):
    """分割成滑动窗,与卷积中的滑动相同

    Args:
        X ([type]): [description]
        kh ([type]): kernel size in x axis.
        kw ([type]): kernel size in y axis.
        s ([type]): stride of x and y axis.

    Returns:
        [type]: [description]
    """
    N, H, W, C = X.shape
    oh = (H - kh) // s + 1
    ow = (W - kw) // s + 1
    shape = (N, oh, ow, kh, kw, C)
    strides = (X.strides[0], X.strides[1] * s, X.strides[2] * s,
               *X.strides[1:])
    A = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return A


def split_by_strides_2D(X, kh, kw, s):
    """分割成滑动窗,与卷积中的滑动相同

    Args:
        X ([type]): [description]
        kh ([type]): kernel size in x axis.
        kw ([type]): kernel size in y axis.
        s ([type]): stride of x and y axis.

    Returns:
        [type]: [description]
    """
    H, W = X.shape
    oh = (H - kh) // s + 1
    ow = (W - kw) // s + 1
    shape = (oh, ow, kh, kw)
    strides = (X.strides[0] * s, X.strides[1] * s, *X.strides)
    A = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return A


def pltshow(img, *args, **kargs):
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
            plt.imshow(img, *args, **kargs)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                plt.imshow(img, *args, **kargs)
            elif img.shape[0] == 3:
                plt.imshow(img.transpose(1, 2, 0), *args, **kargs)
            else:
                raise ValueError('showing error, array dim is wrong')
        else:
            raise ValueError('showing error, array dim is wrong')
    else:
        plt.imshow(pil_to_np(img), *args, **kargs)


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
