'''
utils for PIL image and numpy array.
'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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


def show(img):
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            plt.imshow(img)
        elif len(img.shape) == 3:
            if img.shape[0] == 3:
                plt.imshow(img)
            elif img.shape[2] == 3:
                plt.imshow(img.transpose(2, 0, 1))
            else:
                raise ValueError('showing error, array dim is wrong')
        else:
            raise ValueError('showing error, array dim is wrong')
    else:
        plt.imshow(pil_to_np(img))
