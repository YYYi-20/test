'''
utils for PIL image and numpy array.
'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from imageio import imread, imsave
import cv2


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


def preds_to_classmap(pred_w_h):
    '''
    ndarray with shape [preds,w,h]
    '''
    value, w_index, h_index = pred_w_h[:, 0], pred_w_h[:, 1], pred_w_h[:, 2]
    w_max, h_max = np.max(w_index), np.max(h_index)
    cls_map = np.zeros((h_max + 1, w_max + 1), dtype='uint8')  #右侧 下侧 加白边
    cls_map[h_index, w_index] = value
    return cls_map


def map_zoom(array, x=50, y=50):
    row = []
    for i in range(array.shape[0]):
        values = array[i, :]
        temp = np.concatenate(
            [np.ones((x, y), dtype='uint8') * v for v in values], axis=1)
        row.append(temp)
    return np.concatenate(row, axis=0)


def classmap_to_img(
    cls_map,
    labels,
    names,
    colors,
    bar_size=3,
    save_path=None,
    resolution=20,
    split=True,
    show=False,
    font_szie=1.5,
    font_thick=4,
):
    """plot or save the class_map into ima files.

    Args:
        cls_map ([type]): 2d-array
        labels ([type]): list with int elemetns that represent predicted labels of the model in the 2-d array.
        names ([type]): corresponding names of the labels.
        colors ([type]): corresponding colors of the labels. e.g. colors = [(70, 130, 180), (0, 0, 0), (114, 64, 70),
        (195, 100, 197), (252, 108, 133), (205, 92, 92), (255, 163, 67)]
        save_path ([type], optional): [description]. Defaults to None.
        resolution (int, optional): [description]. Defaults to 50. if the class map is too small, use this to expand the size of the map.
        split (bool, optional): [description]. Defaults to True. if True, we save each class into one iamges additionally.
        show (bool, optional): [description]. Defaults to False. if True, preview the img.
        font_szie (float, optional): font_szie of bar
    """
    assert len(labels) == len(names) and len(labels) == len(colors)
    h, w = cls_map.shape
    if not bar_size == 0:
        assert h > len(names)
        padding = map_zoom(
            np.asarray(labels).reshape(-1, 1), bar_size, bar_size)
        H = (np.linspace(0, padding.shape[0], len(names), endpoint=False) +
             bar_size) * resolution
        W = np.repeat(padding.shape[1], len(names)) * resolution
        positions = list(zip(W.astype('int64'), H.astype('int64')))
        padding_bottom = np.zeros(shape=(h - padding.shape[0],
                                         padding.shape[1]))
        padding = np.r_[padding, padding_bottom]
        padding = np.c_[padding, np.zeros(shape=(h, 9))]
        padding = map_zoom(padding, resolution, resolution)

        padding_image = np.ones(
            (padding.shape[0], padding.shape[1], 3), dtype='uint8') * 255
        for i, label_ in enumerate(labels):
            X, Y = np.where(padding == label_)
            padding_image[X, Y] = colors[i]

        for i in range(len(names)):
            cv2.putText(padding_image, names[i], positions[i],
                        cv2.FONT_HERSHEY_SIMPLEX, font_szie, (0, 0, 0),
                        font_thick, cv2.LINE_AA)
            #图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细

    cls_map = map_zoom(cls_map, resolution, resolution)
    h, w = cls_map.shape
    all_image = np.ones((h, w, 3), dtype='uint8') * 255
    split_images = [np.ones((h, w, 3), dtype='uint8') * 255 for i in names]
    for i, label_ in enumerate(labels):
        X, Y = np.where(cls_map == label_)
        all_image[X, Y] = colors[i]
        split_images[i][X, Y] = colors[i]

    if not bar_size == 0:
        all_image = np.concatenate([all_image, padding_image], axis=1)
        for i in range(len(split_images)):
            split_images[i] = np.concatenate([split_images[i], padding_image],
                                             axis=1)

    if show:
        pltshow(all_image)
    if save_path is not None:
        if split:
            for i in range(len(split_images)):
                imsave(os.path.join(save_path, f'{names[i]}.jpeg'),
                       split_images[i])
        imsave(os.path.join(save_path, 'ALL.jpeg'), all_image)