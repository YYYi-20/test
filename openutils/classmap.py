'''
Descripttion: python project
version: 0.1
Author: Yuni
LastEditors: XRZHANG
LastEditTime: 2020-11-11 11:48:36
'''

import os

import numpy as np
from scipy import stats
from skimage import color

from openutils import *
from openutils import classmap_to_img
from openutils import colormap_dec as colormap
from openutils import preds_to_classmap


class ClassmapStatistic(object):
    def __init__(self, cls_map, labels, names, colors=None):
        """[summary]

        Args:
            cls_map (ndarray): 2d-array.
            labels (list): the value in the classmap which we want to plot. For
                            the value not in labels list, we will plot white. !!!!!Donot use 0
                            in labels except background, as we defaulty set tiles with value 0 to white color.
            names (list): corresponding names of the labels.
            colors (list): corresponding colors of the labels. e.g. colors = [(70, 130, 180), (0, 0, 0), (114, 64, 70),
            (195, 100, 197), (252, 108, 133), (205, 92, 92), (255, 163, 67)]
        """
        self.cls_map = cls_map
        self.labels = labels
        self.names = names
        self.colors = colors
        if self.colors is None:
            self.colors = [colormap[i] for i in self.names]
        self.h = self.cls_map.shape[0]
        self.w = self.cls_map.shape[1]
        assert len(self.labels) == len(self.names) and len(self.labels) == len(
            self.colors)
        # self.seed = np.random.seed(0)

    def save_img(self,
                 save_path=None,
                 split=True,
                 bar_size=3,
                 resolution=20,
                 font_szie=1.5,
                 font_thick=3,
                 show=False):
        """plot or save the class_map into image files name.jpeg.

        Args:
            save_path (str, optional): The dir (it's dirname not filename) to save images. Defaults to None.
            resolution (int, optional): Defaults to 50. if the class map is too small, use this to expand the size of the map.
            split (bool, optional): [description]. Defaults to True. if True, we save each class into one iamges additionally.
            show (bool, optional): [description]. Defaults to False. if True, preview the img.
            font_szie (float, optional): font_szie of bar
        """
        # if not bar_size == 0:
        #     # generate bar
        #     try:
        #         assert self.h > len(self.names)
        #         padding = map_zoom(
        #             np.asarray(self.labels).reshape(-1, 1), bar_size, bar_size)
        #         H = (np.linspace(
        #             0, padding.shape[0], len(self.names), endpoint=False) +
        #              bar_size) * resolution
        #         W = np.repeat(padding.shape[1], len(self.names)) * resolution
        #         positions = list(zip(W.astype('int64'), H.astype('int64')))
        #         padding_bottom = np.zeros(shape=(self.h - padding.shape[0],
        #                                          padding.shape[1]))
        #         padding = np.r_[padding, padding_bottom]
        #         padding = np.c_[padding, np.zeros(shape=(self.h, 9))]
        #         padding = map_zoom(padding, resolution, resolution)

        #         padding_image = np.ones(
        #             (padding.shape[0], padding.shape[1], 3),
        #             dtype='uint8') * 255
        #         for i, label_ in enumerate(self.labels):
        #             X, Y = np.where(padding == label_)
        #             padding_image[X, Y] = self.colors[i]

        #         for i in range(len(self.names)):
        #             cv2.putText(padding_image, self.names[i], positions[i],
        #                         cv2.FONT_HERSHEY_SIMPLEX, font_szie, (0, 0, 0),
        #                         font_thick, cv2.LINE_AA)
        #             #图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        #     except:
        #         bar_size = 0
        #         logging.info('pleaser check the bar plot code')
        # if rasie error in try, we set bar_size=0

        all_color = color.label2rgb(self.cls_map,
                                    colors=self.colors,
                                    bg_label=0,
                                    bg_color=(255, 255, 255)).astype('uint8')
        if split:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                for i, label_ in enumerate(self.labels):
                    X, Y = np.where(self.cls_map != label_)
                    one_color = all_color.copy()
                    one_color[X, Y] = [255, 255, 255]
                    one_color = img_zoom(one_color, resolution)
                    imsave(Path(save_path, f'{self.names[i]}.jpeg'), one_color)

        all_color = img_zoom(all_color, resolution)
        # if bar_size is not 0:
        #     all_image = np.concatenate([all_image, padding_image], axis=1)
        #     if split:
        #         for i in range(len(split_images)):
        #             split_images[i] = np.concatenate(
        #                 [split_images[i], padding_image], axis=1)
        if show:
            pltshow(all_color)
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            imsave(Path(save_path, '0_ALL.jpeg'), all_color)

    def distance_to_tomor(self):
        pass

    def proportion(self,
                   tumor_label=7,
                   interest_label=range(1, 8),
                   interaction_label=[2, 4],
                   submap_size=(10, 10),
                   sample_fraction=1):
        """以一个肿瘤点为中心，选择submap,计算interset_label 中每个label  占所有 interest label的比例
            公式(number of each one interset_label) / (number of all interest_label); 背景的label为0.
            返回结果可以再输入score函数，计算90%分位数
            

        Args:
            tumor_label (int, optional): [description]. Defaults to 7.
            interest_label ([type], optional): [description]. Defaults to range(1, 8).
            interaction_label (list, optional): [description]. Defaults to [2, 4].
            submap_size (tuple, optional): [description]. Defaults to (10, 10).
            sample_fraction (int, optional): 采样肿瘤tiles. Defaults to 1.

        Returns:
            [type]: [description]
        """
        x_index, y_index = np.where(self.cls_map == tumor_label)
        idx = np.random.choice(range(len(y_index)),
                               size=int(sample_fraction * len(y_index)),
                               replace=False)
        x_index, y_index = x_index[idx], y_index[idx]

        interest = []
        interaction = []
        for i, j in zip(x_index, y_index):
            submap = self._submap((i, j), submap_size)
            if submap is None:
                continue
            # else, submap is not None
            non_back_num = submap.size - self._count(0, submap)  #非背景 tiles 个数

            interest_counts = self._count(interest_label, submap)  # list
            sum_interest_counts = sum(interest_counts)
            tumor_counts = self._count(tumor_label, submap)
            if not (tumor_counts / non_back_num >= 0.3
                    and tumor_counts / non_back_num < 0.95):
                continue
            # else:
            interest_ratio = interest_counts / sum_interest_counts
            interest.append(interest_ratio)

            interaction_counts = np.zeros(len(interaction_label))
            M, N = np.where(submap == tumor_label)
            for m, n in zip(M, N):
                if m - 1 >= 0 and m + 1 < submap_size[
                        0] + 1 and n - 1 >= 0 and n + 1 < submap_size[
                            1] + 1 and (m % 3 == 0):
                    submap2 = submap[m - 1:m + 2, n - 1:n + 2]
                    tmp = self._count(interaction_label, submap2)
                    interaction_counts += tmp

            interaction_ratio = interaction_counts / sum_interest_counts
            interaction.append(interaction_ratio)

        interaction = np.asarray(interaction)
        interest = np.asarray(interest)
        return interest, interaction

    def _count(self, labels, array):
        array = array.flatten().tolist()
        if isinstance(labels, int):
            return array.count(labels)
        elif len(labels) == 1:
            return array.count(labels[0])
        else:
            counts = []
            for i in labels:
                tmp = array.count(i)
                counts.append(tmp)
            return np.asarray(counts)

    def _submap(self, center, sub_size):
        top, bottom, = center[0] - int(sub_size[0] / 2), center[0] + int(
            (sub_size[0] + 1) / 2),
        left, right = center[1] - int(sub_size[1] / 2), center[1] + int(
            (sub_size[1] + 1) / 2)
        if top > 0 and bottom < self.h and left > 0 and right < self.w:
            return self.cls_map[top:bottom, left:right]
        else:
            return None

    def score(self, array, percent=90):
        score = np.percentile(array, percent, axis=0)
        return score

    def entropy(self, array):
        #计算概率分布==========计算熵=========
        probs = array / array.sum()
        s = stats.entropy(probs, base=2)
        return s


if __name__ == '__main__':
    preds = np.load(
        '/data/backup2/xianrui/data/ptmc/images/pred_maps/18 28528 5/preds.npy'
    )
    cls_map = preds_to_classmap(preds)
    names = [
        'ConneTissue', 'SquamoEpithe', 'Gland', 'LYM', 'SmooMus',
        'CanAssoStro', 'TUM'
    ]

    labels = list(range(1, 8))
    statis = ClassmapStatistic(cls_map, labels, names)
    a, b = statis.proportion(tumor_label=7,
                             interest_label=range(1, 8),
                             interaction_label=[2, 4],
                             submap_size=(10, 10),
                             sample_fraction=0.5)
    statis.save_img(save_path=None, show=True)
