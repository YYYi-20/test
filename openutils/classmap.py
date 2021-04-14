'''
Descripttion: python project
version: 0.1
Author: Yuni
LastEditors: ZHANG XIANRUI
LastEditTime: 2021-04-14 15:17:20
'''
"""
凡是自己写的函数默认数组维度为 `纵轴x横轴`，注意与一些opencv库函数区分
"""

import cv2
import logging
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from skimage import color, morphology
from imageio import imsave

from .utils import *
from .image_utli import *


class FeatureExtractor():
    def __init__(self):
        pass

    def entropy(self, p, base=2):
        # ==========计算熵=========
        if p.sum() != 1:
            p = p / p.sum()
        self.s = stats.entropy(p, base=base)
        return self.s

    def gmm(self, X, **kwargs):
        """Gaussian Mixture.
        Representation of a Gaussian mixture model probability distribution.
        This class allows to estimate the parameters of a Gaussian mixture
        distribution.
        Parameters
        ----------
        X: input data. Each row is a sample. Must be 2-D array.
        
        n_components : int, defaults to 1.
            The number of mixture components.

        covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}
            String describing the type of covariance parameters to use.
            Must be one of:

            'full'
                each component has its own general covariance matrix
            'tied'
                all components share the same general covariance matrix
            'diag'
                each component has its own diagonal covariance matrix
            'spherical'
                each component has its own single variance

        tol : float, defaults to 1e-3.
            The convergence threshold. EM iterations will stop when the
            lower bound average gain is below this threshold.

        reg_covar : float, defaults to 1e-6.
            Non-negative regularization added to the diagonal of covariance.
            Allows to assure that the covariance matrices are all positive.

        max_iter : int, defaults to 100.
            The number of EM iterations to perform.

        n_init : int, defaults to 1.
            The number of initializations to perform. The best results are kept.

        init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
            The method used to initialize the weights, the means and the
            precisions.
            Must be one of::

                'kmeans' : responsibilities are initialized using kmeans.
                'random' : responsibilities are initialized randomly.

        weights_init : array-like, shape (n_components, ), optional
            The user-provided initial weights, defaults to None.
            If it None, weights are initialized using the `init_params` method.

        means_init : array-like, shape (n_components, n_features), optional
            The user-provided initial means, defaults to None,
            If it None, means are initialized using the `init_params` method.

        precisions_init : array-like, optional.
            The user-provided initial precisions (inverse of the covariance
            matrices), defaults to None.
            If it None, precisions are initialized using the 'init_params' method.
            The shape depends on 'covariance_type'::

                (n_components,)                        if 'spherical',
                (n_features, n_features)               if 'tied',
                (n_components, n_features)             if 'diag',
                (n_components, n_features, n_features) if 'full'

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        warm_start : bool, default to False.
            If 'warm_start' is True, the solution of the last fitting is used as
            initialization for the next call of fit(). This can speed up
            convergence when fit is called several times on similar problems.
            In that case, 'n_init' is ignored and only a single initialization
            occurs upon the first call.
            See :term:`the Glossary <warm_start>`.

        verbose : int, default to 0.
            Enable verbose output. If 1 then it prints the current
            initialization and each iteration step. If greater than 1 then
            it prints also the log probability and the time needed
            for each step.

        verbose_interval : int, default to 10.
            Number of iteration done before the next print.
        """
        from sklearn.mixture import GaussianMixture
        self.gmm_model = GaussianMixture(**kwargs).fit(X)
        labels = self.gmm_model.predict(X)
        return labels, self.gmm_model.means_, self.gmm_model.covariances_, self.gmm_model.weights_

    def percentile_score(self, array, percent=90):
        return np.percentile(array, percent, axis=0)

    def distance_classification(self, array, thres=-20):
        """calculate the number of far, inside, around pixels in the array according to thres.
        < thres, far;  > 0, inside pixels;  [thres,0] around.

        Args:
            array ([type]): [description]
            thres (int, optional): [description]. Defaults to -20.

        Returns:
            [type]: inside, around, far.
        """
        far = np.sum(array < thres)
        inside = np.sum(array > 0)
        around = len(array.flatten()) - far - inside
        return inside, around, far


def _count(labels, array, fraction=False):
    """分别计算每个label在整个array中的数量,label长度可以为1，也可以是int,float
    if not exist, return counts=0

    Args:
        labels (int, float): list or ndarray with only one element is also properly.
        array (ndarray): [description]
        ratio: return counts or ratios to the total number.

    Returns:
        ndarray, int: the length is same as labels.
    """
    array = array.flatten().tolist()
    if not fraction:
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
    else:
        total_num = len(array)
        if isinstance(labels, int):
            return array.count(labels) / total_num
        elif len(labels) == 1:
            return array.count(labels[0]) / total_num
        else:
            counts = []
            for i in labels:
                tmp = array.count(i)
                counts.append(tmp)
            return np.asarray(counts) / total_num


def ratio(top, down, array):
    """top = [1,2,3], down = [1,3,5],计算数组中 1,2,3 占sum[1,3,5]的比例
    分别计算top中每个label的数量，与占down的总和相除

    Args:
        top ([type]): [description]
        down ([type]): [description]
        array ([type]): [description]

    Returns:
        [type]: [description]
    """
    a = _count(top, array)
    b = _count(down, array).sum()
    return list(a / b)


class ClassmapStatistic(object):
    def __init__(self, cls_map, tumor_label, show=False, seed=0):
        """[summary]

        Args:
            cls_map (ndarray): 2d-array.
            labels (list): the value in the classmap which we want to plot. For
                            the value not in labels list, we will use white color to plot. 
                            !!!!!we defaulty set tiles with value 0 to white color.
            names (list): corresponding names of the labels.
            colors (list): corresponding colors of the labels. e.g. colors = [(70, 130, 180), (0, 0, 0), (114, 64, 70),
            (195, 100, 197), (252, 108, 133), (205, 92, 92), (255, 163, 67)]
        """
        self.cls_map = cls_map

        self.h = self.cls_map.shape[0]
        self.w = self.cls_map.shape[1]
        self.seed = np.random.seed(seed)
        self.show = show
        self.tumor_label = tumor_label
        self.tumor_exist = np.any(self.cls_map == self.tumor_label)
        if not self.tumor_exist:
            logging.warning('no tumor in this slide.')

    def save_img(self, colors=None, save_path=None, resolution=20):
        """plot or save the class_map into image files name.jpeg.
        Args:
            colors ([type], optional): [description]. Defaults to None.
            save_path (str, optional): The  filename used to save images. Defaults to None.
            resolution (int, optional): Defaults to 50. if the class map is too small, use this to expand the size of the map.
        """
        all_color = color.label2rgb(self.cls_map,
                                    colors=colors,
                                    bg_label=0,
                                    bg_color=(255, 255, 255)).astype('uint8')
        all_color = img_zoom(all_color, resolution)
        if self.show:
            pltshow(all_color)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            imsave(save_path, all_color)
            logging.info(f'save in {save_path}')

    def roi_count(self,
                  interest_class,
                  kernel_size=10,
                  stride=10,
                  threshold_non_back=(0.5, 1),
                  threshold_other={
                      1: (0.3, 0.95),
                      2: (0.3, 0.95)
                  }):
        """ 滑动窗，选择submap,计算interset_label 中每个label占所有
        interest label的比例一阶公式为(number of each first_label) / (number of all first_label); 背景的label为0.  返回list结果可以再输入score函数，计算90%分位数。                                                                   
        Args:
            interest_class (list): The label that we interested. Defaults to range(1, 8).
            kernel_size (tuple or int, optional): The size of sub_classmap in 1st order. Defaults to (10, 10).
            threshold: 用于选择submap是否合格，每个class对应的比例阈值.

        Returns:
            tuple: [description]
        """
        if not self.tumor_exist:
            return
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size[0], kernel_size[1]
        elif isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        splited_maps = split_by_strides_2D(self.cls_map, kh, kw,
                                           stride).reshape(-1, kh, kw)
        if len(splited_maps) == 0:
            logging.warning('size of ROI is too large!')
            return
        rois = []

        def fun_count(interest_class, submap):
            total_num, back_num = submap.size, _count(0, submap)
            non_back_num = total_num - back_num
            # the number of each interest label in submap
            return total_num, non_back_num, _count(interest_class, submap)

        rois = []
        for submap in splited_maps:
            # 非背景 tiles 个数
            total_num, non_back_num, interest_num = fun_count(
                interest_class, submap)
            #判断阈值
            #先保证除数non_back_num不为0
            if non_back_num / total_num in interval(*threshold_non_back):
                if np.all([
                        _count(label, submap) / non_back_num
                        in interval(*thres)
                        for label, thres in threshold_other.items()
                ]):
                    rois.append(
                        np.hstack([total_num, non_back_num, interest_num]))
        if len(rois) != 0:
            rois = np.vstack(rois)
            #在whole slide level 计算
            #排在第一行
            whole_count = np.hstack(fun_count(interest_class, self.cls_map))

            index = ['whole_slide'] + [f'roi_{i}' for i in (range(len(rois)))]
            columns = ['total', 'non_back'] + list(interest_class)
            result = pd.DataFrame(data=np.vstack([whole_count, rois]),
                                  columns=columns,
                                  index=index)
            return result
        else:
            logging.warning(
                'ROI threshold is too small/large. No proper ROI is selected, so we add nan'
            )
            #没有合适的ROI,只返回whole
            whole_count = np.hstack(fun_count(interest_class, self.cls_map))
            index = ['whole_slide', 'roi_1']
            columns = ['total', 'non_back'] + list(interest_class)
            rois = np.full([1, len(columns)], np.nan)
            result = pd.DataFrame(data=np.vstack([whole_count, rois]),
                                  columns=columns,
                                  index=index)
            return result

    def tumor_mask_preprocess(self, tumor_label, disk, small_object):
        '''腐蚀膨胀，选取合适参数
        '''
        self.tumor_mask = np.where(self.cls_map == tumor_label, 255,
                                   0).astype('uint8')
        if self.show:
            logging.info('before process')
            pltshow(self.tumor_mask)
        kernel = morphology.disk(disk)
        tumor_mask = cv2.morphologyEx(self.tumor_mask, cv2.MORPH_CLOSE, kernel)
        # tumor_mask =morphology.closing(tumor_mask, selem=None, out=None)
        # tumor_mask=morphology.remove_small_holes(tumor_mask, 1000)
        tumor_mask = ndimage.binary_fill_holes(tumor_mask)
        tumor_mask = morphology.remove_small_objects(tumor_mask, small_object)
        tumor_mask_ = tumor_mask.astype(np.uint8) * 255
        if self.show:
            logging.info('after process')
            pltshow(tumor_mask_)
        return tumor_mask_

    def calc_distance(self, tumor_mask, interest_class, new_cls_map=None):
        """寻找肿瘤区域，并计算感兴趣的label到肿瘤区域的距离分布.

        Args:
            tumor_mask ([type]): [description]
            interest_class ([type]): [description]
            new_cls_map ([type], optional): [description]. Defaults to None.

        Returns:
            tuple: far, around, inside.
        """
        if not self.tumor_exist:
            return
        if new_cls_map is None:
            new_cls_map = self.cls_map
        contours, _ = cv2.findContours(tumor_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            logging.warning('no tumor contours were found.')
            return
        else:
            all_distance = pd.DataFrame()
            for cls in interest_class:
                idx_x, idx_y = np.where(new_cls_map == cls)
                if len(idx_x) != 0:
                    distance_i = []  # 一维列表，存储一个cls下的所有tile的距离
                    for i, j in zip(idx_x, idx_y):
                        #选择最近的距离添加到distance_i
                        nearst = max([
                            cv2.pointPolygonTest(cnt, (i, j), True)
                            for cnt in contours
                        ])
                        distance_i.append(nearst)
                else:  #如果找不到  添加 nan
                    logging.info(
                        f'class {cls} is not found in ROI,so we use nan')
                    distance_i = [None]

                # 把所有cls的距离按照每一列拼接起来
                all_distance = pd.concat(
                    [all_distance, pd.DataFrame(distance_i)], axis=1)
            all_distance.columns = interest_class
            # distance = [[] for _ in range(len(first_label))]
            # for interest_label in first_label
            # for i in range(self.h):
            #     for j in range(self.w):
            #         c = self.cls_map[i, j]
            #         if c in first_label:
            #             tmp = max([
            #                 cv2.pointPolygonTest(cnt, (i, j), True)
            #                 for cnt in contours
            #             ])
            #             # 遍历所有轮廓，temp记录最近的轮廓。当点在轮廓外时返回负值，当点在内部时返回正值,如果点在轮廓上则返回零.
            #             idx = first_label.index(c)
            #             distance[idx].append(tmp)
            # for i, dis in enumerate(distance):
            #     tmp = count_by_thres(np.asarray(dis), thres)
            #     distance[i] = tmp

            # num_tumor = _count(tumor_label, self.cls_map)
            # distance = np.asarray(distance)
            # distance_ratio = distance / num_tumor
        # return distance, distance_ratio
        return all_distance


'''
    def proportion(self,
                   tumor_label=7,
                   first_label=[1, 2],
                   first_kernel_size=(10, 10),
                   first_stride=2,
                   second_label=[1, 2],
                   second_kernel_size=(3, 3),
                   second_stride=2,
                   threshold_non_back=(0.5, 1),
                   threshold_other={
                       1: (0.3, 0.95),
                       2: (0.3, 0.95)
                   }):
        """ 滑动窗，选择submap,计算interset_label 中每个label占所有
        interest label的比例一阶公式为(number of each first_label) / (number of all first_label); 背景的label为0.  返回list结果可以再输入score函数，计算90%分位数。
        Args:
            tumor_label (int, optional): [description]. Defaults to 7.
            first_label ([type], optional): The label that we interested. Defaults to range(1, 8).
            first_kernel_size (tuple, optional): The size of sub_classmap in 1st order. Defaults to (10, 10).
            second_label (list, optional): the label we want to observed in sub_sub_classmap. Defaults to [2, 4].
            second_kernel_size (tuple, optional): The size of sub_classmap in 1st order. Defaults to (10, 10).
            threshold: 用于选择submap是否合格，每个class对应的比例阈值.

        Returns:
            tuple: [description]
        """
        if not self.tumor_exist:
            first, second = None, None
        splited_maps = split_by_strides_2D(self.cls_map, *first_kernel_size,
                                           first_stride).reshape(
                                               -1, *first_kernel_size)
        first = []
        second = []
        for submap in splited_maps:
            # 非背景 tiles 个数
            total_num, back_num = submap.size, _count(0, submap)
            non_back_num = total_num - back_num
            # the number of each interest label in submap
            #判断阈值
            flag_non_back = non_back_num / total_num in interval(
                *threshold_non_back)

            #保证除数non_back_num不为0
            if flag_non_back and np.all([
                    _count(label, submap) / non_back_num in interval(*thres)
                    for label, thres in threshold_other.items()
            ]):
                # if (non_back_num !=
                # 0) and (tumor_counts / non_back_num >= threshold[0]) and (
                #     tumor_counts / non_back_num <= threshold[1]):
                # make sure the proporation of tumor  in submap is in threshold
                # if tumor counts in threshold, add result to list
                # add (counts,proporation) to list
                first_counts = _count(first_label, submap)
                first.append(first_counts)
                second_splited_maps = split_by_strides_2D(
                    submap, *second_kernel_size,
                    second_stride).reshape(-1, *second_kernel_size)
                # 在一个submap中累加
                second_counts = np.zeros(len(second_label))  # 初始为0
                for s_submap in second_splited_maps:
                    second_counts += _count(second_label, s_submap)
                second.append(
                    (second_counts, second_counts / sum(first_counts)))
        first, second = np.asarray(first), np.asarray(second)
        return first, second


    def _proportion(self,
                    tumor_label=7,
                    first_label=range(1, 8),
                    first_kernel_size=(10, 10),
                    second_label=[2, 4],
                    second_kernel_size=(3, 3),
                    sample_fraction=1,
                    threshold=(0.3, 0.95)):
        """ 每一个肿瘤点为中心，选择submap,计算interset_label 中每个label占所有
        interest label的比例一阶公式为(number of each first_label) / (number of all first_label); 背景的label为0.  返回list结果可以再输入score函数，计算90%分位数。                                                                   
        Args:
            tumor_label (int, optional): [description]. Defaults to 7.
            first_label ([type], optional): The label that we interested used for 1st order. Defaults to range(1, 8).
            first_kernel_size (tuple, optional): The size of sub_classmap in 1st order. Defaults to (10, 10).
            second_label (list, optional): the label we want to observed in sub_sub_classmap. Defaults to [2, 4].
            second_kernel_size (tuple, optional): The size of sub_classmap in 1st order. Defaults to (10, 10).
            sample_fraction (int, optional): n不需要以每个肿瘤tile为中心计算，可以随机选取一定比例的tumor tiles. Defaults to 1.
            threshold: [].

        Returns:
            tuple: [description]
        """
        x_index, y_index = np.where(self.cls_map == tumor_label)
        idx = np.random.choice(range(len(y_index)),
                               size=int(sample_fraction * len(y_index)),
                               replace=False)
        # 随机选取的肿瘤tile的index: x and y
        x_index, y_index = x_index[idx], y_index[idx]

        first = []
        second = []
        for x, y in zip(x_index, y_index):
            # 以tumor tile 的 x,y为中心，选取sub_map
            submap = get_submap((x, y), first_kernel_size, self.cls_map)
            if submap is None:  # size不符合要求  太小就略过
                continue
            # if submap is not None, run below
            # 非背景 tiles 个数
            non_back_num = submap.size - _count(0, submap)
            # the number of each interest label in submap
            interest_counts = _count(first_label, submap)
            sum_interest_counts = sum(interest_counts)
            # the number of tumor tiles in submap
            tumor_counts = _count(tumor_label, submap)

            #当该sub_mp的肿瘤tile个数满足一定比例在进行后续计算
            if non_back_num == 0:
                continue
            if not (tumor_counts / non_back_num >= threshold[0]
                    and tumor_counts / non_back_num < threshold[1]):
                continue
            # make sure the proporation of tumor  in submap is in threshold
            # if tumor counts in threshold, add result to list
            # add (counts,proporation) to list
            first.append(
                (interest_counts, interest_counts / sum_interest_counts))

            second_counts = np.zeros(len(second_label))  # 初始为0
            tumor_x, tumor_y = np.where(submap == tumor_label)
            for x_, y_ in zip(tumor_x, tumor_y):
                s_submap = get_submap((x_, y_), second_kernel_size, submap)
                if s_submap is not None:
                    # ??需要取%,,保证遍历整个submap但是不要有overlap  #  最好保证恰好分成等大小的方格?????
                    if x_ % second_kernel_size[0] == 0:
                        second_counts += _count(second_label, s_submap)
            # add (counts,proporation) to list
            second.append(
                (second_counts, second_counts / sum_interest_counts))  # 为什么除
        return np.asarray(first), np.asarray(second)
def get_submap(center, size, array):
    x, y = center
    height, width = size
    h, w = array.shape
    x1, x2, = x - int(height / 2), x + int((height + 1) / 2)
    y1, y2 = y - int(width / 2), y + int((width + 1) / 2)
    if x1 > -1 and x2 < h + 1 and y1 > -1 and y2 < w + 1:
        return array[x1:x2, y1:y2]
    else:
        return None


def split_to_submap(array, submap_size=None, submap_num=None):
    """必须保证整除

    Args:
        array ([type]): [description]
        submap_size ([type], optional): [description]. Defaults to None.
        submap_num ([type], optional): [description]. Defaults to None.
    """
    if submap_size is not None:
        if isinstance(submap_size, int):
            submap_size = (submap_size, submap_size)

        assert array.shape[0] % submap_size[0] == 0
        assert array.shape[1] % submap_size[1] == 0
        rows = int(array.shape[0] / submap_size[0])
        cols = int(array.shape[1] / submap_size[1])
    else:
        assert submap_num is not None
        rows, cols = submap_num
    row_arrays = np.split(array, rows, axis=0)
    splited = []
    for row_ in row_arrays:
        col_arrays = np.split(row_, cols, axis=1)
        splited.append(col_arrays)
    return splited

'''