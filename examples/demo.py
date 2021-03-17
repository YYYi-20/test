'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: Please set LastEditors
LastEditTime: 2021-03-17 11:48:09
'''

from openutils import *


def main(filepath):
    preds = np.load(filepath)
    classmap_array = preds_to_classmap(preds)
    colors = [(105, 105, 105), (70, 130, 180), (195, 100, 197), (60, 179, 113)]
    tumor_label = 4
    statistic = ClassmapStatistic(classmap_array, tumor_label, show=True)
    statistic.save_img(colors=colors)

    rois = statistic.roi_count(interest_class=[1, 2, 3, 4],
                               kernel_size=5,
                               stride=5,
                               threshold_non_back=(0.5, 1),
                               threshold_other={4: (0.3, 1)})

    tumor_mask = statistic.tumor_mask_preprocess(tumor_label,
                                                 disk=3,
                                                 small_object=20)
    # cls_map = statistic.cls_map
    # map1 = np.where(cls_map == 4, 0, cls_map)
    # new_cls_map = np.where(tumor_mask == 255, 4, map1)
    distance = statistic.calc_distance(tumor_mask,
                                       interest_class=[0, 1, 2, 3, 4])
    # 每个感兴趣的label的far around inside的tile个数 ，以及这些个数与肿瘤个数的比例
    return rois, distance


if __name__ == '__main__':
    filepath = Path.home(
    ) / 'code/ptmc/data/ckpt_pencil_lr0.01_alpha_0.1_beat_0.1lamd_1000_epo_30/pred_map/8-26593 C1_5um_preds.npy'
    main(filepath)
