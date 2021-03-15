'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: ZHANG XIANRUI
LastEditTime: 2021-03-15 18:49:37
'''

from openutils import *


def main(filepath):
    preds = np.load(filepath)
    classmap_array = preds_to_classmap(preds)
    colors = [(105, 105, 105), (70, 130, 180), (195, 100, 197), (60, 179, 113)]
    tumor_label = 4
    statistic = ClassmapStatistic(classmap_array, tumor_label, show=True)
    statistic.save_img(colors=colors)

    count_ratio_1st, count_ratio_2nd = statistic.proportion(
        tumor_label,
        first_label=[1, 2, 3],
        first_kernel_size=(10, 10),
        first_stride=2,
        second_label=[1, 2, 3],
        second_kernel_size=(3, 3),
        second_stride=2,
        threshold=(0.3, 0.95))

    tumor_mask = statistic.tumor_mask_preprocess(tumor_label,
                                                 disk=5,
                                                 small_object=50)
    distance, distance_rario_to_tumor = statistic.calc_distance(
        tumor_mask,
        thres=-20,
        first_label=[0, 1, 2, 3],
        tumor_label=tumor_label,
    )  # 每个感兴趣的label的far around inside的tile个数 ，以及这些个数与肿瘤个数的比例


if __name__ == '__main__':
    filepath = Path.home(
    ) / 'code/ptmc/data/ckpt_pencil_lr0.01_alpha_0.1_beat_0.1lamd_1000_epo_30/pred_map/8-26593 C1_5um_preds.npy'
    main(filepath)
