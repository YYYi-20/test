'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-11-11 13:01:12
'''

import numpy as np
from openutils import preds_to_classmap
from openutils import ClassmapStatistic

if __name__ == '__main__':
    preds = np.load(
        '/data/backup2/xianrui/data/ptmc/images/pred_maps/18 28528 5/preds.npy'
    )
    cls_map = preds_to_classmap(preds)
    names = [
        'ConneTissue', 'SquamoEpithe', 'Gland', 'LYM', 'SmooMus',
        'CanAssoStro', 'TUM'
    ]

    labels = list(
        range(1, 8)
    )  # 0 is background and empty tiles, it is set defaultly in ClassmapStatis
    tumor_label = 7
    colors = None  # use default color
    statis = ClassmapStatistic(cls_map, labels, names, colors=colors)
    interest, interation = statis.proportion(tumor_label=tumor_label,
                                             interest_label=range(1, 8),
                                             interaction_label=[2, 4],
                                             submap_size=(10, 10),
                                             sample_fraction=0.5)
    statis.save_img(save_path=None, show=True)  # donot save
    print(interest.shape, '\n', interation.shape)
    interest_score, interation_score = statis.score(interest), statis.score(
        interation)
    print(interest_score, '\n', interation_score)

    score_ = interest_score[[0, 1, 3]]  #用于计算entropy的score
    entropy_ = statis.entropy(score_)
    print(entropy_)
