'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-11-11 15:03:37
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
    statis = ClassmapStatistic(cls_map,
                               labels,
                               names,
                               colors=colors,
                               show=True,
                               seed=0)
    interest, interation = statis.proportion(
        tumor_label=tumor_label,
        interest_label=range(1, 8),
        interaction_label=[2, 4],
        submap_size=(10, 10),
        sample_fraction=0.5,
    )
    statis.save_img(save_path=None)  # path is Noe, we donot save
    print('interest.shape: ', interest.shape)
    print('interation.shape: ', interation.shape)
    interest_score = statis.score(interest)
    interation_score = statis.score(interation)
    print(
        f'interest_score: {interest_score}\ninteration_score: {interation_score}'
    )

    score_ = interest_score[[0, 1, 3]]  #用于计算entropy的score
    entropy_ = statis.entropy(score_)
    print(f'entropy_: {entropy_}')

    tumor_mask = statis.tumor_mask_preprocess(tumor_label=tumor_label,
                                              disk=8,
                                              small_object=50)

    statis.calc_distance(tumor_mask,
                         thres=-20,
                         interest_label=[0, 1, 3],
                         tumor_label=tumor_label,
                         ratio=True)  # far around inside 个数 ，以及 与肿瘤个数的比例
    print(
        f'distance: {statis.distance} \n\n ration_to_tumor: {statis.distance_ratio}'
    )
