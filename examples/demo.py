'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: Please set LastEditors
LastEditTime: 2021-03-18 16:05:54
'''

from openutils import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(process)d - %(message)s',
    # filename='data/1.log',
)


def main(filepath='../data/preds/19-17298 C1_5um_preds.npy'):
    logging.info(f'procress {filepath}')
    preds = np.load(filepath)
    classmap_array = preds_to_classmap(preds)
    colors = [(105, 105, 105), (70, 130, 180), (195, 100, 197), (60, 179, 113)]
    tumor_label = 4
    statistic = ClassmapStatistic(classmap_array, tumor_label, show=True)
    statistic.save_img(colors=colors)
    if not statistic.tumor_exist:
        return

    rois = statistic.roi_count(interest_class=[1, 2, 3, 4],
                               kernel_size=10,
                               stride=10,
                               threshold_non_back=(0.5, 1),
                               threshold_other={4: (0.3, 1)})
    if rois is None:  # 没有合适的ROI
        return
    rois.to_csv(Path('data/rois') / f'{i.stem}.csv')
    rois.columns = ['total', 'non_back', 'foli', 'fiber', 'lym', 'tumor']

    # 每个感兴趣的label的far around inside的tile个数 ，以及这些个数与肿瘤个数的比例
    def fun_(top=['foli', 'fiber', 'lym', 'tumor'], down=rois['non_back']):
        fraction = pd.concat([rois[i] / down for i in top], axis=1)
        fraction.columns = top
        fraction.head()
        ##
        whole = fraction.loc['whole_slide']
        roi_mean = fraction.iloc[1:].mean(axis=0)
        return whole.values, roi_mean.values

    whole_infiltratio_ratio, roi_infiltration_ratio = fun_(
        ['foli', 'fiber', 'lym', 'tumor'], rois['non_back'])
    whole_intratumor_ratio, roi_intratumor_ratio = fun_(
        ['foli', 'fiber', 'lym'], rois['tumor'])
    ratio_features = [
        whole_infiltratio_ratio, roi_infiltration_ratio,
        whole_intratumor_ratio, roi_intratumor_ratio
    ]

    extractor = FeatureExtractor()
    ##
    entropy_features = [extractor.entropy(i) for i in ratio_features]

    tumor_mask = statistic.tumor_mask_preprocess(tumor_label,
                                                 disk=3,
                                                 small_object=20)
    distance = statistic.calc_distance(tumor_mask,
                                       interest_class=[0, 1, 2, 3, 4])
    if distance is None:  # 没有轮廓
        return
    distance.to_csv(Path('data/distance') / f'{i.stem}.csv')

    distance.head()
    distance.columns = ['bkg', 'foli', 'fiber', 'lym', 'tumor']

    # 是用Gauss混合分布拟合出inside around far  还是直接设置阈值？？
    # GMM这里给出函数了，但是结果暂时没使用
    # X = distance['lym'].dropna().values.reshape([-1, 1]).astype('float32')
    # if len(X) > 3:
    #     labels, means, covariances, weights = extractor.gmm(X,
    #                                                         n_components=3,
    #                                                         means_init=None)

    #distance
    # 使用distance_classification直接划分远近
    distance_features = []
    for tissue_type in ['foli', 'fiber', 'lym']:
        X = distance[tissue_type].dropna().values
        if len(X) > 0:
            inside, around, far = extractor.distance_classification(X,
                                                                    thres=-20)
            distance_features += [inside, around, far]
        else:
            distance_features += [0, 0, 0]

    all_features = np.hstack(ratio_features).tolist() + entropy_features + [
        inside, around, far
    ]
    return all_features


if __name__ == '__main__':
    filepaths = Path('data/preds').glob('*_5um_preds.npy')
    # filepaths = load_json('data/1.json')
    all_features = []
    for i in filepaths:
        i = Path(i)
        # main(i)
        # print(i)
        all_features.append(main(i))
        # print(i.stem.split('_')[0] + '.mrxs')
