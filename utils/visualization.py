import matplotlib.pyplot as plt
import numpy as np


def get_row_col(num_pic):
    """
    计算行列的值
    :param num_pic: 特征图的数量
    :return:
    """
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(feature_batch):
    """
    创建特征子图，创建叠加后的特征图
    :param feature_batch: 一个卷积层所有特征图
    :return:
    """
    feature_map = np.squeeze(feature_batch, axis=0)

    feature_map_combination = []
    plt.figure(figsize=(8, 7))
    row, col = get_row_col(feature_map.shape[0])
    # 将 每一层卷积的特征图，拼接层 5 × 5
    for i in range(feature_map.shape[0]):
        feature_map_split = feature_map[i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i+1)
        plt.imshow(feature_map_split)
        plt.axis('off')
    plt.show()


def visualize_feature_map_sum(feature_batch):
    """
    将每张子图进行相加
    :param feature_batch:
    :return:
    """
    feature_map = np.squeeze(feature_batch, axis=0)

    feature_map_combination = []

    # 取出 featurn map 的数量
    num_pic = feature_map.shape[0]

    # 将 每一层卷积的特征图，拼接层 5 × 5
    for i in range(0, num_pic):
        feature_map_split = feature_map[i]
        feature_map_combination.append(feature_map_split)

    # 按照特征图 进行 叠加代码

    feature_map_sum = sum(one for one in feature_map_combination)
    plt.imshow(feature_map_sum)

    plt.show()
