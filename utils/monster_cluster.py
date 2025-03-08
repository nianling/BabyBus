# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'


import numpy as np
from scipy.spatial.distance import pdist, squareform


class MonsterCluster:
    def __init__(self, monster_xywh_list, max_distance=400):
        self.monster_xywh_list = monster_xywh_list
        self.max_distance = max_distance
        self.coordinates = np.array([[m[0], m[1]] for m in monster_xywh_list])

    def find_densest_cluster(self):
        if len(self.coordinates) == 0:
            return None, 0

        # 计算所有点之间的距离矩阵
        dist_matrix = squareform(pdist(self.coordinates))

        max_count = 0
        best_center = None

        # 对每个点进行检查
        for i, point in enumerate(self.coordinates):
            # 找出在最大距离范围内的所有点
            in_range = dist_matrix[i] <= self.max_distance
            count = np.sum(in_range)

            if count > max_count:
                max_count = count
                # 计算在范围内的所有点的平均位置作为中心
                best_center = np.mean(self.coordinates[in_range], axis=0)

        return best_center.tolist() if best_center is not None else None, max_count
