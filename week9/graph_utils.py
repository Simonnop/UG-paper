"""
第九周构图工具模块。

目标：
1. 基于经纬度构造距离图；
2. 基于功率序列构造相关性图；
3. 提供邻接矩阵归一化与 edge_index 转换工具，便于第十周复用。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def haversine_distance_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """计算两点球面距离，单位 km。"""
    radius_km = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return radius_km * c


def build_distance_matrix(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    根据 metadata 中的经纬度构造站点距离矩阵。
    """
    station_ids = metadata["Station_ID"].tolist()
    distance_matrix = np.zeros((len(station_ids), len(station_ids)), dtype=np.float32)

    for i, row_i in metadata.iterrows():
        for j, row_j in metadata.iterrows():
            distance_matrix[i, j] = haversine_distance_km(
                row_i["Longitude"],
                row_i["Latitude"],
                row_j["Longitude"],
                row_j["Latitude"],
            )

    return pd.DataFrame(distance_matrix, index=station_ids, columns=station_ids)


def build_distance_adjacency(
    metadata: pd.DataFrame,
    threshold_km: float = 150.0,
    include_self_loop: bool = True,
) -> pd.DataFrame:
    """
    根据距离阈值构造邻接矩阵。

    规则：
    - 距离 <= threshold_km 时视为相邻，记为 1；
    - 否则记为 0。
    """
    distance_df = build_distance_matrix(metadata)
    adjacency = (distance_df <= threshold_km).astype(np.float32)

    if not include_self_loop:
        np.fill_diagonal(adjacency.values, 0.0)
    return adjacency


def build_correlation_matrix(power_frame: pd.DataFrame) -> pd.DataFrame:
    """
    根据多站点功率序列计算 Pearson 相关系数矩阵。
    """
    return power_frame.corr(method="pearson")


def build_correlation_adjacency(
    power_frame: pd.DataFrame,
    threshold: float = 0.8,
    include_self_loop: bool = True,
) -> pd.DataFrame:
    """
    根据相关系数阈值构造相关性图。
    """
    corr_df = build_correlation_matrix(power_frame)
    adjacency = (corr_df >= threshold).astype(np.float32)

    if not include_self_loop:
        np.fill_diagonal(adjacency.values, 0.0)
    return adjacency


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """
    计算对称归一化邻接矩阵：D^{-1/2} A D^{-1/2}
    """
    adjacency = np.asarray(adjacency, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    d_inv_sqrt = np.diag(inv_sqrt_degree)
    return d_inv_sqrt @ adjacency @ d_inv_sqrt


def adjacency_to_edge_index(adjacency: np.ndarray) -> np.ndarray:
    """
    将邻接矩阵转换为 edge_index 形式，便于 PyG 使用。

    输出形状：
        [2, num_edges]
        第一行：源节点 src
        第二行：目标节点 dst
    """
    adjacency = np.asarray(adjacency)
    src, dst = np.nonzero(adjacency > 0) #np.nonzero()：找出所有 值 > 0 的位置的行坐标 (src) + 列坐标 (dst) 
    return np.vstack([src, dst]).astype(np.int64)
