"""
第十一周时空融合数据模块。

本模块复用第九周的 PVOD 多站点数据对齐与图构建逻辑，
进一步构造 GCN-LSTM 所需的时空序列样本。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CURRENT_DIR)
_WEEK9_DIR = os.path.join(_ROOT_DIR, "week9") #复用第九周的数据对齐和图构建函数


def _load_module(module_name: str, file_path: str):
    """按文件路径加载模块，避免 notebook 中不同周次的 data.py 出现同名冲突。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_week9_data = _load_module("week9_data_for_week11", os.path.join(_WEEK9_DIR, "data.py"))
_week9_graph = _load_module("week9_graph_for_week11", os.path.join(_WEEK9_DIR, "graph_utils.py"))

load_aligned_pvod_data = _week9_data.load_aligned_pvod_data
build_distance_adjacency = _week9_graph.build_distance_adjacency
build_correlation_matrix = _week9_graph.build_correlation_matrix
adjacency_to_edge_index = _week9_graph.adjacency_to_edge_index


@dataclass
class STPreparedData:
    """
    保存第十一、十二周实验所需的数据。

    train_x_15min / test_x_15min:
        [samples, seq_len_15min, stations, features]
    train_x_1h / test_x_1h:
        [samples, seq_len_1h, stations, features]
    train_y / test_y:
        [samples, stations]
    edge_index:
        [2, num_edges]，供 PyG 的 GCNConv 使用。
    """

    station_ids: list[str]
    feature_columns: list[str]
    train_x_15min: np.ndarray
    train_x_1h: np.ndarray
    train_y: np.ndarray
    test_x_15min: np.ndarray
    test_x_1h: np.ndarray
    test_y: np.ndarray
    edge_index: np.ndarray
    adjacency: np.ndarray
    correlation: np.ndarray


def create_multiscale_spatiotemporal_samples(
    feature_tensor: np.ndarray,
    power_frame,
    seq_len_15min: int = 16,
    seq_len_1h: int = 4,
    hour_stride: int = 4,
    pred_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构造对齐的 15 分钟分支和 1 小时分支样本。

    Args:
        feature_tensor: 原始对齐特征张量，形状为 [time, stations, features]。
        power_frame: 功率矩阵，形状为 [time, stations]。
        seq_len_15min: 15 分钟分支使用的历史步数。
        seq_len_1h: 1 小时分支使用的历史步数。
        hour_stride: 15 分钟数据中每隔 4 个点取一个 1 小时点。
        pred_horizon: 预测步长，单位仍然是 15 分钟时间步。

    Returns:
        x_15min: [samples, seq_len_15min, stations, features]
        x_1h: [samples, seq_len_1h, stations, features]
        y: [samples, stations]
    """
    time_steps = feature_tensor.shape[0]
    required_history = max(seq_len_15min, seq_len_1h * hour_stride)
    min_required_steps = required_history + pred_horizon
    if time_steps < min_required_steps:
        raise ValueError(
            f"time_steps={time_steps} is too short for required_history={required_history} "
            f"and pred_horizon={pred_horizon}"
        )

    x_15min_list = []
    x_1h_list = []
    y_list = []

    for t in range(required_history, time_steps - pred_horizon + 1):
        # 15 分钟分支保留最近的连续高频历史。
        history_15min = feature_tensor[t - seq_len_15min : t]

        # 1 小时分支从 15 分钟数据中按 hour_stride 下采样得到低频历史。
        hour_indices = np.arange(t - seq_len_1h * hour_stride, t, hour_stride)
        history_1h = feature_tensor[hour_indices]

        target = power_frame.iloc[t + pred_horizon - 1].to_numpy(dtype=np.float32)

        x_15min_list.append(history_15min.astype(np.float32))
        x_1h_list.append(history_1h.astype(np.float32))
        y_list.append(target.astype(np.float32))

    return (
        np.stack(x_15min_list).astype(np.float32),
        np.stack(x_1h_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
    )


def prepare_st_data(
    station_ids: list[str] | None = None,
    feature_columns: list[str] | None = None,
    distance_threshold_km: float = 150.0,
    seq_len_15min: int = 16,
    seq_len_1h: int = 4,
    pred_horizon: int = 1,
    test_ratio: float = 0.2,
) -> STPreparedData:
    """
    构造第十一、十二周实验数据。

    处理流程：
    1. 读取并对齐 PVOD 多站点数据；
    2. 用经纬度距离图生成 GCN 的 edge_index；
    3. 计算 Pearson 相关矩阵，供第十二周可视化；
    4. 切分训练集和测试集。
    """
    aligned = load_aligned_pvod_data(
        station_ids=station_ids,
        feature_columns=feature_columns,
    )
    adjacency_df = build_distance_adjacency(
        aligned.metadata,
        threshold_km=distance_threshold_km,
    )
    corr_df = build_correlation_matrix(aligned.power_frame)

    x_15min, x_1h, y = create_multiscale_spatiotemporal_samples(
        aligned.feature_tensor,
        aligned.power_frame,
        seq_len_15min=seq_len_15min,
        seq_len_1h=seq_len_1h,
        pred_horizon=pred_horizon,
    )

    n_samples = len(y)
    n_train = int(n_samples * (1 - test_ratio))
    if n_train <= 0 or n_train >= n_samples:
        raise ValueError(
            f"invalid train/test split: total_samples={n_samples}, test_ratio={test_ratio}"
        )

    return STPreparedData(
        station_ids=aligned.station_ids,
        feature_columns=aligned.feature_columns,
        train_x_15min=x_15min[:n_train],
        train_x_1h=x_1h[:n_train],
        train_y=y[:n_train],
        test_x_15min=x_15min[n_train:],
        test_x_1h=x_1h[n_train:],
        test_y=y[n_train:],
        edge_index=adjacency_to_edge_index(adjacency_df.to_numpy(dtype=np.float32)), #将邻接矩阵转换为 edge_index（COO格式）
        adjacency=adjacency_df.to_numpy(dtype=np.float32),
        correlation=corr_df.to_numpy(dtype=np.float32),
    )
