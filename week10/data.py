"""
第十周数据模块。

目标：
1. 基于 week9 的对齐结果构造滑窗样本；
2. 生成 GCN 一步预测所需的节点特征、标签和图结构；
3. 输出可直接供 PyTorch / PyG 使用的 numpy 数组。
"""

from __future__ import annotations

import os
import sys
import importlib.util
from dataclasses import dataclass

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CURRENT_DIR)
_WEEK9_DIR = os.path.join(_ROOT_DIR, "week9")


def _load_week9_module(module_name: str, file_name: str):
    """
    以独立模块名加载 week9 下的脚本，避免与 week10/data.py 出现同名导入冲突。
    """
    module_path = os.path.join(_WEEK9_DIR, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_week9_data = _load_week9_module("week9_data_module", "data.py")
_week9_graph_utils = _load_week9_module("week9_graph_utils_module", "graph_utils.py")

load_aligned_pvod_data = _week9_data.load_aligned_pvod_data
build_distance_adjacency = _week9_graph_utils.build_distance_adjacency
adjacency_to_edge_index = _week9_graph_utils.adjacency_to_edge_index


@dataclass
class GCNPreparedData:
    """保存第十周 GCN 实验所需的数据。"""

    station_ids: list[str]
    feature_columns: list[str]
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    edge_index: np.ndarray
    adjacency: np.ndarray


def create_temporal_graph_samples(
    feature_tensor: np.ndarray,
    power_frame,
    seq_len: int = 4,
    pred_horizon: int = 1,
):
    """
    从多站点时序张量中切出滑窗样本。

    输入：
        feature_tensor: [time, stations, features]
        power_frame: DataFrame，形状 [time, stations]

    输出：
        x: [samples, stations, seq_len * features]
        y: [samples, stations]
    """
    time_steps, num_stations, num_features = feature_tensor.shape
    min_required_steps = seq_len + pred_horizon
    if time_steps < min_required_steps:
        raise ValueError(
            f"time_steps={time_steps} is too short for seq_len={seq_len} "
            f"and pred_horizon={pred_horizon}"
        )
    x_list = []
    y_list = []

    for t in range(seq_len, time_steps - pred_horizon + 1):
        history = feature_tensor[t - seq_len : t]  # [seq_len, stations, features]
        target = power_frame.iloc[t + pred_horizon - 1].to_numpy(dtype=np.float32)  # [stations]

        # 将时间维和特征维拼接，形成每个节点的输入向量。
        # 输出形状：[stations, seq_len * features]
        node_features = history.transpose(1, 0, 2).reshape(num_stations, seq_len * num_features)
        x_list.append(node_features.astype(np.float32))
        y_list.append(target.astype(np.float32))

    return np.stack(x_list).astype(np.float32), np.stack(y_list).astype(np.float32)


def prepare_gcn_data(
    station_ids: list[str] | None = None,
    feature_columns: list[str] | None = None,
    distance_threshold_km: float = 150.0,
    seq_len: int = 4,
    pred_horizon: int = 1,
    test_ratio: float = 0.2,
) -> GCNPreparedData:
    """
    构造第十周 GCN 实验所需数据。
    """
    aligned = load_aligned_pvod_data(
        station_ids=station_ids,
        feature_columns=feature_columns,
    )
    adjacency_df = build_distance_adjacency(
        aligned.metadata,
        threshold_km=distance_threshold_km,
    )
    edge_index = adjacency_to_edge_index(adjacency_df.to_numpy(dtype=np.float32))
    x, y = create_temporal_graph_samples(
        aligned.feature_tensor,
        aligned.power_frame,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
    )

    n_samples = len(x)
    n_train = int(n_samples * (1 - test_ratio))
    if n_train <= 0 or n_train >= n_samples:
        raise ValueError(
            f"invalid train/test split: total_samples={n_samples}, "
            f"test_ratio={test_ratio}"
        )

    return GCNPreparedData(
        station_ids=aligned.station_ids,
        feature_columns=aligned.feature_columns,
        train_x=x[:n_train],
        train_y=y[:n_train],
        test_x=x[n_train:],
        test_y=y[n_train:],
        edge_index=edge_index,
        adjacency=adjacency_df.to_numpy(dtype=np.float32),
    )
