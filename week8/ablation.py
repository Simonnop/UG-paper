"""
第八周消融实验数据模块。

核心思路：
1. 对照组 A/B 不做分解，直接把原始序列送入 CNN；
2. 实验组 C/D 先做二项分解：
   - trend 分量送入 MLP；
   - periodic 分量 reshape 成 2D 后送入 CNN；
   - 两个分支的输出做相加融合，得到最终预测。
"""

from __future__ import annotations

import os
import sys

import numpy as np

# 兼容 notebook 直接运行的导入路径。
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CURRENT_DIR)
_WEEK6_DIR = os.path.join(_ROOT_DIR, "week6")
_WEEK7_DIR = os.path.join(_ROOT_DIR, "week7")

for _path in (_WEEK6_DIR, _WEEK7_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from decomposition import moving_average_trend
from transforms import reshape_sequences_to_2d


def decompose_window(
    window: np.ndarray,
    trend_window: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对单个滑窗序列做二项分解。

    Args:
        window: 单个样本窗口，形状为 [T]。
        trend_window: 移动平均窗口长度。

    Returns:
        trend: 趋势项，形状为 [T]
        periodic: 周期项，形状为 [T]
    """
    window = np.asarray(window, dtype=np.float32)
    trend = moving_average_trend(window, window=trend_window).astype(np.float32)
    periodic = (window - trend).astype(np.float32)
    return trend, periodic


def batch_decompose(
    x: np.ndarray,
    trend_window: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对一批原始序列做二项分解。

    Args:
        x: 原始输入，形状为 [Batch, T]。
        trend_window: 移动平均窗口长度。

    Returns:
        trend_array: [Batch, T]
        periodic_array: [Batch, T]
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )
    if x.ndim != 2:
        raise ValueError(f"x 必须是 [Batch, T]，当前形状为 {x.shape}")

    trend_list = []
    periodic_list = []
    for sample in x:
        trend, periodic = decompose_window(sample, trend_window=trend_window)
        trend_list.append(trend)
        periodic_list.append(periodic)

    trend_array = np.stack(trend_list).astype(np.float32)
    periodic_array = np.stack(periodic_list).astype(np.float32)
    return trend_array, periodic_array


def make_ablation_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    height: int,
    width: int,
    trend_window: int = 25,
) -> dict[str, np.ndarray]:
    """
    构造四组实验共用的数据输入。

    Returns:
        一个字典，包含：
        - raw_train_2d / raw_test_2d: 原始输入二维化，[N, 1, H, W]
        - trend_train_1d / trend_test_1d: 趋势项一维输入，[N, T]
        - periodic_train_2d / periodic_test_2d: 周期项二维输入，[N, 1, H, W]
        - train_y / test_y: 标签，[N, H_out]
    """
    raw_train_2d = reshape_sequences_to_2d(x_train, height=height, width=width)
    raw_test_2d = reshape_sequences_to_2d(x_test, height=height, width=width)

    trend_train_1d, periodic_train_1d = batch_decompose(
        x_train,
        trend_window=trend_window,
    )
    trend_test_1d, periodic_test_1d = batch_decompose(
        x_test,
        trend_window=trend_window,
    )

    periodic_train_2d = reshape_sequences_to_2d(
        periodic_train_1d,
        height=height,
        width=width,
    )
    periodic_test_2d = reshape_sequences_to_2d(
        periodic_test_1d,
        height=height,
        width=width,
    )

    return {
        "raw_train_2d": raw_train_2d,
        "raw_test_2d": raw_test_2d,
        "trend_train_1d": trend_train_1d.astype(np.float32),
        "trend_test_1d": trend_test_1d.astype(np.float32),
        "periodic_train_2d": periodic_train_2d.astype(np.float32),
        "periodic_test_2d": periodic_test_2d.astype(np.float32),
        "train_y": np.asarray(y_train, dtype=np.float32),
        "test_y": np.asarray(y_test, dtype=np.float32),
    }
