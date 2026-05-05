"""
第七周数据变换模块。

目标：
1. 将一维时序样本 [Batch, T] 变换成二维“图像”格式 [Batch, 1, H, W]。
2. 为后续 Conv2d 与 Multi-sight CNN 提供统一输入。
"""

from __future__ import annotations

import numpy as np


def reshape_sequences_to_2d(
    x: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    将一批一维序列重排成二维矩阵格式。

    Args:
        x: 输入数组，形状应为 [Batch, T]。
        height: 变换后的高度 H。
        width: 变换后的宽度 W。

    Returns:
        形状为 [Batch, 1, H, W] 的四维张量。
    """
    x = np.asarray(x, dtype=np.float32)
    expected_t = height * width

    # 当上游没有构造出任何样本时，NumPy 常会得到形状为 (0,) 的空数组。
    if x.size == 0:
        return np.empty((0, 1, height, width), dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"x 必须是 [Batch, T]，当前形状为 {x.shape}")
    if x.shape[1] != expected_t:
        raise ValueError(f"T 必须等于 height * width = {expected_t}，当前 T = {x.shape[1]}")
    return x.reshape(x.shape[0], 1, height, width)


def make_2d_forecasting_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将训练集/测试集滑窗样本统一转换为 2D CNN 输入。

    输入形状：
        x_train: [N_train, T]
        y_train: [N_train, H_out]
        x_test: [N_test, T]
        y_test: [N_test, H_out]

    输出形状：
        x_train_2d: [N_train, 1, H, W]
        y_train: [N_train, H_out]
        x_test_2d: [N_test, 1, H, W]
        y_test: [N_test, H_out]
    """
    return (
        reshape_sequences_to_2d(x_train, height, width),
        np.asarray(y_train, dtype=np.float32),
        reshape_sequences_to_2d(x_test, height, width),
        np.asarray(y_test, dtype=np.float32),
    )
