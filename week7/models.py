"""
第七周 CNN 模型模块。

包含：
1. 单尺度 CNN 回归模型；
2. 多尺度 Multi-sight CNN 模型。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _normalize_kernel_size(kernel_size: int | tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    将卷积核统一为二维形式，并计算对应的 same padding。

    约定：
    - int: 视为方形卷积核 k x k；
    - tuple[int, int]: 视为 (kernel_height, kernel_width)。
    """
    if isinstance(kernel_size, int):
        kernel_hw = (kernel_size, kernel_size)
    else:
        if len(kernel_size) != 2:
            raise ValueError(f"kernel_size must be int or tuple[int, int], got {kernel_size}")
        kernel_hw = (int(kernel_size[0]), int(kernel_size[1]))

    padding_hw = (kernel_hw[0] // 2, kernel_hw[1] // 2)
    return kernel_hw, padding_hw


class ConvBlock(nn.Module):
    """基础卷积特征提取块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int]):
        super().__init__()
        kernel_hw, padding_hw = _normalize_kernel_size(kernel_size)
        self.block = nn.Sequential(
            # 当 kernel_size 为 tuple 时，使用矩形卷积核。
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_hw, padding=padding_hw),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_hw, padding=padding_hw),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: [Batch, Channels, Height, Width]
        out = self.block(x)
        return out.flatten(start_dim=1)


class CNNRegressor(nn.Module):
    """
    单尺度 CNN 回归模型。

    输入形状：
        x: [Batch, 1, H, W]
    输出形状：
        y_hat: [Batch, output_dim]
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        output_dim: int = 1,
        kernel_width: int = 5,
    ):
        super().__init__()
        # 与论文 Multi-sight 设定保持一致：宽度固定为 5，长度沿时间方向变化。
        self.encoder = ConvBlock(
            in_channels,
            hidden_channels,
            kernel_size=(kernel_size, kernel_width),
        )
        self.head = nn.Linear(hidden_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


class MultiSightCNN(nn.Module):
    """
    Multi-sight 多分支 CNN。

    每个分支使用不同长度的矩形卷积核，宽度固定为 5：
    - (3, 5)
    - (5, 5)
    - (7, 5)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        output_dim: int = 1,
        dropout: float = 0.1,
        kernel_width: int = 5,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.kernel_width = kernel_width
        self.branches = nn.ModuleList(
            [
                ConvBlock(in_channels, hidden_channels, kernel_size=(k, kernel_width))
                for k in kernel_sizes
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * len(kernel_sizes), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: [Batch, 1, H, W]
        multi_scale_features = [branch(x) for branch in self.branches]
        fused = torch.cat(multi_scale_features, dim=1)
        return self.fusion(fused)
