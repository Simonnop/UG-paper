"""
第七周 CNN 模型模块。

包含：
1. 单尺度 CNN 回归模型；
2. 多尺度 Multi-sight CNN 模型。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """基础卷积特征提取块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            # 第一个卷积层：提取局部模式。
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # 第二个卷积层：进一步组合局部特征。
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # 自适应池化：将不同位置的响应汇总到固定长度特征向量。
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

    def __init__(self, in_channels: int = 1, hidden_channels: int = 16, kernel_size: int = 3, output_dim: int = 1):
        super().__init__()
        self.encoder = ConvBlock(in_channels, hidden_channels, kernel_size=kernel_size)
        self.head = nn.Linear(hidden_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


class MultiSightCNN(nn.Module):
    """
    Multi-sight 多分支 CNN。

    每个分支使用不同卷积核大小，模拟论文中“多视角观察”：
    小卷积核更关注高频、局部波动；
    大卷积核更关注低频、平滑趋势与更大感受野。
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        # 不同分支对应不同感受野。
        self.branches = nn.ModuleList(
            [
                ConvBlock(in_channels, hidden_channels, kernel_size=k)
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
        # 拼接多个尺度的特征，再做融合回归。
        fused = torch.cat(multi_scale_features, dim=1)
        return self.fusion(fused)
