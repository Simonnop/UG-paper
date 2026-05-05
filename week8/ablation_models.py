"""
第八周消融实验模型模块。

包含两类实验模型：
1. 对照组：直接使用原始序列 + CNN；
2. 实验组：trend 通过 MLP，periodic 通过 CNN，最后将两支输出相加。
"""

from __future__ import annotations

import importlib.util
import os

import torch
import torch.nn as nn

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CURRENT_DIR)


def _load_module(module_name: str, file_path: str):
    """按文件路径动态加载模块，避免目录不是 package 时导入失败。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_name} <- {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_week5_models = _load_module(
    "week5_models_for_ablation",
    os.path.join(_ROOT_DIR, "week5", "models.py"),
)
_week7_models = _load_module(
    "week7_models_for_ablation",
    os.path.join(_ROOT_DIR, "week7", "models.py"),
)
# 复用前几周的MLP、CNN模型
MLP = _week5_models.MLP
CNNRegressor = _week7_models.CNNRegressor
MultiSightCNN = _week7_models.MultiSightCNN


class TrendPeriodicSingleSightModel(nn.Module):
    """
    趋势项 + 周期项融合模型（单视角版本）。

    输入：
        trend_x: [Batch, T]
        periodic_x: [Batch, 1, H, W]

    输出：
        y_hat: [Batch, output_dim]
    """

    def __init__(
        self,
        trend_input_dim: int,
        trend_hidden_dims: tuple[int, ...] = (64, 32),
        periodic_kernel_size: int = 3,
        periodic_hidden_channels: int = 16,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trend_branch = MLP(
            input_dim=trend_input_dim,
            hidden_dims=trend_hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.periodic_branch = CNNRegressor(
            in_channels=1,
            hidden_channels=periodic_hidden_channels,
            kernel_size=periodic_kernel_size,
            output_dim=output_dim,
        )

    def forward(self, trend_x: torch.Tensor, periodic_x: torch.Tensor) -> torch.Tensor:
        trend_out = self.trend_branch(trend_x)
        periodic_out = self.periodic_branch(periodic_x)
        return trend_out + periodic_out


class TrendPeriodicMultiSightModel(nn.Module):
    """
    趋势项 + 周期项融合模型（多视角版本）。

    输入：
        trend_x: [Batch, T]
        periodic_x: [Batch, 1, H, W]

    输出：
        y_hat: [Batch, output_dim]
    """

    def __init__(
        self,
        trend_input_dim: int,
        trend_hidden_dims: tuple[int, ...] = (64, 32),
        periodic_kernel_sizes: tuple[int, ...] = (3, 5, 7),
        periodic_hidden_channels: int = 16,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trend_branch = MLP(
            input_dim=trend_input_dim,
            hidden_dims=trend_hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.periodic_branch = MultiSightCNN(
            in_channels=1,
            hidden_channels=periodic_hidden_channels,
            kernel_sizes=periodic_kernel_sizes,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, trend_x: torch.Tensor, periodic_x: torch.Tensor) -> torch.Tensor:
        trend_out = self.trend_branch(trend_x)
        periodic_out = self.periodic_branch(periodic_x)
        return trend_out + periodic_out
