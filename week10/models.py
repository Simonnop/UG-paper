"""
第十周 GCN 模型模块。

说明：
1. 该模块优先使用 PyTorch Geometric 的 GCNConv；
2. 若当前环境没有安装 torch_geometric，将抛出明确错误；
3. 本周目标是做一个简单的空间图卷积模型，输入节点特征，输出每个站点下一时刻功率预测值。
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise ImportError("第十周模型依赖 PyTorch，请先在 notebook 使用的环境中安装 torch。") from exc

try:
    from torch_geometric.nn import GCNConv
except Exception as exc:
    raise ImportError(
        "第十周模型依赖 torch_geometric。请在 notebook 使用的环境中安装 torch_geometric 后再运行。"
    ) from exc


class GCNRegressor(nn.Module):
    """
    简单 GCN 回归模型。

    输入：
        x: [num_nodes, input_dim]
        edge_index: [2, num_edges]

    输出：
        y_hat: [num_nodes]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.head(x)
        return x.squeeze(-1)
