"""
第十一周时空融合模型模块。

主要复现训练计划中的简化结构：
Input -> GCN -> LSTM -> Output。
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise ImportError("Week11 models require PyTorch. Please install torch first.") from exc

try:
    from torch_geometric.nn import GCNConv
except Exception as exc:
    raise ImportError("Week11 models require torch_geometric. Please install torch-geometric first.") from exc


class IndependentLSTM(nn.Module):
    """
    单站点独立预测基线。

    该模型不使用 edge_index，也不进行邻居聚合。
    它把每个站点视为一条独立时间序列，用于和 GCN-LSTM 对比。

    Input shape:
        x: [Batch, Seq_Len, Num_Stations, Features]
    Output shape:
        y_hat: [Batch, Num_Stations]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # batch_first=True 表示 LSTM 输入为 [Batch, Seq_Len, Features]。
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, num_stations, num_features = x.shape

        # 将站点维并入 batch 维，实现每个站点独立建模。
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_stations, seq_len, num_features)
        out, _ = self.lstm(x)
        last_hidden = self.dropout(out[:, -1, :])
        y_hat = self.head(last_hidden)
        return y_hat.reshape(batch_size, num_stations)


class GCNTemporalBranch(nn.Module):
    """
    一个时空融合分支。

    先在每个时间步使用 GCN 聚合空间邻居信息，
    再对每个站点的空间特征序列使用 LSTM 提取时间依赖。

    Input shape:
        x: [Batch, Seq_Len, Num_Stations, Features]
        edge_index: [2, Num_Edges]
    Output shape:
        y_hat: [Batch, Num_Stations]
    """

    def __init__(
        self,
        input_dim: int,
        gcn_hidden_dim: int = 32,
        lstm_hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gcn = GCNConv(input_dim, gcn_hidden_dim)
        self.lstm = nn.LSTM(gcn_hidden_dim, lstm_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_stations, _ = x.shape
        spatial_steps = []

        # 对每个样本、每个时间步使用同一张站点图。
        for batch_idx in range(batch_size):
            per_time = []
            for time_idx in range(seq_len):
                node_features = x[batch_idx, time_idx]  # [Num_Stations, Features]
                spatial_features = self.gcn(node_features, edge_index)
                spatial_features = torch.relu(spatial_features)
                per_time.append(spatial_features)
            spatial_steps.append(torch.stack(per_time, dim=0))

        # GCN 输出序列形状: [Batch, Seq_Len, Num_Stations, GCN_Hidden]
        spatial_seq = torch.stack(spatial_steps, dim=0)

        # 调整为 LSTM 输入形状: [Batch * Num_Stations, Seq_Len, GCN_Hidden]
        temporal_input = spatial_seq.permute(0, 2, 1, 3).reshape(
            batch_size * num_stations,
            seq_len,
            -1,
        )
        out, _ = self.lstm(temporal_input)
        last_hidden = self.dropout(out[:, -1, :])
        y_hat = self.head(last_hidden)
        return y_hat.reshape(batch_size, num_stations)


class GCNLSTMRegressor(nn.Module):
    """单尺度 GCN-LSTM 模型，默认用于 15 分钟输入分支。"""

    def __init__(
        self,
        input_dim: int,
        gcn_hidden_dim: int = 32,
        lstm_hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.branch = GCNTemporalBranch(
            input_dim=input_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.branch(x, edge_index)


class MultiScaleGCNLSTMRegressor(nn.Module):
    """
    简化版多尺度融合模型。

    15 分钟分支和 1 小时分支分别经过 GCN-LSTM，
    最后在预测值层面做 Add 融合。
    """

    def __init__(
        self,
        input_dim: int,
        gcn_hidden_dim: int = 32,
        lstm_hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.branch_15min = GCNTemporalBranch(
            input_dim=input_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.branch_1h = GCNTemporalBranch(
            input_dim=input_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x_15min: torch.Tensor, x_1h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.branch_15min(x_15min, edge_index) + self.branch_1h(x_1h, edge_index)
