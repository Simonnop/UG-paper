"""
第十周训练模块。

提供：
1. numpy -> torch 的数据封装；
2. GCN 训练与评估循环；
3. MAE / RMSE 指标计算。
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise ImportError("第十周训练模块依赖 PyTorch，请先安装 torch。") from exc


class GraphSequenceDataset(torch.utils.data.Dataset):
    """
    图时序样本数据集。

    输入：
        x: [samples, num_nodes, input_dim]
        y: [samples, num_nodes]
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def evaluate_gcn(model, data_loader, edge_index, criterion, device):
    """评估 GCN 模型。"""
    if len(data_loader.dataset) == 0:
        raise ValueError("data_loader is empty")
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for x_b, y_b in data_loader:
            batch_preds = []
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            for sample_idx in range(x_b.size(0)):
                pred = model(x_b[sample_idx], edge_index)
                batch_preds.append(pred)
            y_hat = torch.stack(batch_preds, dim=0)
            loss = criterion(y_hat, y_b)
            total_loss += loss.item() * x_b.size(0)
            preds.append(y_hat.cpu().numpy())
            targets.append(y_b.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss, mae, rmse


def train_gcn(
    model,
    train_loader,
    test_loader,
    edge_index,
    epochs: int,
    lr: float,
    device,
):
    """训练 GCN 模型。"""
    if len(train_loader.dataset) == 0:
        raise ValueError("train_loader is empty")
    if len(test_loader.dataset) == 0:
        raise ValueError("test_loader is empty")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_mae": [],
        "test_rmse": [],
    }
    best_state = model.state_dict()
    best_rmse = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()

            batch_preds = []
            for sample_idx in range(x_b.size(0)):
                pred = model(x_b[sample_idx], edge_index)
                batch_preds.append(pred)
            y_hat = torch.stack(batch_preds, dim=0)

            loss = criterion(y_hat, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_b.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        test_loss, test_mae, test_rmse = evaluate_gcn(
            model,
            test_loader,
            edge_index,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_mae"].append(test_mae)
        history["test_rmse"].append(test_rmse)

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"test_loss={test_loss:.5f} | "
            f"test_mae={test_mae:.5f} | "
            f"test_rmse={test_rmse:.5f}"
        )

    model.load_state_dict(best_state)
    return model, history
