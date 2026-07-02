"""第十一周时空模型训练工具。"""

from __future__ import annotations

import copy

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise ImportError("Week11 training requires PyTorch. Please install torch first.") from exc


class SingleScaleGraphDataset(torch.utils.data.Dataset):
    """
    单尺度图时序数据集。

    x shape: [samples, seq_len, stations, features]
    y shape: [samples, stations]
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MultiScaleGraphDataset(torch.utils.data.Dataset):
    """
    多尺度图时序数据集。

    x_15min shape: [samples, seq_len_15min, stations, features]
    x_1h shape: [samples, seq_len_1h, stations, features]
    y shape: [samples, stations]
    """

    def __init__(self, x_15min: np.ndarray, x_1h: np.ndarray, y: np.ndarray):
        self.x_15min = torch.from_numpy(x_15min).float()
        self.x_1h = torch.from_numpy(x_1h).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_15min[idx], self.x_1h[idx], self.y[idx]


def _forward_batch(model, batch, edge_index, device):
    """
    统一处理单尺度与多尺度 batch。

    单尺度 batch:
        (x, y)
    多尺度 batch:
        (x_15min, x_1h, y)
    """
    if len(batch) == 2:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x, edge_index)
        return y_hat, y

    if len(batch) == 3:
        x_15min, x_1h, y = batch
        x_15min = x_15min.to(device)
        x_1h = x_1h.to(device)
        y = y.to(device)
        y_hat = model(x_15min, x_1h, edge_index)
        return y_hat, y

    raise ValueError(f"unsupported batch format with length={len(batch)}")


def evaluate_model(model, data_loader, edge_index, criterion, device, return_predictions: bool = False):
    """
    评估模型并计算 MAE / RMSE。

    Args:
        return_predictions: True 时额外返回预测值和真实值，供第十二周画预测曲线。
    """
    if len(data_loader.dataset) == 0:
        raise ValueError("data_loader is empty")

    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            y_hat, y = _forward_batch(model, batch, edge_index, device)
            loss = criterion(y_hat, y)
            total_loss += loss.item() * y.size(0)
            preds.append(y_hat.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    avg_loss = total_loss / len(data_loader.dataset)

    if return_predictions:
        return avg_loss, mae, rmse, preds, targets
    return avg_loss, mae, rmse


def train_model(
    model,
    train_loader,
    test_loader,
    edge_index,
    epochs: int,
    lr: float,
    device,
):
    """
    训练单尺度或多尺度时空模型。

    history 中保存：
    - train_loss: 训练集 MSE
    - test_loss: 测试集 MSE
    - test_mae: 测试集 MAE
    - test_rmse: 测试集 RMSE
    """
    if len(train_loader.dataset) == 0:
        raise ValueError("train_loader is empty")
    if len(test_loader.dataset) == 0:
        raise ValueError("test_loader is empty")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "test_loss": [], "test_mae": [], "test_rmse": []}
    best_state = copy.deepcopy(model.state_dict())
    best_rmse = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            y_hat, y = _forward_batch(model, batch, edge_index, device)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        test_loss, test_mae, test_rmse = evaluate_model(
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

        # 以测试集 RMSE 为准保存当前最优模型参数。
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"test_loss={test_loss:.5f} | "
            f"test_mae={test_mae:.5f} | "
            f"test_rmse={test_rmse:.5f}"
        )

    model.load_state_dict(best_state)
    return model, history
