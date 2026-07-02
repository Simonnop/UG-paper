"""第十二周最终评估与可视化工具。"""

from __future__ import annotations

import contextlib

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as exc:
    raise ImportError("Week12 visualization requires matplotlib and seaborn.") from exc

try:
    import torch
except Exception as exc:
    raise ImportError("Week12 spatial influence visualization requires PyTorch.") from exc


def summarize_histories(histories: dict[str, dict]) -> list[dict[str, float | str]]:
    """
    将多个模型的训练历史汇总成结果表。

    输入:
        histories: {model_name: history}
    输出:
        按 best_test_rmse 升序排列的结果列表。
    """
    rows = []
    for model_name, history in histories.items():
        rows.append(
            {
                "model": model_name,
                "best_test_mae": float(min(history["test_mae"])),
                "best_test_rmse": float(min(history["test_rmse"])),
                "best_test_loss": float(min(history["test_loss"])),
            }
        )
    return sorted(rows, key=lambda row: row["best_test_rmse"])


def plot_training_curves(histories: dict[str, dict]):
    """绘制多个模型的测试集 RMSE 曲线。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    for model_name, history in histories.items():
        ax.plot(history["test_rmse"], label=model_name)
    ax.set_title("Test RMSE Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.show()


def plot_station_prediction(
    preds: np.ndarray,
    targets: np.ndarray,
    station_ids: list[str],
    station_index: int = 0,
    steps: int = 96,
):
    """
    绘制单个站点的预测值与真实值曲线。

    preds / targets shape: [samples, stations]
    """
    steps = min(steps, len(preds))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(targets[:steps, station_index], label="Actual")
    ax.plot(preds[:steps, station_index], label="Predicted")
    ax.set_title(f"Prediction Curve - {station_ids[station_index]}")
    ax.set_xlabel("Test Step")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.show()


def plot_matrix(matrix: np.ndarray, station_ids: list[str], title: str, cmap: str = "viridis"):
    """绘制站点到站点的矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        xticklabels=station_ids,
        yticklabels=station_ids,
        cmap=cmap,
        annot=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Station")
    ax.set_ylabel("Station")
    plt.show()


def compute_spatial_influence_matrix(model, x_sample: np.ndarray, edge_index, device) -> np.ndarray:
    """
    使用输入梯度估计空间影响矩阵。

    influence[target_station, source_station] 表示目标站点输出
    对源站点输入序列的梯度强度。该矩阵是解释性诊断结果，
    不是模型显式学习到的 attention 参数。

    x_sample shape: [1, seq_len, stations, features]
    """
    was_training = model.training
    model.eval()

    x = torch.from_numpy(x_sample).float().to(device)
    x.requires_grad_(True)

    # cuDNN 的 RNN 在 eval 模式下不支持 backward。
    # 这里仅为了计算输入梯度，临时关闭 cuDNN RNN 路径，不改变外部模型状态。
    cudnn_context = (
        torch.backends.cudnn.flags(enabled=False)
        if x.is_cuda and torch.backends.cudnn.is_available()
        else contextlib.nullcontext()
    )

    try:
        with cudnn_context:
            y_hat = model(x, edge_index)  # [1, stations]
            num_stations = y_hat.shape[1]
            influence = []

            for target_idx in range(num_stations):
                model.zero_grad(set_to_none=True)
                if x.grad is not None:
                    x.grad.zero_()
                y_hat[0, target_idx].backward(retain_graph=True)

                # 对时间维和特征维求和，得到每个源站点的总体影响强度。
                grad = x.grad.detach().abs().sum(dim=(0, 1, 3)).cpu().numpy()
                influence.append(grad)
    finally:
        model.train(was_training)

    influence = np.stack(influence, axis=0)
    row_sum = influence.sum(axis=1, keepdims=True)
    return np.divide(influence, row_sum, out=np.zeros_like(influence), where=row_sum > 0)
