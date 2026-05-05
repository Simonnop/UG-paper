
from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_odd_window(window: int) -> int:
    """Return an odd window size for centered moving-average smoothing."""
    if window < 3:
        raise ValueError("window must be >= 3")
    return window if window % 2 == 1 else window + 1


def moving_average_trend(series: np.ndarray, window: int = 25) -> np.ndarray:
    """
    Estimate the low-frequency trend with a centered moving average.

    Args:
        series: 1D load sequence.
        window: Smoothing window. The function upgrades even windows to odd
            windows to keep the average centered.

    Returns:
        Trend component with the same length as the input series.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series must be 1D")

    window = ensure_odd_window(window)
    # Repeat edge values so the output length matches the original series.
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    trend = np.convolve(padded, kernel, mode="valid")
    return trend


# def seasonal_pattern_from_detrended(
#     detrended: np.ndarray,
#     period: int = 24,
# ) -> np.ndarray:
#     """
#     Compute a reusable periodic template from a detrended series.

#     For hourly load data, period=24 corresponds to the daily pattern.
#     """
#     if period < 2:
#         raise ValueError("period must be >= 2")

#     detrended = np.asarray(detrended, dtype=np.float64)
#     pattern = np.zeros(period, dtype=np.float64)

#     for offset in range(period):
#         slot_values = detrended[offset::period]
#         pattern[offset] = slot_values.mean()

#     # Center the periodic pattern so the long-run bias stays in the trend term.
#     pattern -= pattern.mean()
#     return pattern


def decompose_series(
    series: np.ndarray,
    period: int = 24,
    trend_window: int = 25,
) -> dict[str, np.ndarray]:
    """
    Decompose a load sequence into trend, periodic and residual components.

    Returns:
        Dictionary with keys: observed, trend, periodic
    """
    observed = np.asarray(series, dtype=np.float64)
    trend = moving_average_trend(observed, window=trend_window)
    # "二分解"模式，周期项为原始负荷减去趋势项
    periodic = observed - trend

    return {
        "observed": observed,
        "trend": trend,
        "periodic": periodic,
    }


def select_representative_users(
    df: pd.DataFrame,
    user_cols: list[str],
    n_users: int = 3,
) -> list[str]:
    """
    Pick a few users with relatively clear variations for visual inspection.
    """
    if n_users < 1:
        raise ValueError("n_users must be >= 1")

    variability = (
        df[user_cols]
        .astype(float)
        .std(axis=0)
        .sort_values(ascending=False)
    )
    return variability.head(n_users).index.tolist()


def decomposition_summary(components: dict[str, np.ndarray]) -> dict[str, float]:
    """Return lightweight summary metrics for the decomposition result."""
    observed = components["observed"]
    trend = components["trend"]
    periodic = components["periodic"]
    reconstructed = trend + periodic
    return {
        "observed_var": float(np.var(observed)),
        "trend_var": float(np.var(trend)),
        "periodic_var": float(np.var(periodic)),
        "reconstruction_mae": float(np.mean(np.abs(observed - reconstructed))),
    }
