"""
第九周数据处理模块。

目标：
1. 读取 PVODdatasets_v1.0 多站点光伏数据；
2. 按公共时间区间对齐多个站点；
3. 生成后续构图与建模所需的功率矩阵、特征张量与元数据。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PVOD_DIR = os.path.join(_BASE_DIR, "data", "PVODdatasets_v1.0")


DEFAULT_FEATURE_COLUMNS = [
    "power",
    "nwp_globalirrad",
    "nwp_directirrad",
    "nwp_temperature",
    "nwp_humidity",
    "nwp_windspeed",
]


@dataclass
class PVODAlignedData:
    """保存多站点对齐后的核心数据。"""

    station_ids: list[str]
    timestamps: pd.DatetimeIndex
    power_frame: pd.DataFrame
    feature_tensor: np.ndarray
    metadata: pd.DataFrame
    feature_columns: list[str]


def load_metadata(metadata_path: str | None = None) -> pd.DataFrame:
    """
    读取站点元数据。

    返回的 DataFrame 至少包含：
    - Station_ID
    - Longitude
    - Latitude
    """
    path = metadata_path or os.path.join(_PVOD_DIR, "metadata.csv")
    metadata = pd.read_csv(path, encoding="utf-8-sig")
    metadata["Longitude"] = metadata["Longitude"].astype(float)
    metadata["Latitude"] = metadata["Latitude"].astype(float)
    metadata["Capacity"] = metadata["Capacity"].astype(float)
    return metadata


def load_station_csv(
    station_id: str,
    pvod_dir: str | None = None,
) -> pd.DataFrame:
    """
    读取单个站点文件，并按时间排序。

    输出列中保留原始数值特征，索引设为 date_time。
    """
    root = pvod_dir or _PVOD_DIR
    path = os.path.join(root, f"{station_id}.csv")
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").set_index("date_time")
    return df


def load_all_stations(
    station_ids: list[str] | None = None,
    pvod_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    批量读取所有站点文件。
    """
    root = pvod_dir or _PVOD_DIR
    if station_ids is None:
        station_ids = [
            os.path.splitext(name)[0]
            for name in os.listdir(root)
            if name.startswith("station") and name.endswith(".csv")
        ]
        station_ids = sorted(station_ids)

    station_frames = {}
    for station_id in station_ids:
        station_frames[station_id] = load_station_csv(station_id, pvod_dir=root)
    return station_frames


def align_station_frames(
    station_frames: dict[str, pd.DataFrame],
    feature_columns: list[str] | None = None,
    dropna: bool = True,
) -> PVODAlignedData:
    """
    不同站点观测数据的起止时间不一致。
    对齐多个站点的时间索引，并抽取统一特征。

    Args:
        station_frames: {station_id: DataFrame}
        feature_columns: 需要保留的特征列
        dropna: True 时删除任一站点任一特征缺失的时间步

    Returns:
        PVODAlignedData
    """
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    station_ids = sorted(station_frames.keys())
    if not station_ids:
        raise ValueError("station_frames is empty")

    # 取所有站点时间索引的交集，确保后续多站点建模可以逐时对齐。
    common_index = None
    for station_id in station_ids:
        idx = station_frames[station_id].index
        common_index = idx if common_index is None else common_index.intersection(idx)
    common_index = common_index.sort_values()
    if len(common_index) == 0:
        raise ValueError("no common timestamps across stations")

    aligned_frames = {}
    for station_id in station_ids:
        missing_columns = [col for col in feature_columns if col not in station_frames[station_id].columns]
        if missing_columns:
            raise KeyError(f"{station_id} is missing columns: {missing_columns}")
        frame = station_frames[station_id].loc[common_index, feature_columns].copy()
        aligned_frames[station_id] = frame

    if dropna:
        valid_mask = pd.Series(True, index=common_index)
        for station_id in station_ids:
            valid_mask &= ~aligned_frames[station_id].isna().any(axis=1)
        common_index = common_index[valid_mask.values]
        for station_id in station_ids:
            aligned_frames[station_id] = aligned_frames[station_id].loc[common_index]
        if len(common_index) == 0:
            raise ValueError("all aligned rows were removed after dropna filtering")

    # 构造功率矩阵：[time, stations]
    power_frame = pd.DataFrame(
        {station_id: aligned_frames[station_id]["power"] for station_id in station_ids},
        index=common_index,
    )

    # 构造特征张量：[time, stations, features]
    feature_tensor = np.stack(
        [aligned_frames[station_id][feature_columns].to_numpy(dtype=np.float32) for station_id in station_ids],
        axis=1,
    ).astype(np.float32)

    metadata = load_metadata()
    metadata = metadata[metadata["Station_ID"].isin(station_ids)].copy()
    metadata = metadata.set_index("Station_ID").loc[station_ids].reset_index()

    return PVODAlignedData(
        station_ids=station_ids,
        timestamps=common_index,
        power_frame=power_frame,
        feature_tensor=feature_tensor,
        metadata=metadata,
        feature_columns=feature_columns,
    )


def load_aligned_pvod_data(
    station_ids: list[str] | None = None,
    feature_columns: list[str] | None = None,
    pvod_dir: str | None = None,
) -> PVODAlignedData:
    """
    读取并对齐 PVOD 多站点数据。
    """
    frames = load_all_stations(station_ids=station_ids, pvod_dir=pvod_dir)
    return align_station_frames(frames, feature_columns=feature_columns)
