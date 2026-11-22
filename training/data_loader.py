"""从本地定稿格式的带标签 CSV 加载脑电试次（trials）。

输入契约（本地采集范式 / 离线对齐产出）：
    CSV 列：timestamp, event_type, EEG Channel 1..16
    旁车 <csv>.meta.json（可选）：{"unit": "...", "sample_rate": 500, ...}

切试次规则（与本地 bci_desk/data/epoching.py 一致）：
    连续相同 event_type 视为一个试次（trial）。

输出：
    trials —— list[np.ndarray]，每个形状 (num_channels, n_samples_trial)，已换算成 µV。
    labels —— np.ndarray[int]，每个试次的类别索引（基于传入的 classes 或默认 DEFAULT_CLASSES）。
    sample_rates —— list[int]，每个试次对应的原始采样率（来自各自 meta）。
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_CLASSES, NUM_CHANNELS, SAMPLE_RATE, UNIT_RAW_COUNT, gain_for_unit


def _meta_for_csv(csv_path: str) -> dict:
    """读取同名 .meta.json；不存在则返回空 dict。"""
    meta_path = os.path.splitext(csv_path)[0] + ".meta.json"
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def infer_classes_from_meta(data_path: str) -> Optional[List[str]]:
    """从数据目录下 CSV 旁车 meta 的 ``classes`` 字段推断类别。

    若多个 CSV 的 classes 不一致则抛出 ValueError。
    若均未找到 classes 则返回 None。
    """
    files = list_csv_files(data_path)
    inferred: Optional[List[str]] = None
    for fp in files:
        meta = _meta_for_csv(fp)
        raw = meta.get("classes")
        if not raw:
            continue
        cls = [str(c) for c in raw]
        if inferred is None:
            inferred = cls
        elif cls != inferred:
            raise ValueError(
                f"数据目录下 meta.classes 不一致：{inferred!r} vs {cls!r}（来自 {fp}）。"
                "请用 --classes 或 --paradigm 显式指定。"
            )
    return inferred


def resolve_classes(
    data_path: str,
    *,
    classes: Optional[List[str]] = None,
    paradigm_id: Optional[str] = None,
) -> List[str]:
    """确定训练/评估使用的类别列表。

    优先级：显式 classes > paradigm_id 预设 > meta 推断 > config 默认。
    """
    if classes is not None:
        return list(classes)
    if paradigm_id is not None:
        from config import classes_for_paradigm

        return classes_for_paradigm(paradigm_id)
    inferred = infer_classes_from_meta(data_path)
    if inferred is not None:
        return inferred
    return list(DEFAULT_CLASSES)


def list_csv_files(data_path: str) -> List[str]:
    """把 data_path 解析成 CSV 文件列表（接受单个文件或目录）。

    目录会递归查找（含 session 子目录），便于直接指向某个 session 目录或其父目录。
    每个 CSV 的试次彼此独立加载，互不拼接。
    """
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))
    else:
        files = [data_path]
    if not files:
        raise FileNotFoundError(f"在 {data_path} 未找到任何 CSV 文件。")
    return files


def load_trials_from_csv(
    csv_path: str,
    classes: List[str] = None,
    num_channels: int = NUM_CHANNELS,
) -> Tuple[List[np.ndarray], List[int], int]:
    """从单个 CSV 加载试次。

    Returns:
        (trials, labels, sample_rate)：
            trials —— list of (num_channels, T)，单位 µV。
            labels —— list[int]，类别索引。
            sample_rate —— 该文件采样率（来自 meta，缺失则用默认 125）。
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    meta = _meta_for_csv(csv_path)
    unit = meta.get("unit", UNIT_RAW_COUNT)   # 无 meta 时按旧 raw_count 处理
    sample_rate = int(meta.get("sample_rate", SAMPLE_RATE))
    gain = gain_for_unit(unit)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "event_type" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 'event_type' 列。")
    eeg_cols = [c for c in df.columns if c.startswith("EEG")]
    if len(eeg_cols) < num_channels:
        raise ValueError(
            f"{csv_path} 只有 {len(eeg_cols)} 个 EEG 通道列，少于要求的 {num_channels}。"
        )
    eeg_cols = eeg_cols[:num_channels]

    # 连续相同 event_type 归为一个试次。
    ev = df["event_type"]
    block_id = ev.ne(ev.shift()).cumsum()

    trials: List[np.ndarray] = []
    labels: List[int] = []
    for _, seg in df.groupby(block_id):
        label = seg["event_type"].iloc[0]
        if label not in classes:
            continue
        data = seg[eeg_cols].values.T.astype(np.float64) * gain  # (C, T) µV
        trials.append(data)
        labels.append(classes.index(label))

    return trials, labels, sample_rate


def load_dataset(
    data_path: str,
    classes: List[str] = None,
    num_channels: int = NUM_CHANNELS,
) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    """加载一个文件或整个目录下所有 CSV 的试次，拼成数据集。

    Returns:
        (trials, labels, sample_rates)：
            trials —— list of (C, T)，µV。
            labels —— np.ndarray[int]。
            sample_rates —— list[int]，与 trials 一一对应。
    """
    if classes is None:
        classes = DEFAULT_CLASSES
    files = list_csv_files(data_path)

    all_trials: List[np.ndarray] = []
    all_labels: List[int] = []
    all_srates: List[int] = []
    for fp in files:
        trials, labels, srate = load_trials_from_csv(fp, classes, num_channels)
        all_trials.extend(trials)
        all_labels.extend(labels)
        all_srates.extend([srate] * len(trials))

    if not all_trials:
        raise RuntimeError(
            f"未从 {data_path} 加载到任何试次。请检查 event_type 是否在 {classes} 中。"
        )
    return all_trials, np.asarray(all_labels, dtype=np.int64), all_srates
