"""离线对齐：PsychoPy 行为日志 + OpenBCI 原始 txt → 带标签 CSV。

干净重写自 origin/EEGali4.py，仅依赖 numpy/pandas。运行时不依赖 origin。

用途：把"旧流程"采集的数据（OpenBCI GUI 录的原始 txt + PsychoPy 提示日志）
按时间戳对齐，切出每个试次执行期的脑电，生成与 Extracted_*.csv 相同格式的带标签 CSV：

    timestamp, event_type, EEG Channel 1, ..., EEG Channel 16

之后即可被 bci_desk.data.epoching.load_epochs_from_csv 加载。

注意：本对齐流程产出的 EEG 是 OpenBCI txt 里的原始 ADC 计数（raw_count），
后续预处理需按 UNIT_RAW_COUNT 处理。
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..config import CONDITION_TO_LABEL, NUM_CHANNELS

# 默认执行期时长（秒）：与 MI.py 的 action_duration 一致。
DEFAULT_ACTION_DURATION_S = 3.0
# 标记"动作执行"阶段的事件名（来自 PsychoPy 日志的 event_name 列）。
ACTION_EVENT_NAME = "action_execute"


def check_file_exists(file_path: str, file_desc: str) -> None:
    """文件不存在时抛出带中文描述的 FileNotFoundError。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到{file_desc}文件：{file_path}")


def load_events(
    log_path: str,
    action_duration_s: float = DEFAULT_ACTION_DURATION_S,
    condition_to_label: Optional[dict] = None,
) -> pd.DataFrame:
    """读取 PsychoPy 行为日志，提取每个试次执行期的 [start, end] 时间窗与标签。

    Args:
        log_path: PsychoPy 日志 CSV 路径（列含 event_name / condition / unix_timestamp_ms）。
        action_duration_s: 执行期时长（秒），窗口 end = start + 该值。
        condition_to_label: condition→标签 映射；默认用 config.CONDITION_TO_LABEL。

    Returns:
        DataFrame，列：trial_id / cond(标签) / start(ms) / end(ms)。
    """
    if condition_to_label is None:
        condition_to_label = CONDITION_TO_LABEL

    log_df = pd.read_csv(log_path)
    action_rows = log_df[log_df["event_name"] == ACTION_EVENT_NAME].reset_index(drop=True)

    events = []
    for idx, row in action_rows.iterrows():
        orig_cond = row["condition"]
        events.append(
            {
                "trial_id": idx,
                "cond": condition_to_label.get(orig_cond, orig_cond),
                "start": row["unix_timestamp_ms"],
                "end": row["unix_timestamp_ms"] + action_duration_s * 1000.0,
            }
        )
    return pd.DataFrame(events)


def load_eeg(eeg_path: str, num_channels: int = NUM_CHANNELS):
    """读取 OpenBCI 原始 txt，返回 (eeg_df, eeg_cols)。

    自动跳过以 '%' 开头的注释行、自动探测分隔符。按 OpenBCI 列布局硬编码：
    第 1..num_channels 列为 EEG 通道，第 31 列（索引 30）为 Unix 时间戳（秒）。

    Args:
        eeg_path: OpenBCI raw txt 路径。
        num_channels: EEG 通道数（默认 16）。

    Returns:
        (df, eeg_cols):
            df —— 含 'timestamp_ms' 列 + 各 EEG 原始列。
            eeg_cols —— EEG 原始列名列表（长度 num_channels）。
    """
    # 探测分隔符（取前几行中出现最多的候选符号）。
    with open(eeg_path, "r", encoding="utf-8", errors="ignore") as f:
        head_lines = [f.readline() for _ in range(5)]
        delimiter = max([",", "\t", ";"], key=lambda x: sum(line.count(x) for line in head_lines))

    # 统计开头的注释行数（以 '%' 开头）。
    comment_rows = 0
    with open(eeg_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("%"):
                comment_rows += 1
            else:
                break

    df = pd.read_csv(
        eeg_path,
        skiprows=comment_rows,
        sep=delimiter,
        engine="python",
        on_bad_lines="skip",
        encoding_errors="ignore",
    )

    # OpenBCI 列布局：[0]=Sample Index, [1..num_channels]=EXG, [30]=Timestamp(秒)。
    eeg_cols = df.columns[1:1 + num_channels].tolist()
    ts_col = df.columns[30]

    df["timestamp_ms"] = pd.to_numeric(df[ts_col], errors="coerce") * 1000
    df = df.dropna(subset=["timestamp_ms"]).round(3)

    return df[["timestamp_ms"] + eeg_cols], eeg_cols


def extract_data(events_df: pd.DataFrame, eeg_df: pd.DataFrame) -> pd.DataFrame:
    """按事件时间窗从脑电数据中切出各试次片段，附上 event_type 标签。

    Args:
        events_df: load_events 的输出（含 start/end/cond）。
        eeg_df: load_eeg 的输出（含 timestamp_ms）。

    Returns:
        拼接后的 DataFrame；若无任何片段匹配则返回空 DataFrame。
    """
    extracted = []
    for _, trial in events_df.iterrows():
        mask = (eeg_df["timestamp_ms"] >= trial["start"]) & (eeg_df["timestamp_ms"] <= trial["end"])
        data_slice = eeg_df[mask].copy()
        if not data_slice.empty:
            data_slice["event_type"] = trial["cond"]
            extracted.append(data_slice)

    if not extracted:
        return pd.DataFrame()
    return pd.concat(extracted, ignore_index=True)


def save_extracted(
    events_df: pd.DataFrame,
    eeg_df: pd.DataFrame,
    eeg_cols: list,
    output_path: str,
    num_channels: int = NUM_CHANNELS,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """从事件窗切片、重命名为标准列、保存为带标签 CSV，并返回统计。

    被 `align_to_csv`（离线对齐）和采集范式录制（在线打标签）共用，保证两者输出格式完全一致：
        timestamp, event_type, EEG Channel 1..num_channels

    Args:
        events_df: 含 start/end/cond 的事件表。
        eeg_df: 含 timestamp_ms + 各原始通道列的脑电表。
        eeg_cols: eeg_df 中原始通道列名（长度 num_channels），将被重命名为 EEG Channel i。
        output_path: 输出 CSV 路径。
        num_channels: 通道数。
        log_fn: 可选日志回调。

    Returns:
        dict：{output_path, n_trials, n_rows, label_counts}。

    Raises:
        RuntimeError: 没有任何试次窗口匹配到脑电数据。
    """
    result_df = extract_data(events_df, eeg_df)
    if result_df.empty:
        raise RuntimeError("没有任何试次的时间窗匹配到脑电数据（检查两端时间戳是否同一时段）。")

    new_eeg_names = [f"EEG Channel {i + 1}" for i in range(num_channels)]
    result_df = result_df.rename(columns=dict(zip(eeg_cols, new_eeg_names)))
    result_df = result_df.rename(columns={"timestamp_ms": "timestamp"})
    result_df = result_df[["timestamp", "event_type"] + new_eeg_names]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    label_counts = result_df["event_type"].value_counts().to_dict()
    if log_fn is not None:
        log_fn(f"完成：写入 {len(result_df)} 行 → {output_path}")

    return {
        "output_path": output_path,
        "n_trials": int(len(events_df)),
        "n_rows": int(len(result_df)),
        "label_counts": label_counts,
    }


def align_to_csv(
    psychopy_log: str,
    openbci_raw: str,
    output_path: str,
    action_duration_s: float = DEFAULT_ACTION_DURATION_S,
    num_channels: int = NUM_CHANNELS,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict:
    """一键对齐：行为日志 + 原始 txt → 带标签 CSV，并返回统计信息。

    Args:
        psychopy_log: PsychoPy 行为日志 CSV 路径。
        openbci_raw: OpenBCI 原始 txt 路径。
        output_path: 输出带标签 CSV 路径。
        action_duration_s: 执行期时长（秒）。
        num_channels: EEG 通道数。
        log_fn: 可选的日志回调（用于 GUI 实时显示进度）；默认无输出。

    Returns:
        dict：{output_path, n_trials, n_rows, label_counts(各标签片段数)}。

    Raises:
        FileNotFoundError: 输入文件不存在。
        RuntimeError: 没有任何试次成功对齐。
    """
    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    check_file_exists(psychopy_log, "PsychoPy 行为日志")
    check_file_exists(openbci_raw, "OpenBCI 原始数据")

    _log("读取行为日志事件...")
    events_df = load_events(psychopy_log, action_duration_s=action_duration_s)
    _log(f"共 {len(events_df)} 个执行期试次。")

    _log("读取 OpenBCI 原始脑电...")
    eeg_df, orig_eeg_cols = load_eeg(openbci_raw, num_channels=num_channels)
    _log(f"脑电样本点 {len(eeg_df)} 行，通道 {len(orig_eeg_cols)}。")

    _log("按时间戳对齐切片...")
    try:
        return save_extracted(events_df, eeg_df, orig_eeg_cols, output_path, num_channels, log_fn=_log)
    except RuntimeError:
        raise RuntimeError("对齐失败：没有任何试次的时间窗匹配到脑电数据（检查两端时间戳是否同一时段）。")
