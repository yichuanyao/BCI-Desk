"""脑电流录制与校验（与采集范式相同的 BrainFlow 取数路径）。

用于注入已知信号（如正弦波）后，录制固定时长并检查各通道波形是否正常。
不依赖范式标签，只连续累积 timestamp + 16 通道 µV 数据。
GUI 落盘前会应用顶部全局带通/陷波，具体参数见同目录 .meta.json。
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from ..config import NUM_CHANNELS, SAMPLE_RATE, UNIT_MICROVOLT

# 绘图时最多显示的点数（超出则等间隔降采样，减轻 pyqtgraph 负担）。
MAX_PLOT_POINTS = 2000


def uniform_time_s(n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """按采样率生成等间隔时间轴（秒），用于绘图。

    BrainFlow 板卡时间戳在每次 get_board_data() 批次内几乎相同、批次间有大间隙，
    直接用作横轴会把正常正弦波画成竖线锯齿；示波器页用的是样本序号轴。
    """
    return np.arange(n_samples, dtype=np.float64) / sample_rate


def downsample_for_plot(time_s: np.ndarray, y: np.ndarray, max_points: int = MAX_PLOT_POINTS):
    """等间隔降采样，仅用于显示，不改变落盘 CSV。

    Returns:
        (time_ds, y_ds, n_original, n_display)
    """
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(time_s)
    if n <= max_points or n == 0:
        return time_s, y, n, n
    step = int(np.ceil(n / max_points))
    return time_s[::step], y[::step], n, len(time_s[::step])


class RawStreamRecorder:
    """连续原始流录制器（无范式、无 event_type）。

    与 ParadigmRecorder 使用相同的 add_samples 接口，便于 GUI 复用同一套
    board.get_board_data() → timestamp_channel + eeg_channels 取数逻辑。
    """

    def __init__(self, num_channels: int = NUM_CHANNELS, sample_rate: int = SAMPLE_RATE):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self._ts_chunks: List[np.ndarray] = []
        self._eeg_chunks: List[np.ndarray] = []
        self._finished = False

    def add_samples(self, timestamps_ms: np.ndarray, eeg_uv: np.ndarray) -> None:
        if timestamps_ms is None or len(timestamps_ms) == 0:
            return
        ts = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
        eeg = np.asarray(eeg_uv, dtype=np.float64)
        if eeg.ndim != 2 or eeg.shape[0] != ts.shape[0]:
            raise ValueError(f"eeg_uv {eeg.shape} 与 timestamps {ts.shape} 不匹配。")
        self._ts_chunks.append(ts)
        self._eeg_chunks.append(eeg)

    @property
    def n_samples(self) -> int:
        return int(sum(len(c) for c in self._ts_chunks))

    def finish(
        self,
        output_path: str,
        duration_s: float,
        extra_meta: Optional[dict] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """写出连续原始 CSV + meta.json，返回统计供界面绘图。"""
        if self._finished:
            raise RuntimeError("recorder 已结束。")
        if self.n_samples == 0:
            raise RuntimeError("没有录到任何样本（检查板卡是否在采集）。")

        ts_all = np.concatenate(self._ts_chunks)
        eeg_all = np.concatenate(self._eeg_chunks, axis=0)

        eeg_names = [f"EEG Channel {i + 1}" for i in range(self.num_channels)]
        df = pd.DataFrame(eeg_all[:, : self.num_channels], columns=eeg_names)
        df.insert(0, "timestamp", ts_all)

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # 相对起始的等间隔时间轴（秒），供绘图；落盘 CSV 仍保留板卡 timestamp。
        time_s = uniform_time_s(len(ts_all), self.sample_rate)
        actual_duration = float(time_s[-1]) if len(time_s) > 1 else 0.0
        expected_samples = int(duration_s * self.sample_rate)

        per_ch = {}
        for i, name in enumerate(eeg_names):
            ch = eeg_all[:, i]
            per_ch[name] = {
                "min": float(np.min(ch)),
                "max": float(np.max(ch)),
                "std": float(np.std(ch)),
                "mean": float(np.mean(ch)),
            }

        meta = {
            "unit": UNIT_MICROVOLT,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "source": "validation_recorder",
            "duration_s_requested": duration_s,
            "duration_s_actual": round(actual_duration, 3),
            "n_samples": int(len(ts_all)),
            "expected_samples": expected_samples,
            "columns": ["timestamp"] + eeg_names,
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_path = os.path.splitext(output_path)[0] + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if log_fn:
            log_fn(f"已保存 {len(df)} 行 → {output_path}")
            log_fn(
                f"实际时长 {actual_duration:.2f}s, 样本 {len(ts_all)} "
                f"(期望约 {expected_samples} @ {self.sample_rate}Hz)"
            )

        self._finished = True
        return {
            "output_path": output_path,
            "meta_path": meta_path,
            "time_s": time_s,
            "eeg": eeg_all[:, : self.num_channels],
            "eeg_names": eeg_names,
            "per_channel": per_ch,
            "n_samples": int(len(ts_all)),
            "duration_s_actual": actual_duration,
            "unit": UNIT_MICROVOLT,
        }


def sample_rate_from_meta(csv_path: str, default: int = SAMPLE_RATE) -> int:
    """从同目录 .meta.json 读取 sample_rate，不存在则返回 default。"""
    meta_path = os.path.splitext(csv_path)[0] + ".meta.json"
    if not os.path.isfile(meta_path):
        return default
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("sample_rate", default))
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def load_validation_csv(csv_path: str, sample_rate: Optional[int] = None) -> dict:
    """读取校验 CSV，返回绘图用数组（横轴为通道采样率等间隔，不用 timestamp 列）。"""
    if sample_rate is None:
        sample_rate = sample_rate_from_meta(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    eeg_cols = [c for c in df.columns if c.startswith("EEG")]
    eeg = df[eeg_cols].values.astype(np.float64)
    time_s = uniform_time_s(len(df), sample_rate)
    return {
        "time_s": time_s,
        "eeg": eeg,
        "eeg_names": eeg_cols,
        "n_samples": len(df),
        "sample_rate": sample_rate,
    }
