"""采集范式的录制状态机（Qt 无关，可无硬件测试）。

ParadigmRecorder 负责：
    1. 持有时间表（schedule），按外部喂入的时钟推进当前阶段；
    2. 累积外部喂入的脑电样本（时间戳 + 各通道，单位微伏）；
    3. 在每个需打标签的阶段（MI 标签期/休息/放松/段间休息等）记录 [start, end] 事件窗；
    4. 结束时把样本按事件窗切片，写出与离线对齐**完全相同格式**的带标签 CSV。

GUI 只需：定时把 board 新数据喂给 add_samples，把当前时钟喂给 update 取回当前阶段用于显示；
结束时调用 finish 落盘。无硬件时也可用合成数据按相同接口驱动，便于测试。

注意：本录制路径的数据来自 BrainFlow，已是微伏（UNIT_MICROVOLT），
finish 时会额外写一个 `<csv>.meta.json` 记录单位，供训练/推理两端对齐。
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import NUM_CHANNELS, SAMPLE_RATE, UNIT_MICROVOLT
from ..data.aligner import save_extracted
from .paradigm import Phase


class ParadigmRecorder:
    """范式录制状态机。

    Args:
        schedule: paradigm.build_schedule 产生的阶段列表。
        num_channels: 录制的 EEG 通道数。
        sample_rate: 采样率（写入 meta，便于训练/推理）。
    """

    def __init__(self, schedule: List[Phase], num_channels: int = NUM_CHANNELS, sample_rate: int = SAMPLE_RATE):
        self.schedule = schedule
        self.num_channels = num_channels
        self.sample_rate = sample_rate

        self._ts_chunks: List[np.ndarray] = []      # 每次喂入的时间戳(ms)片段
        self._eeg_chunks: List[np.ndarray] = []      # 每次喂入的 (n, num_channels) 片段
        self.events: List[dict] = []                  # 每个打标签阶段一个 {cond,label,start,end}

        self._idx = -1
        self._phase_start_ms: Optional[float] = None
        self._started = False
        self._finished = False

    # ---------- 生命周期 ----------
    def start(self, now_ms: float) -> Optional[Phase]:
        """开始范式，进入第 0 个阶段。返回当前阶段（无阶段时返回 None）。"""
        self._started = True
        self._idx = 0
        self._phase_start_ms = now_ms
        if self.schedule:
            self._on_enter_phase(now_ms)
            return self.schedule[0]
        return None

    def update(self, now_ms: float) -> Tuple[bool, Optional[Phase], float]:
        """按当前时钟推进阶段。

        Returns:
            (done, current_phase, remaining_s):
                done —— 全部阶段是否结束。
                current_phase —— 当前阶段（结束后为 None）。
                remaining_s —— 当前阶段剩余秒数（结束后为 0）。
        """
        if not self._started or self._idx < 0 or self._idx >= len(self.schedule):
            return True, None, 0.0

        cur = self.schedule[self._idx]
        if cur.is_manual_wait:
            return False, cur, 0.0

        # 若 UI 卡顿，一次 tick 内补推进所有已到期阶段，避免界面与录制器脱节。
        while True:
            elapsed_ms = now_ms - self._phase_start_ms
            if elapsed_ms < cur.duration_s * 1000.0:
                break
            self._idx += 1
            self._phase_start_ms = now_ms
            if self._idx >= len(self.schedule):
                return True, None, 0.0
            self._on_enter_phase(now_ms)
            cur = self.schedule[self._idx]
            if cur.is_manual_wait:
                return False, cur, 0.0

        elapsed_ms = now_ms - self._phase_start_ms
        remaining_s = max(0.0, cur.duration_s - elapsed_ms / 1000.0)
        return False, cur, remaining_s

    @property
    def phase_index(self) -> int:
        """当前阶段在 schedule 中的下标（用于界面刷新去重）。"""
        return self._idx

    def confirm_manual_wait(self, now_ms: float) -> Tuple[bool, Optional[Phase], float]:
        """在 manual_wait 阶段由操作者确认（如按空格）后进入下一阶段。"""
        if not self._started or self._idx < 0 or self._idx >= len(self.schedule):
            return True, None, 0.0

        cur = self.schedule[self._idx]
        if not cur.is_manual_wait:
            return self.update(now_ms)

        if self.events and self.events[-1].get("end") is None:
            self.events[-1]["end"] = now_ms

        self._idx += 1
        self._phase_start_ms = now_ms
        if self._idx >= len(self.schedule):
            return True, None, 0.0
        self._on_enter_phase(now_ms)
        nxt = self.schedule[self._idx]
        remaining_s = 0.0 if nxt.is_manual_wait else max(0.0, nxt.duration_s)
        return False, nxt, remaining_s

    @property
    def current_phase(self) -> Optional[Phase]:
        if not self._started or self._idx < 0 or self._idx >= len(self.schedule):
            return None
        return self.schedule[self._idx]

    def _on_enter_phase(self, now_ms: float) -> None:
        """进入新阶段时的回调：登记打标签事件窗。"""
        phase = self.schedule[self._idx]
        event_type = phase.event_type
        if not event_type and phase.is_execute and phase.label:
            event_type = phase.label
        if not event_type:
            return
        end_ms = None if phase.is_manual_wait else now_ms + phase.duration_s * 1000.0
        self.events.append(
            {
                "cond": event_type,
                "label": phase.label or event_type,
                "start": now_ms,
                "end": end_ms,
            }
        )

    def _close_open_events(self, now_ms: float) -> None:
        for ev in self.events:
            if ev.get("end") is None:
                ev["end"] = now_ms

    # ---------- 数据喂入 ----------
    def add_samples(self, timestamps_ms: np.ndarray, eeg_uv: np.ndarray) -> None:
        """累积一批样本。

        Args:
            timestamps_ms: 形状 (n,) 的时间戳（毫秒，与阶段时钟同一时基）。
            eeg_uv: 形状 (n, num_channels) 的脑电（微伏）。
        """
        if timestamps_ms is None or len(timestamps_ms) == 0:
            return
        ts = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
        eeg = np.asarray(eeg_uv, dtype=np.float64)
        if eeg.ndim != 2 or eeg.shape[0] != ts.shape[0]:
            raise ValueError(
                f"eeg_uv 形状 {eeg.shape} 与 timestamps {ts.shape} 不匹配，应为 (n, num_channels)。"
            )
        self._ts_chunks.append(ts)
        self._eeg_chunks.append(eeg)

    # ---------- 结束落盘 ----------
    @property
    def n_samples(self) -> int:
        return int(sum(len(c) for c in self._ts_chunks))

    def finish(self, output_path: str, extra_meta: Optional[dict] = None, log_fn=None) -> dict:
        """把累积样本按事件窗切片，写出带标签 CSV + meta.json，返回统计。

        Raises:
            RuntimeError: 没有录到样本或没有任何事件窗匹配到数据。
        """
        if self._finished:
            raise RuntimeError("recorder 已结束。")
        if not self.events:
            raise RuntimeError("没有任何打标签事件被记录（范式可能未正常启动）。")
        if self.n_samples == 0:
            raise RuntimeError("没有录到任何脑电样本（检查板卡是否在采集）。")

        ts_all = np.concatenate(self._ts_chunks)
        self._close_open_events(float(ts_all[-1]) if len(ts_all) else 0.0)
        eeg_all = np.concatenate(self._eeg_chunks, axis=0)

        # 组装 eeg_df（timestamp_ms + 原始通道列），交给与离线对齐共用的 save_extracted。
        raw_cols = [f"ch{i}" for i in range(self.num_channels)]
        eeg_df = pd.DataFrame(eeg_all[:, : self.num_channels], columns=raw_cols)
        eeg_df.insert(0, "timestamp_ms", ts_all)

        events_df = pd.DataFrame(self.events)[["cond", "start", "end"]]

        stats = save_extracted(events_df, eeg_df, raw_cols, output_path, self.num_channels, log_fn=log_fn)

        # 写单位 meta，标记本数据为微伏。
        meta = {
            "unit": UNIT_MICROVOLT,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "source": "paradigm_recorder",
            "columns": ["timestamp", "event_type"] + [f"EEG Channel {i + 1}" for i in range(self.num_channels)],
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_path = os.path.splitext(output_path)[0] + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        stats["meta_path"] = meta_path
        stats["unit"] = UNIT_MICROVOLT

        self._finished = True
        return stats
