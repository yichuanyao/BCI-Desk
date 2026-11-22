"""ERD/ERS（事件相关去同步 / 同步）运动执行实验的纯逻辑模块。

与 Qt / brainflow 解耦，只依赖 numpy，便于离线复算与单元测试。
GUI 层（main.py「ERD/ERS 检测」标签页）负责板卡取数、范式显示与绘图，
本模块负责：范式时序定义、频带功率（Welch PSD）、ERD/ERS 计算、结果 CSV schema。

实验范式（单个 Trial，运动执行 / 真实握拳）：
    1. Baseline 静息  2.0s —— 记录基线 EEG（ERD/ERS 的参考功率）
    2. Ready  准备    1.0s —— 提示准备，不参与分析
    3. Task   任务    4.0s —— 随机左/右手持续握拳；分析窗 = cue 后 0.5~4.0s（算 ERD）
    4. Relax  放松    2.0s —— 整段用于算 ERS（Beta 反弹）
    5. ITI            1.0s —— 试次间隔，进入下一 Trial

频带：Mu 8~13 Hz、Beta 13~30 Hz。
功率：Welch PSD 在频带内积分（与 8~30Hz 带通 + Hilbert 等价的可离线替换方案）。

ERD = (TaskPower  − BaselinePower) / BaselinePower × 100%   （负值 = ERD）
ERS = (RelaxPower − BaselinePower) / BaselinePower × 100%   （正值 = ERS）
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------- 方向（左右手） ----------------
DIR_LEFT = "LEFT"
DIR_RIGHT = "RIGHT"
DIRECTION_NAMES_CN = {DIR_LEFT: "左手", DIR_RIGHT: "右手"}

# ---------------- 阶段类型 ----------------
PHASE_BASELINE = "baseline"
PHASE_READY = "ready"
PHASE_MI = "mi"
PHASE_RELAX = "relax"
PHASE_ITI = "iti"

# 每个阶段写入原始数据的 marker（便于离线按事件切片）。
PHASE_MARKERS = {
    PHASE_BASELINE: 10,
    PHASE_READY: 20,
    PHASE_MI: 0,        # MI 期的 marker 由方向覆盖（见 MI_DIRECTION_MARKERS）
    PHASE_RELAX: 40,
    PHASE_ITI: 0,
}
MI_DIRECTION_MARKERS = {DIR_LEFT: 31, DIR_RIGHT: 32}

# ---------------- 频带 ----------------
MU_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# trial_results.csv 字段顺序（与最终输出规格一致）。
TRIAL_RESULTS_COLUMNS = [
    "Trial",
    "Direction",
    "Baseline Mu Power",
    "Baseline Beta Power",
    "Task Mu Power",
    "Task Beta Power",
    "Relax Mu Power",
    "Relax Beta Power",
    "ERD Mu",
    "ERD Beta",
    "ERS Mu",
    "ERS Beta",
]


@dataclass
class ErdErsTiming:
    """单试次各阶段时长（秒）与 MI 分析窗延迟。"""
    baseline_s: float = 2.0
    ready_s: float = 1.0
    mi_s: float = 4.0
    relax_s: float = 2.0
    iti_s: float = 1.0
    mi_analysis_delay_s: float = 0.5   # cue 后延迟，避开视觉诱发与起始暂态

    @property
    def trial_s(self) -> float:
        return self.baseline_s + self.ready_s + self.mi_s + self.relax_s + self.iti_s


@dataclass
class ErdErsConfig:
    """ERD/ERS 实验配置。"""
    n_trials_left: int = 40
    n_trials_right: int = 40
    mu_band: Tuple[float, float] = MU_BAND
    beta_band: Tuple[float, float] = BETA_BAND
    bandpass: Tuple[float, float] = (8.0, 30.0)   # 仅用于显示/Hilbert（功率走 Welch 积分）
    notch_hz: Optional[float] = 50.0
    timing: ErdErsTiming = field(default_factory=ErdErsTiming)

    @property
    def n_trials(self) -> int:
        return self.n_trials_left + self.n_trials_right


# ---------------- 范式：方向序列 ----------------

def generate_directions(
    n_left: int, n_right: int, *, shuffle: bool = True, seed: Optional[int] = None
) -> List[str]:
    """生成左右手随机交错的方向序列，长度 = n_left + n_right。"""
    if n_left < 0 or n_right < 0:
        raise ValueError("n_left / n_right 须 >= 0")
    seq = [DIR_LEFT] * n_left + [DIR_RIGHT] * n_right
    if shuffle:
        random.Random(seed).shuffle(seq)
    return seq


# ---------------- 频带功率（Welch PSD，numpy-only） ----------------

def welch_psd(sig: np.ndarray, fs: float, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """numpy 实现的 Welch PSD（Hann 窗、50% 重叠、去均值）。

    Returns:
        (freqs, psd)；信号过短时返回空数组。
    """
    arr = np.asarray(sig, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.array([]), np.array([])
    if nperseg is None:
        nperseg = int(min(arr.size, max(32, fs)))   # 默认 ~1s 窗
    seg_len = int(min(max(8, nperseg), arr.size))
    step = max(1, seg_len // 2)
    window = np.hanning(seg_len)
    scale = fs * np.sum(window ** 2)
    if scale <= 0:
        return np.array([]), np.array([])

    chunks = []
    for start in range(0, arr.size - seg_len + 1, step):
        seg = arr[start:start + seg_len]
        seg = seg - np.mean(seg)
        fft_vals = np.fft.rfft(seg * window)
        psd = (np.abs(fft_vals) ** 2) / scale
        if psd.size > 2:
            psd[1:-1] *= 2.0
        chunks.append(psd)
    if not chunks:
        return np.array([]), np.array([])
    freqs = np.fft.rfftfreq(seg_len, 1.0 / fs)
    return freqs, np.mean(chunks, axis=0)


def band_power(sig: np.ndarray, fs: float, band: Tuple[float, float],
               nperseg: Optional[int] = None) -> float:
    """频带平均功率：Welch PSD 在 [lo, hi] 内积分后除以带宽（µV²/Hz × Hz → 平均功率密度）。"""
    lo, hi = band
    freqs, psd = welch_psd(sig, fs, nperseg=nperseg)
    if freqs.size == 0:
        return 0.0
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    # 用平均 PSD 作为频带功率，避免受窗长/带宽影响，利于跨窗比较。
    return float(np.mean(psd[mask]))


def band_power_timecourse(
    sig: np.ndarray, fs: float, band: Tuple[float, float],
    win_s: float = 0.5, step_s: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """滑窗频带功率时程。返回 (窗中心时间 s, 频带功率)。"""
    arr = np.asarray(sig, dtype=np.float64).ravel()
    win = max(8, int(round(win_s * fs)))
    step = max(1, int(round(step_s * fs)))
    if arr.size < win:
        return np.array([]), np.array([])
    times, powers = [], []
    for start in range(0, arr.size - win + 1, step):
        seg = arr[start:start + win]
        powers.append(band_power(seg, fs, band, nperseg=win))
        times.append((start + win / 2.0) / fs)
    return np.asarray(times), np.asarray(powers)


# ---------------- ERD/ERS ----------------

def erd_ers_percent(reference_power: float, window_power: float) -> float:
    """(window − reference) / reference × 100%。reference<=0 时返回 0。"""
    if reference_power is None or reference_power <= 0:
        return 0.0
    return float((window_power - reference_power) / reference_power * 100.0)


@dataclass
class TrialAnalysis:
    """单试次分析结果（功率 + ERD/ERS）。"""
    trial: int
    direction: str
    baseline_mu: float
    baseline_beta: float
    task_mu: float
    task_beta: float
    relax_mu: float
    relax_beta: float
    erd_mu: float
    erd_beta: float
    ers_mu: float
    ers_beta: float

    def to_row(self) -> Dict[str, object]:
        return {
            "Trial": self.trial,
            "Direction": self.direction,
            "Baseline Mu Power": self.baseline_mu,
            "Baseline Beta Power": self.baseline_beta,
            "Task Mu Power": self.task_mu,
            "Task Beta Power": self.task_beta,
            "Relax Mu Power": self.relax_mu,
            "Relax Beta Power": self.relax_beta,
            "ERD Mu": self.erd_mu,
            "ERD Beta": self.erd_beta,
            "ERS Mu": self.ers_mu,
            "ERS Beta": self.ers_beta,
        }


def analyze_trial(
    trial: int,
    direction: str,
    baseline_sig: np.ndarray,
    task_sig: np.ndarray,
    relax_sig: np.ndarray,
    fs: float,
    cfg: Optional[ErdErsConfig] = None,
) -> TrialAnalysis:
    """对单试次的三段信号计算 Mu/Beta 功率与 ERD/ERS。

    baseline_sig / task_sig / relax_sig 均为一维（已选定分析通道，单位 µV）。
    """
    if cfg is None:
        cfg = ErdErsConfig()
    mu, beta = cfg.mu_band, cfg.beta_band

    b_mu = band_power(baseline_sig, fs, mu)
    b_beta = band_power(baseline_sig, fs, beta)
    t_mu = band_power(task_sig, fs, mu)
    t_beta = band_power(task_sig, fs, beta)
    r_mu = band_power(relax_sig, fs, mu)
    r_beta = band_power(relax_sig, fs, beta)

    return TrialAnalysis(
        trial=trial,
        direction=direction,
        baseline_mu=b_mu,
        baseline_beta=b_beta,
        task_mu=t_mu,
        task_beta=t_beta,
        relax_mu=r_mu,
        relax_beta=r_beta,
        erd_mu=erd_ers_percent(b_mu, t_mu),
        erd_beta=erd_ers_percent(b_beta, t_beta),
        ers_mu=erd_ers_percent(b_mu, r_mu),
        ers_beta=erd_ers_percent(b_beta, r_beta),
    )


# ---------------- 切片工具 ----------------

def slice_by_time(
    ts_ms: np.ndarray, eeg: np.ndarray, start_ms: float, end_ms: float
) -> np.ndarray:
    """按时间戳窗 [start_ms, end_ms) 切片。

    Args:
        ts_ms: (n,) 毫秒时间戳。
        eeg: (n, C) 多通道数据。
    Returns:
        (m, C) 窗内样本；无样本时返回 (0, C)。
    """
    ts = np.asarray(ts_ms, dtype=np.float64).ravel()
    data = np.asarray(eeg, dtype=np.float64)
    if ts.size == 0 or data.shape[0] != ts.size:
        return data[:0]
    mask = (ts >= start_ms) & (ts < end_ms)
    return data[mask]


def average_erd_curve(
    trial_curves: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """把多条 (times, values) 曲线按公共时间轴平均（要求时间轴一致）。

    Returns:
        (times, mean_values)；输入为空返回空数组。
    """
    valid = [(t, v) for t, v in trial_curves if t.size > 0 and v.size == t.size]
    if not valid:
        return np.array([]), np.array([])
    min_len = min(v.size for _, v in valid)
    if min_len == 0:
        return np.array([]), np.array([])
    times = valid[0][0][:min_len]
    stack = np.vstack([v[:min_len] for _, v in valid])
    return times, np.mean(stack, axis=0)
