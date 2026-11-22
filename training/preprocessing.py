"""预处理流水线（干净重写自参考工程 Finger-BCI-Decoding 的 Functions.py）。

对每个试次（trial）依次执行：
    CAR（共平均参考） → 1s 滑窗切片增强 → resample 降采样到 downsample_rate
    → 4-40Hz 带通（前后零填充抑制暂态） → 按时间轴 zscore。

输出训练样本张量 X 形状 (N, num_channels, desired_len)，y 形状 (N,)。
每个滑窗都会复制其所属试次的标签（数据增强）。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import signal

from config import PreprocConfig


def apply_car(trial: np.ndarray) -> np.ndarray:
    """共平均参考：每个时间点减去所有通道的均值。trial 形状 (C, T)。"""
    return trial - trial.mean(axis=0, keepdims=True)


def segment_trial(trial: np.ndarray, segment_size: int, step_size: int) -> List[np.ndarray]:
    """在单试次时间轴上滑窗切片，返回若干 (C, segment_size) 片段。

    不足一个完整窗口的尾部会被丢弃（与参考工程丢弃含 NaN 段等价）。
    """
    if segment_size <= 0 or step_size <= 0:
        raise ValueError("segment_size 与 step_size 必须为正。")
    _, t = trial.shape
    segments = []
    for start in range(0, t - segment_size + 1, step_size):
        segments.append(trial[:, start:start + segment_size])
    return segments


def resample_time(x: np.ndarray, desired_len: int) -> np.ndarray:
    """沿时间轴（最后一维）FFT 重采样到 desired_len。"""
    return signal.resample(x, desired_len, axis=-1)


def bandpass_filter(x: np.ndarray, band, fs: int, order: int, pad: int) -> np.ndarray:
    """对 (..., T) 数据做 Butterworth 带通；前后零填充 pad 点再裁掉以抑制边缘暂态。"""
    b, a = signal.butter(order, band, btype="bandpass", fs=fs)
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="constant")
    filtered = signal.lfilter(b, a, padded, axis=-1)
    return filtered[..., pad:-pad]


def zscore_time(x: np.ndarray) -> np.ndarray:
    """按时间轴（最后一维）标准化，每个 (样本, 通道) 独立。"""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-8
    return (x - mean) / std


def build_dataset(
    trials: List[np.ndarray],
    labels: np.ndarray,
    sample_rates: List[int],
    cfg: PreprocConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """把试次列表预处理成训练样本张量。

    Args:
        trials: list of (C, T)，单位 µV。
        labels: (n_trials,) 类别索引。
        sample_rates: 每个试次的原始采样率。
        cfg: 预处理配置。

    Returns:
        (X, y)：
            X —— float32，(N, C, desired_len)。
            y —— int64，(N,)。
    """
    desired_len = cfg.desired_len
    seg_segments: List[np.ndarray] = []
    seg_labels: List[int] = []

    for trial, label, srate in zip(trials, labels, sample_rates):
        segment_size = int(cfg.window_s * srate)
        max_len = int(cfg.max_trial_s * srate)

        data = apply_car(trial) if cfg.use_car else trial
        data = data[:, :max_len]                       # 截断到最大试次长度

        windows = segment_trial(data, segment_size, cfg.step_samples)
        for w in windows:
            seg_segments.append(resample_time(w, desired_len))   # (C, desired_len)
            seg_labels.append(int(label))

    if not seg_segments:
        raise RuntimeError(
            "切片后没有得到任何训练样本：可能试次时长短于 window_s，"
            "请减小 window_s/step_samples 或检查数据。"
        )

    X = np.stack(seg_segments, axis=0)                 # (N, C, desired_len)
    y = np.asarray(seg_labels, dtype=np.int64)

    X = bandpass_filter(
        X, cfg.bandpass, fs=cfg.downsample_rate,
        order=cfg.bandpass_order, pad=cfg.bandpass_pad,
    )
    X = zscore_time(X)
    return X.astype(np.float32), y
