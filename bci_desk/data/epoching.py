"""EEG 预处理与切 epoch（干净重写自 origin/train.py，仅依赖 numpy/scipy/pandas）。

提供两样东西：
    1. EEGPreprocessor —— 单段 (C, T) 信号的预处理流水线（换算/CAR/陷波/带通/基线/标准化/截断）。
    2. load_epochs_from_csv —— 把带标签 CSV 按连续相同 event_type 切成等长 epoch，得到 (X, y)。

数值上与 origin/train.py 的 EEGPreprocessor.preprocess_pipeline 一致，
以保证将来服务器训练与本地推理使用同一套预处理。

与原脚本的唯一增强：新增 `input_unit` 单位开关——
    - 原始 ADC 计数（旧 txt / Extracted CSV）：×MICROVOLT_GAIN 换算成 µV
    - BrainFlow 实时采集（已是 µV）：不再缩放
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal

from ..config import (
    CLASSES,
    MICROVOLT_GAIN,
    MIN_TIME_POINTS,
    TARGET_TIME_POINTS,
    UNIT_RAW_COUNT,
    gain_for_unit,
)


class EEGPreprocessor:
    """单段 EEG 信号预处理流水线。

    输入/输出均为形状 (C, T) 的二维数组：C 个通道、T 个时间点。
    `preprocess_pipeline` 依次执行：单位换算 → CAR → 陷波 → 带通 → 基线校正 → 标准化 → 截断。

    Args:
        sampling_rate: 采样率 (Hz)。
        input_unit: 输入数据单位，UNIT_RAW_COUNT 或 UNIT_MICROVOLT，决定单位换算增益。
        microvolt_gain: 显式指定 µV/count 增益；为 None 时按 input_unit 自动取值。
        lowcut/highcut/bandpass_order: 带通滤波参数（默认 8–30 Hz，4 阶）。
        notch_freq/notch_quality: 陷波参数（默认 50 Hz，Q=30）。
        baseline_samples: 用于基线校正的起始样本数。
        use_car: 是否做共平均参考（Common Average Reference）。
        use_standardize: 是否按通道标准化。
        standardize_mode: "zscore"（均值/标准差）或 "robust"（中位数/MAD）。
        clip_std: 标准化后按 ±clip_std 截断；<=0 或 None 表示不截断。
    """

    def __init__(
        self,
        sampling_rate: int,
        input_unit: str = UNIT_RAW_COUNT,
        microvolt_gain: Optional[float] = None,
        lowcut: float = 8.0,
        highcut: float = 30.0,
        bandpass_order: int = 4,
        notch_freq: float = 50.0,
        notch_quality: float = 30.0,
        baseline_samples: int = 50,
        use_car: bool = True,
        use_standardize: bool = True,
        standardize_mode: str = "zscore",
        clip_std: float = 6.0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.input_unit = input_unit
        self.microvolt_gain = (
            gain_for_unit(input_unit) if microvolt_gain is None else float(microvolt_gain)
        )
        self.lowcut = lowcut
        self.highcut = highcut
        self.bandpass_order = bandpass_order
        self.notch_freq = notch_freq
        self.notch_quality = notch_quality
        self.baseline_samples = baseline_samples
        self.use_car = use_car
        self.use_standardize = use_standardize
        self.standardize_mode = standardize_mode
        self.clip_std = clip_std

        # 预计算固定滤波器系数（与原脚本相同的归一化方式：w0 = freq / nyquist）。
        nyquist = 0.5 * self.sampling_rate
        self._bandpass_ba = signal.butter(
            self.bandpass_order,
            [self.lowcut / nyquist, self.highcut / nyquist],
            btype="band",
        )
        self._notch_ba = signal.iirnotch(self.notch_freq / nyquist, self.notch_quality)

    # ---------- 各步骤 ----------
    def raw_to_microvolts(self, data: np.ndarray) -> np.ndarray:
        """按单位换算增益把原始计数转成微伏（µV 数据时增益为 1.0，等同不变）。"""
        return data * self.microvolt_gain

    def common_average_reference(self, data: np.ndarray) -> np.ndarray:
        """共平均参考：每个时间点减去所有通道的均值。"""
        return data - np.mean(data, axis=0, keepdims=True)

    def _filtfilt_safe(self, b, a, data: np.ndarray, name: str) -> np.ndarray:
        """零相位滤波；filtfilt 失败时退化为 lfilter，再失败则返回原数据。"""
        try:
            return signal.filtfilt(b, a, data, axis=1)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{name}: filtfilt 失败，退化为 lfilter。Err: {e}")
            try:
                return signal.lfilter(b, a, data, axis=1)
            except Exception as e2:  # noqa: BLE001
                warnings.warn(f"{name}: lfilter 也失败，返回原数据。Err: {e2}")
                return data

    def notch_filter(self, data: np.ndarray) -> np.ndarray:
        """陷波滤波（默认 50 Hz）去工频干扰。"""
        b, a = self._notch_ba
        return self._filtfilt_safe(b, a, data, "notch_filter")

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """带通滤波（默认 8–30 Hz）保留 mu/beta 频段。"""
        b, a = self._bandpass_ba
        return self._filtfilt_safe(b, a, data, "bandpass_filter")

    def baseline_correction(self, data: np.ndarray) -> np.ndarray:
        """基线校正：减去前 baseline_samples 个样本的均值。"""
        n = self.baseline_samples
        if data.shape[1] > n:
            baseline = np.mean(data[:, :n], axis=1, keepdims=True)
            return data - baseline
        return data

    def standardize(self, data: np.ndarray) -> np.ndarray:
        """按通道标准化（zscore 或 robust）。"""
        if self.standardize_mode == "robust":
            med = np.median(data, axis=1, keepdims=True)
            mad = np.median(np.abs(data - med), axis=1, keepdims=True) + 1e-8
            return (data - med) / (1.4826 * mad)
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + 1e-8
        return (data - mean) / std

    def preprocess_pipeline(self, data: np.ndarray) -> np.ndarray:
        """对单段 (C, T) 信号执行完整预处理流水线，返回处理后的 (C, T)。"""
        data = np.asarray(data, dtype=np.float64)
        data = self.raw_to_microvolts(data)
        if self.use_car:
            data = self.common_average_reference(data)
        data = self.notch_filter(data)
        data = self.bandpass_filter(data)
        data = self.baseline_correction(data)
        if self.use_standardize:
            data = self.standardize(data)
        if self.clip_std is not None and self.clip_std > 0:
            data = np.clip(data, -self.clip_std, self.clip_std)
        return data


def _fit_to_target_length(data: np.ndarray, target: int) -> np.ndarray:
    """把 (C, T) 统一到 (C, target)：过长取中间段，过短用每通道最后一个值右侧填充。"""
    t = data.shape[1]
    if t > target:
        start = (t - target) // 2
        return data[:, start:start + target]
    if t < target:
        pad_width = target - data.shape[1]
        last_vals = data[:, -1:]
        pad = np.repeat(last_vals, pad_width, axis=1)
        return np.hstack([data, pad])
    return data


def load_epochs_from_csv(
    csv_path: str,
    classes: Optional[list[str]] = None,
    input_unit: str = UNIT_RAW_COUNT,
    preprocessor: Optional[EEGPreprocessor] = None,
    sampling_rate: int = None,
):
    """从带标签 CSV 加载并预处理成等长 epoch。

    CSV 列要求：`timestamp, event_type, EEG Channel 1..N`（列名以 "EEG" 开头者视为通道）。
    连续相同的 event_type 视为一个 block（一个 trial），每个 block 切成一个 epoch。

    Args:
        csv_path: 带标签 CSV 路径。
        classes: 参与训练的类别列表（默认 config.CLASSES）；不在其中的 block 被忽略。
        input_unit: 数据单位（默认原始计数），仅在未显式传入 preprocessor 时生效。
        preprocessor: 自定义预处理器；为 None 时按 input_unit 新建一个默认 EEGPreprocessor。
        sampling_rate: 采样率，仅在新建默认 preprocessor 时使用（默认 config.SAMPLE_RATE）。

    Returns:
        (X, y):
            X —— float32 数组，形状 (N, C, TARGET_TIME_POINTS)。
            y —— int64 数组，形状 (N,)，取值为 classes.index(event_type)。

    Raises:
        RuntimeError: 没有加载到任何 epoch（检查列名与 event_type）。
    """
    if classes is None:
        classes = CLASSES
    if sampling_rate is None:
        from ..config import SAMPLE_RATE
        sampling_rate = SAMPLE_RATE
    if preprocessor is None:
        preprocessor = EEGPreprocessor(sampling_rate=sampling_rate, input_unit=input_unit)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "event_type" not in df.columns:
        raise RuntimeError(f"{csv_path} 缺少 'event_type' 列。")
    eeg_cols = [c for c in df.columns if c.startswith("EEG")]
    if len(eeg_cols) == 0:
        raise RuntimeError(f"{csv_path} 未找到以 'EEG' 开头的通道列。")

    # 连续相同 event_type 归为同一 block。
    event_shift = df["event_type"].ne(df["event_type"].shift())
    block_id = event_shift.cumsum()
    block_first_event = df.groupby(block_id)["event_type"].first()

    X, y = [], []
    for bid, event in block_first_event.items():
        if event not in classes:
            continue
        seg = df[block_id == bid]
        data = seg[eeg_cols].values.T  # (C, T)
        if data.shape[1] < MIN_TIME_POINTS:
            continue
        data = preprocessor.preprocess_pipeline(data)
        data = _fit_to_target_length(data, TARGET_TIME_POINTS)
        X.append(data.astype(np.float32))
        y.append(classes.index(event))

    if len(X) == 0:
        raise RuntimeError(
            f"未从 {csv_path} 加载到任何 epoch。请检查通道列名与 event_type 取值是否在 {classes} 中。"
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y
