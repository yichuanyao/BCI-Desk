"""EEG 带通/陷波（scipy 零相位为主，BrainFlow 兜底）。"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

from brainflow.data_filter import DataFilter, FilterTypes

FilterBand = Optional[Tuple[float, float]]
NotchList = Optional[Sequence[float]]

DEFAULT_ORDER = 4
MIN_DURATION_S = 0.4


def min_filter_samples(fs: int, order: int = DEFAULT_ORDER) -> int:
    fs = int(fs)
    return max(order * 8, int(fs * MIN_DURATION_S), 32)


def apply_bandpass_notch(
    data: np.ndarray,
    fs: int,
    band: FilterBand,
    notch_freqs: NotchList,
    order: int = DEFAULT_ORDER,
    notch_half_width: float = 2.0,
) -> np.ndarray:
    """对 1D float64 信号做去均值 + 带通 + 陷波，返回新数组。"""
    out = np.ascontiguousarray(data, dtype=np.float64).copy()
    if out.size == 0:
        return out
    out -= np.mean(out)

    fs = int(round(float(fs)))
    if fs <= 0 or out.size < min_filter_samples(fs, order):
        return out

    try:
        return _apply_brainflow(out, fs, band, notch_freqs, notch_half_width, order)
    except Exception:  # noqa: BLE001
        if _HAS_SCIPY:
            return _apply_scipy(out, fs, band, notch_freqs, order)
        raise


def _apply_scipy(
    x: np.ndarray,
    fs: int,
    band: FilterBand,
    notch_freqs: NotchList,
    order: int,
) -> np.ndarray:
    nyq = fs / 2.0
    padlen = min(3 * (order + 1), max(0, x.size - 1))

    if band is not None:
        lo, hi = float(band[0]), float(band[1])
        lo = max(0.05, lo)
        hi = min(hi, nyq - 0.5)
        if hi > lo:
            sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
            x = sosfiltfilt(sos, x, padtype="odd", padlen=padlen)

    if notch_freqs:
        for f0 in notch_freqs:
            f0 = float(f0)
            if f0 <= 0 or f0 >= nyq:
                continue
            q = max(15.0, f0 / 2.0)
            b, a = iirnotch(f0, q, fs=fs)
            x = filtfilt(b, a, x, padtype="odd", padlen=padlen)
    return x


def _apply_brainflow(
    x: np.ndarray,
    fs: int,
    band: FilterBand,
    notch_freqs: NotchList,
    notch_half_width: float,
    order: int,
) -> np.ndarray:
    ftype = FilterTypes.BUTTERWORTH.value
    if band is not None:
        DataFilter.perform_bandpass(
            x, fs, float(band[0]), float(band[1]), order, ftype, 0.0,
        )
    if notch_freqs:
        for f0 in notch_freqs:
            DataFilter.perform_bandstop(
                x, fs,
                float(f0) - notch_half_width,
                float(f0) + notch_half_width,
                order, ftype, 0.0,
            )
    return x
