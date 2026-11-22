"""流式实测采样率：统计 16 通道完整样本数 / 时间（通道 Hz，非板卡帧 Hz）。"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from .board_profile import DEFAULT_VERIFY_TOLERANCE, BoardProfile

DEFAULT_MEASURE_DURATION_S = 2.0
MIN_SAMPLES_FOR_MEASURE = 8


def _measure_via_sample_counter(board, duration_s: float) -> Optional[float]:
    """使用板卡内部累计的 16 通道完整帧计数（不与 UI 争抢 get_board_data）。"""
    counter = getattr(board, "stream_samples_total", None)
    if counter is None:
        return None
    c0 = int(counter)
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        time.sleep(0.02)
    elapsed = time.monotonic() - t0
    delta = int(counter) - c0
    if delta < MIN_SAMPLES_FOR_MEASURE or elapsed <= 0:
        return None
    return delta / elapsed


def _measure_via_timestamp_window(board, duration_s: float) -> Optional[float]:
    """用 ring buffer 时间戳估算（不 drain pending）。"""
    ts_ch = getattr(board, "timestamp_channel", None)
    get_recent = getattr(board, "get_current_board_data", None)
    if ts_ch is None or get_recent is None:
        return None

    time.sleep(0.15)
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        time.sleep(0.02)
    elapsed = time.monotonic() - t0
    if elapsed <= 0:
        return None

    want = max(MIN_SAMPLES_FOR_MEASURE + 2, int(elapsed * 1200) + 32)
    block = get_recent(want)
    if block.shape[1] < MIN_SAMPLES_FOR_MEASURE:
        return None

    ts = block[int(ts_ch), :].astype(float)
    dt = float(ts[-1] - ts[0])
    if dt <= 0:
        return None
    return (block.shape[1] - 1) / dt


def _measure_via_drain(board, duration_s: float, poll_interval_s: float) -> float:
    """旧路径：反复 drain get_board_data（与 UI 刷新竞争，仅作兜底）。"""
    total_samples = 0
    deadline = time.monotonic() + duration_s
    t0 = time.monotonic()

    while time.monotonic() < deadline:
        data = board.get_board_data()
        if data.shape[1] > 0:
            total_samples += int(data.shape[1])
        time.sleep(poll_interval_s)

    elapsed = time.monotonic() - t0
    if total_samples < MIN_SAMPLES_FOR_MEASURE or elapsed <= 0:
        raise RuntimeError(
            f"实测样本不足: {total_samples} 点 / {elapsed:.2f}s "
            f"(请确认已开始采集且硬件有数据)。"
        )
    return total_samples / elapsed


def measure_stream_sample_rate(
    board,
    eeg_row: int,
    duration_s: float = DEFAULT_MEASURE_DURATION_S,
    poll_interval_s: float = 0.02,
    pause_ui_poll: Optional[Callable[[], None]] = None,
    resume_ui_poll: Optional[Callable[[], None]] = None,
) -> float:
    """在已 start_stream 的板卡上实测 **16 通道通道采样率** (Hz)。

    Cyton+Daisy 交错：每 2 个 UART 包拼 1 个完整样本，此处统计的是完整样本速率。

    Raises:
        RuntimeError: 超时内样本不足或 board 无数据。
    """
    del eeg_row
    if board is None:
        raise RuntimeError("板卡未连接。")
    if duration_s <= 0:
        raise ValueError("duration_s 必须 > 0。")

    if pause_ui_poll is not None:
        pause_ui_poll()
    try:
        rate = _measure_via_sample_counter(board, duration_s)
        if rate is not None:
            return rate

        rate = _measure_via_timestamp_window(board, duration_s)
        if rate is not None:
            return rate

        return _measure_via_drain(board, duration_s, poll_interval_s)
    finally:
        if resume_ui_poll is not None:
            resume_ui_poll()


def verify_profile(
    profile: BoardProfile,
    measured_hz: float,
    tolerance: float = DEFAULT_VERIFY_TOLERANCE,
) -> Tuple[bool, str]:
    """比较实测通道 Hz 与 profile 期望值，返回 (通过, 说明)。"""
    expected = float(profile.expected_sample_rate)
    frame = float(profile.frame_sample_rate)
    if expected <= 0:
        return False, "期望通道采样率无效。"
    rel_err = abs(measured_hz - expected) / expected
    lo = expected * (1.0 - tolerance)
    hi = expected * (1.0 + tolerance)
    ok = rel_err <= tolerance
    detail = (
        f"期望通道 {expected:.0f} Hz（帧 {frame:.0f} Hz），"
        f"实测 {measured_hz:.1f} Hz "
        f"(允许 {lo:.0f}–{hi:.0f} Hz，相对误差 {rel_err * 100:.1f}%)"
    )
    if ok:
        return True, f"确认通过：{detail}"
    return False, f"确认未通过：{detail}"
