"""Cyton+Daisy 包解析自检（无需硬件）。"""

from __future__ import annotations

import struct

import numpy as np

from .serial_board import EEG_SCALE, PySerialCytonDaisyBoard, START_BYTE, _cast_24bit_to_int32


def _encode_channel_uv(value_uv: float) -> bytes:
    raw = int(round(value_uv / EEG_SCALE))
    raw = max(-0x800000, min(0x7FFFFF, raw))
    if raw < 0:
        raw += 0x1000000
    return bytes([(raw >> 16) & 0xFF, (raw >> 8) & 0xFF, raw & 0xFF])


def _make_packet(sample_id: int, ch_values_uv: list[float]) -> bytes:
    body = bytes([sample_id])
    for v in ch_values_uv:
        body += _encode_channel_uv(v)
    body += b"\x00" * 6  # aux
    return bytes([START_BYTE]) + body + bytes([0xC0])


def test_cyton_daisy_merge_order():
    """偶数包=Daisy CH9-16，奇数包=Cyton CH1-8；CH1 应来自奇数包。"""
    board = PySerialCytonDaisyBoard.__new__(PySerialCytonDaisyBoard)
    board.num_rows = 32
    board.eeg_channels = list(range(1, 17))  # row 1 = CH1
    board.timestamp_channel = 30
    board._partial = np.zeros(board.num_rows, dtype=np.float64)
    board._pending = []
    board._history = __import__("collections").deque(maxlen=100)
    board._lock = __import__("threading").Lock()
    board._samples_total = 0

    daisy_vals = [10.0] * 8
    cyton_vals = [95.0] + [0.0] * 7
    board._handle_packet(_make_packet(0, daisy_vals))  # even → daisy
    assert not board._pending, "偶数包不应提交整帧"
    board._handle_packet(_make_packet(1, cyton_vals))  # odd → cyton + commit

    assert len(board._pending) == 1
    pkg = board._pending[0]
    ch1_row = board.eeg_channels[0]
    ch9_row = board.eeg_channels[8]
    assert abs(pkg[ch1_row] - 95.0) < 0.5, f"CH1 应≈95µV，得 {pkg[ch1_row]}"
    assert abs(pkg[ch9_row] - 10.0) < 0.5, f"CH9 应≈10µV，得 {pkg[ch9_row]}"


if __name__ == "__main__":
    test_cyton_daisy_merge_order()
    print("test_cyton_daisy_merge_order OK")
