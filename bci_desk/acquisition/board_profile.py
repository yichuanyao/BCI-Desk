"""板卡运行时配置：波特率 + 帧/通道采样率（Cyton+Daisy 16 通道交错采样）。

OpenBCI Cyton+Daisy 在串口上 Cyton/Daisy 包交替发送，每 2 个 UART 包拼成 1 帧 16 通道数据。
板卡 ~ 命令设置的是每块 ADS1299 的帧采样率；软件滤波/落盘/FFT 应使用 **通道采样率**：

    通道采样率 (16 CH) = 帧采样率 / 2

例如 ~6 → 帧 250 Hz → 通道 125 Hz；~4 → 帧 1000 Hz → 通道 500 Hz。
默认档：115200 波特率 + 帧 250 Hz → 通道 125 Hz。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..config import SAMPLE_RATE

PROFILE_DEFAULT_125 = "default_125"
PROFILE_DEFAULT_500 = "default_500"     # 旧 meta / 高速档兼容名
PROFILE_LEGACY_125 = "legacy_125"       # 旧 meta 兼容
PROFILE_MANUAL_250 = "manual_250"       # 115200 · 帧 250 · 通道 125
PROFILE_MANUAL_1000 = "manual_1000"     # 921600 · 帧 1000 · 通道 500

# 16 通道 Daisy 交错：通道率 = 帧率 / 2
DAISY_CHANNEL_DIVISOR = 2

DEFAULT_VERIFY_TOLERANCE = 0.08

# 默认档：115200 + 帧 250 → 通道 125
DEFAULT_CHANNEL_RATE = 125
DEFAULT_FRAME_RATE = 250
DEFAULT_BAUD_RATE = 115200

# 高速档：921600 + 帧 1000 → 通道 500
HIGH_SPEED_CHANNEL_RATE = 500
HIGH_SPEED_FRAME_RATE = 1000
HIGH_SPEED_BAUD_RATE = 921600


def channel_rate_from_frame(frame_hz: int) -> int:
    return int(frame_hz) // DAISY_CHANNEL_DIVISOR


@dataclass(frozen=True)
class BoardProfile:
    name: str
    baud_rate: int
    frame_sample_rate: int
    expected_sample_rate: int
    confirmed: bool = False
    measured_sample_rate: Optional[float] = None

    @classmethod
    def default(cls) -> "BoardProfile":
        """默认档：115200 + 帧 250 Hz → 通道 125 Hz。"""
        return cls(
            name=PROFILE_DEFAULT_125,
            baud_rate=DEFAULT_BAUD_RATE,
            frame_sample_rate=DEFAULT_FRAME_RATE,
            expected_sample_rate=SAMPLE_RATE if SAMPLE_RATE == DEFAULT_CHANNEL_RATE else DEFAULT_CHANNEL_RATE,
            confirmed=False,
        )

    @classmethod
    def legacy(cls) -> "BoardProfile":
        """向后兼容：旧代码中的 legacy() 现指向默认 125 Hz 档。"""
        return cls.default()

    @classmethod
    def manual_250(cls) -> "BoardProfile":
        return cls(
            name=PROFILE_MANUAL_250,
            baud_rate=DEFAULT_BAUD_RATE,
            frame_sample_rate=DEFAULT_FRAME_RATE,
            expected_sample_rate=channel_rate_from_frame(DEFAULT_FRAME_RATE),
            confirmed=False,
        )

    @classmethod
    def manual_1000(cls) -> "BoardProfile":
        return cls(
            name=PROFILE_MANUAL_1000,
            baud_rate=HIGH_SPEED_BAUD_RATE,
            frame_sample_rate=HIGH_SPEED_FRAME_RATE,
            expected_sample_rate=channel_rate_from_frame(HIGH_SPEED_FRAME_RATE),
            confirmed=False,
        )

    @classmethod
    def high_speed(cls) -> "BoardProfile":
        """高速档：921600 + 帧 1000 Hz → 通道 500 Hz。"""
        return cls(
            name=PROFILE_DEFAULT_500,
            baud_rate=HIGH_SPEED_BAUD_RATE,
            frame_sample_rate=HIGH_SPEED_FRAME_RATE,
            expected_sample_rate=HIGH_SPEED_CHANNEL_RATE,
            confirmed=False,
        )

    @classmethod
    def profile_for_baud_rate(cls, baud_rate: int) -> "BoardProfile":
        """按工具栏波特率选择期望档位（连接/实测前使用）。"""
        if int(baud_rate) == HIGH_SPEED_BAUD_RATE:
            return cls.high_speed()
        return cls.default()

    @classmethod
    def from_name(cls, name: str) -> "BoardProfile":
        if name in (PROFILE_DEFAULT_125, PROFILE_MANUAL_250, PROFILE_LEGACY_125):
            return cls.default() if name == PROFILE_DEFAULT_125 else cls.manual_250()
        if name in (PROFILE_MANUAL_1000, PROFILE_DEFAULT_500):
            return cls.manual_1000() if name == PROFILE_MANUAL_1000 else cls.high_speed()
        return cls.default()

    @property
    def channel_sample_rate(self) -> int:
        return int(self.expected_sample_rate)

    @property
    def display_label(self) -> str:
        labels = {
            PROFILE_DEFAULT_125: (
                f"默认 (通道 {self.expected_sample_rate} Hz · 帧 {self.frame_sample_rate} · 115200)"
            ),
            PROFILE_LEGACY_125: (
                f"默认 (通道 {self.expected_sample_rate} Hz · 帧 {self.frame_sample_rate} · 115200)"
            ),
            PROFILE_MANUAL_250: (
                f"Profile A (通道 {self.expected_sample_rate} Hz · 帧 {self.frame_sample_rate} · 115200)"
            ),
            PROFILE_DEFAULT_500: (
                f"高速 (通道 {self.expected_sample_rate} Hz · 帧 {self.frame_sample_rate} · 921600)"
            ),
            PROFILE_MANUAL_1000: (
                f"Profile B (通道 {self.expected_sample_rate} Hz · 帧 {self.frame_sample_rate} · 921600)"
            ),
        }
        return labels.get(self.name, self.name)

    @property
    def short_label(self) -> str:
        return f"通道 {self.expected_sample_rate} Hz / 帧 {self.frame_sample_rate} Hz"

    @property
    def active_sample_rate(self) -> int:
        if self.measured_sample_rate is not None:
            return int(round(self.measured_sample_rate))
        return int(self.expected_sample_rate)

    def with_measurement(self, measured_hz: float) -> "BoardProfile":
        return replace(
            self,
            measured_sample_rate=float(measured_hz),
            confirmed=True,
        )

    def meta_fields(self) -> dict:
        fields = {
            "board_profile": self.name,
            "baud_rate": self.baud_rate,
            "sample_rate": self.active_sample_rate,
            "sample_rate_frame": self.frame_sample_rate,
            "sample_rate_channel": self.active_sample_rate,
            "sample_rate_source": "measured" if self.confirmed else "default",
        }
        if self.measured_sample_rate is not None:
            fields["sample_rate_measured"] = round(self.measured_sample_rate, 2)
        return fields


MANUAL_PROFILES = (BoardProfile.default(), BoardProfile.high_speed())
