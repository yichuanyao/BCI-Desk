"""服务器训练程序的全局配置（自包含，不依赖本地 BCI-Desk 包）。

本文件夹设计为可整体拷贝到服务器独立运行，因此这里复制了与本地
采集端一致的关键常量（类别 / 标签映射 / 单位增益），
两端必须保持同步，否则训练标签与本地采集标签会对不上。

数据契约（来自本地采集/对齐定稿格式）：
    CSV 列：timestamp, event_type, EEG Channel 1..16
    旁车 <csv>.meta.json：{"unit": "microvolt"|"raw_count", "sample_rate": 125,
                          "paradigm_id": "...", "classes": [...], ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# ---------------- 采集参数（与本地一致）----------------
SAMPLE_RATE = 125          # Hz, Cyton+Daisy 16 通道通道率（meta 缺失时默认；帧 250 Hz）
NUM_CHANNELS = 16

# ---------------- 范式 → 类别（与 bci_desk/paradigms.py 对齐）----------------
# 顺序即标签索引（y=0..N-1）。训练时优先用 --paradigm / --classes 或从 meta 推断。
PARADIGM_CLASSES: Dict[str, List[str]] = {
    "finger_2": ["THUMB", "PINKY"],
    "finger_3": ["THUMB", "INDEX", "PINKY"],
    "finger_4": ["THUMB", "INDEX", "MIDDLE", "PINKY"],
    "hand_2class": ["FIST", "RELAX"],
    "hand_3class": ["FIST", "RELAX", "PALM"],
}

# 未指定范式时的默认（手指二分类）
DEFAULT_PARADIGM_ID = "finger_2"
DEFAULT_CLASSES: List[str] = list(PARADIGM_CLASSES[DEFAULT_PARADIGM_ID])
# 向后兼容旧脚本中的 CLASSES 引用
CLASSES = DEFAULT_CLASSES


def parse_classes_arg(text: str) -> List[str]:
    """解析 ``--classes THUMB,PINKY`` 形式的类别列表。"""
    parts = [c.strip() for c in text.split(",") if c.strip()]
    if not parts:
        raise ValueError("类别列表不能为空。")
    return parts


def classes_for_paradigm(paradigm_id: str) -> List[str]:
    """按范式 id 返回类别列表。"""
    if paradigm_id not in PARADIGM_CLASSES:
        known = ", ".join(sorted(PARADIGM_CLASSES))
        raise ValueError(f"未知范式 paradigm_id={paradigm_id!r}，可选: {known}")
    return list(PARADIGM_CLASSES[paradigm_id])


# ---------------- 数据单位换算（与本地一致）----------------
# ADS1299 增益 24 下，每个原始 ADC 计数对应的微伏值（µV/count）。
MICROVOLT_GAIN = 0.022351744455307063
UNIT_RAW_COUNT = "raw_count"
UNIT_MICROVOLT = "microvolt"


def gain_for_unit(input_unit: str) -> float:
    """根据数据单位返回应乘的换算增益（raw_count→µV；microvolt→不缩放）。"""
    if input_unit == UNIT_MICROVOLT:
        return 1.0
    if input_unit == UNIT_RAW_COUNT:
        return MICROVOLT_GAIN
    raise ValueError(
        f"未知数据单位 input_unit={input_unit!r}，应为 {UNIT_RAW_COUNT!r} 或 {UNIT_MICROVOLT!r}"
    )


@dataclass
class PreprocConfig:
    """预处理参数（沿用参考工程 Finger-BCI-Decoding 的流水线）。

    流程：单位换算 → CAR → 按 maxtrial 长度对齐 → 1s 滑窗切片增强
          → resample 降采样到 downsample_rate → 4-40Hz 带通 → 按时间 zscore。
    """
    sample_rate: int = SAMPLE_RATE        # 输入采样率（会被各 CSV 的 meta 覆盖）
    max_trial_s: float = 5.0              # 单试次截断/补齐长度（s）；手指 ME 5s，手形握拳 4s/放松 3s 均适用
    window_s: float = 1.0                 # 滑窗长度（s），即每个训练样本时长
    step_samples: int = 250               # 滑窗步长（原始采样点；500 Hz 下约 0.5 s）
    downsample_rate: int = 100            # 降采样后的采样率（Hz）
    bandpass: tuple = (4.0, 40.0)         # 带通频段（Hz），在降采样率下设计
    bandpass_order: int = 4
    bandpass_pad: int = 100               # lfilter 前后零填充点数，抑制边缘暂态
    use_car: bool = True

    @property
    def desired_len(self) -> int:
        """每个训练样本降采样后的时间点数 = window_s × downsample_rate。"""
        return int(self.window_s * self.downsample_rate)


@dataclass
class EEGNetConfig:
    """EEGNet 架构超参（与参考工程默认一致：EEGNet-8,2）。"""
    kern_length: int = 32
    F1: int = 8
    D: int = 2
    F2: int = 16
    dropout_rate: float = 0.5
    norm_rate: float = 0.25               # 分类层 max_norm 约束


@dataclass
class TrainConfig:
    """训练超参。"""
    epochs: int = 300
    batch_size: int = 16
    lr: float = 1e-3
    train_split: float = 0.8
    early_stop_patience: int = 80
    lr_patience: int = 30
    lr_factor: float = 0.5
    weight_decay: float = 0.0
    seed: int = 42
    num_workers: int = 0

    classes: List[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    model: EEGNetConfig = field(default_factory=EEGNetConfig)
