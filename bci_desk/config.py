"""全局配置：采样、通道、类别、标签映射、数据单位。

这是本地程序（main.py 各标签页）与数据/预处理模块共用的唯一配置源。
所有常量集中在此，避免散落在各脚本里写死。

任务定义（对齐参考工程 Finger-BCI-Decoding 的离线 ME）：
    单指运动执行（Motor Execution）—— 受试者用优势手做单根手指的反复屈伸
    （flexion & extension），四个类别分别对应拇指 / 食指 / 中指 / 小指。

与 origin/ 的对应关系（仅作参考，运行时不依赖 origin）：
    - sample_rate / num_channels  ←  origin/config.json
    - TARGET/MIN_TIME_POINTS       ←  origin/train.py
    - microvolt_gain               ←  origin/train.py EEGPreprocessor 默认值
"""

# ---------------- 采集参数 ----------------
SAMPLE_RATE = 125          # Hz, Cyton + Daisy 16 通道通道采样率（默认档：帧 250 Hz 交错后）
NUM_CHANNELS = 16          # EEG 通道数

# ---------------- 切 epoch 参数 ----------------
# 统一目标时间点长度（2 秒）。每个 epoch 裁/补到该长度。
TARGET_TIME_POINTS = 2 * SAMPLE_RATE   # 250
# 短于此长度的片段直接丢弃。
MIN_TIME_POINTS = 50

# ---------------- 分类类别 ----------------
# 四分类单指运动执行。顺序即标签索引（y=0..3），训练/推理两端必须一致。
# 与参考工程的手指编号对应：THUMB=1, INDEX=2, MIDDLE=3, PINKY=4。
CLASSES = ["THUMB", "INDEX", "MIDDLE", "PINKY"]

# 采集范式可选的分类模式（对齐参考论文 Finger-BCI-Decoding 的 2/3 类 + 全四指 4 类）。
# condition 键与 CONDITION_TO_LABEL 一致；labels 为落盘 event_type。
CLASS_MODE_OPTIONS = {
    2: {
        "conditions": ["thumb", "pinky"],
        "labels": ["THUMB", "PINKY"],
        "description": "二分类（拇指 / 小指）",
    },
    3: {
        "conditions": ["thumb", "index", "pinky"],
        "labels": ["THUMB", "INDEX", "PINKY"],
        "description": "三分类（拇指 / 食指 / 小指）",
    },
    4: {
        "conditions": ["thumb", "index", "middle", "pinky"],
        "labels": list(CLASSES),
        "description": "四分类（拇指 / 食指 / 中指 / 小指）",
    },
}

# 采集范式 condition  ->  训练标签（event_type）。
CONDITION_TO_LABEL = {
    "thumb": "THUMB",
    "index": "INDEX",
    "middle": "MIDDLE",
    "pinky": "PINKY",
}

# PsychoPy / 论文 marker 编号（1~4 对应四指；0=休息；91=段间休息；99=初始放松）。
MARKER_REST = 0
MARKER_THUMB = 1
MARKER_INDEX = 2
MARKER_MIDDLE = 3
MARKER_PINKY = 4
MARKER_BREAK = 91
MARKER_RELAX = 99

CONDITION_TO_MARKER = {
    "thumb": MARKER_THUMB,
    "index": MARKER_INDEX,
    "middle": MARKER_MIDDLE,
    "pinky": MARKER_PINKY,
}

EVENT_TYPE_REST = "0"
EVENT_TYPE_RELAX = "RELAX"
EVENT_TYPE_BREAK = "BREAK"

# 当前主范式：两指 ME（拇指 + 小指，marker 1 / 4）。
MI_DEFAULT_CONDITIONS = ["thumb", "pinky"]

# 类别 -> 中文名（界面显示用）
LABEL_NAMES_CN = {
    "THUMB": "拇指",
    "INDEX": "食指",
    "MIDDLE": "中指",
    "PINKY": "小指",
}

# ---------------- 数据单位 ----------------
# ADS1299 在增益 24 下，每个原始 ADC 计数对应的微伏值（µV/count）。
MICROVOLT_GAIN = 0.022351744455307063

# 数据单位标识：
#   RAW_COUNT  —— OpenBCI GUI 导出的 txt / 旧 Extracted CSV，是原始 ADC 计数，需 ×MICROVOLT_GAIN 换算成 µV
#   MICROVOLT  —— 本地 BrainFlow 实时采集，已经是 µV，不可再次缩放
UNIT_RAW_COUNT = "raw_count"
UNIT_MICROVOLT = "microvolt"


def gain_for_unit(input_unit: str) -> float:
    """根据数据单位返回应乘的换算增益。

    Args:
        input_unit: UNIT_RAW_COUNT 或 UNIT_MICROVOLT。

    Returns:
        原始计数返回 MICROVOLT_GAIN；已是微伏返回 1.0（不缩放）。
    """
    if input_unit == UNIT_MICROVOLT:
        return 1.0
    if input_unit == UNIT_RAW_COUNT:
        return MICROVOLT_GAIN
    raise ValueError(
        f"未知数据单位 input_unit={input_unit!r}，应为 {UNIT_RAW_COUNT!r} 或 {UNIT_MICROVOLT!r}"
    )


def conditions_for_class_mode(n_classes: int) -> list:
    """返回指定分类模式下的 condition 列表（如 ['thumb', 'pinky']）。"""
    if n_classes not in CLASS_MODE_OPTIONS:
        raise ValueError(f"n_classes 须为 {sorted(CLASS_MODE_OPTIONS)} 之一，收到 {n_classes}")
    return list(CLASS_MODE_OPTIONS[n_classes]["conditions"])


def labels_for_class_mode(n_classes: int) -> list:
    """返回指定分类模式下的 event_type 标签列表。"""
    if n_classes not in CLASS_MODE_OPTIONS:
        raise ValueError(f"n_classes 须为 {sorted(CLASS_MODE_OPTIONS)} 之一，收到 {n_classes}")
    return list(CLASS_MODE_OPTIONS[n_classes]["labels"])


def class_mode_description(n_classes: int) -> str:
    """返回分类模式的中文说明。"""
    return CLASS_MODE_OPTIONS[n_classes]["description"]


def finger_names_cn(conditions: list) -> str:
    """把 condition 列表格式化为中文手指名，用 / 分隔。"""
    return " / ".join(
        LABEL_NAMES_CN.get(CONDITION_TO_LABEL.get(c, c), c) for c in conditions
    )
