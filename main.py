# -*- coding: utf-8 -*-
"""
BCI-Desk 本地 GUI 主程序（main.py）
--------------------------------
OpenBCI Cyton + Daisy 16 通道 EEG 运动执行工作台（开源）。

标签页：双通道对比 / 阻抗检测 / 专注指标 / 实时佩戴检测 /
        信号校验 / 采集范式 / ERD/ERS 检测

依赖: pip install -r requirements.txt
说明: 115200 走 BrainFlow；921600 走 pyserial 直连。返回值已缩放为微伏(uV)。
Cyton+Daisy 16 通道交错采样：默认 115200 · 帧 250 Hz · 通道 125 Hz；921600 高速档通道 500 Hz。
调试虚拟端口：python main.py --debug 或环境变量 BCI_DESK_DEBUG=1
"""

import sys
import time
import csv
import json
import os
import re
import html
import logging
import traceback
import argparse
from dataclasses import replace
from datetime import datetime
from collections import deque

import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from serial.tools import list_ports

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from brainflow.exit_codes import BrainFlowError

from bci_desk.__about__ import (
    APP_NAME,
    APP_NAME_CN,
    APP_ORG,
    APP_TAGLINE,
    GITHUB_URL,
    LICENSE_SPDX,
    __version__,
    hashtags_line,
)
# 本地业务模块(干净重写自 origin/, 不依赖 origin)
from bci_desk.config import (
    CLASSES,
    SAMPLE_RATE,
    UNIT_MICROVOLT,
)
from bci_desk.paradigms import (
    DEFAULT_PARADIGM_ID,
    PARADIGM_COMBO_GROUPS,
    default_paradigm,
    get_paradigm,
    paradigm_intro_html,
)
from bci_desk.data.epoching import load_epochs_from_csv
from bci_desk.acquisition.paradigm import (
    KIND_BREAK,
    KIND_LABEL,
    KIND_MANUAL_WAIT,
    KIND_RELAX,
    KIND_REST,
    MISessionTiming,
    build_mi_session_schedule,
    mi_session_duration_s,
    total_duration_s,
)
from bci_desk.paths import (
    ERDERS_RECORD_ROOT,
    TRAIN_RECORD_ROOT,
    TEST_RECORD_ROOT,
    UI_ASSETS_DIR,
    VALIDATION_RECORD_ROOT,
    project_root_from,
    resolve_cue_image,
)
from bci_desk.acquisition.session import (
    mi_session_csv_path,
    new_session_id,
)
from bci_desk.acquisition.recording import ParadigmRecorder
from bci_desk import erders
from bci_desk.acquisition.validation import (
    RawStreamRecorder,
    load_validation_csv,
    downsample_for_plot,
    uniform_time_s,
)
from bci_desk.acquisition.board_profile import BoardProfile
from bci_desk.acquisition.hardware_setup_dialog import HardwareSetupDialog, SampleRateVerifyWorker
from bci_desk.acquisition.serial_board import SerialBoardError, create_board
from bci_desk.acquisition.signal_filters import apply_bandpass_notch, min_filter_samples


# ---------------- 配置 ----------------
BOARD_ID = BoardIds.CYTON_DAISY_BOARD.value   # 16 通道 (Cyton + Daisy)
DEBUG_VIRTUAL_PORT = "__DEBUG_VIRTUAL__"
DEBUG_VIRTUAL_PORT_LABEL = "[调试] 虚拟端口 (仅动画预览 · 不落盘)"

# 调试虚拟端口默认隐藏；--debug 或 BCI_DESK_DEBUG=1 时显示
_DEBUG_MODE = False


def is_debug_mode() -> bool:
    return bool(_DEBUG_MODE)


def setup_app_logging(project_root: str) -> str:
    """配置应用日志，返回日志文件路径。"""
    log_dir = os.path.join(project_root, "data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"bci_desk_{datetime.now().strftime('%Y%m%d')}.log")
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)
    return log_path


def install_exception_hooks(log_path: str = ""):
    """未捕获异常写入日志，并在有 QApplication 时弹窗。"""

    def _report(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("bci_desk").error("未捕获异常:\n%s", text)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            detail = text[-2500:] if len(text) > 2500 else text
            extra = f"\n\n日志: {log_path}" if log_path else ""
            QtWidgets.QMessageBox.critical(
                None,
                f"{APP_NAME} — 未处理错误",
                f"程序遇到未处理错误，详情已写入日志。{extra}\n\n{detail}",
            )

    def _sys_hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        _report(exc_type, exc, tb)

    sys.excepthook = _sys_hook
    if hasattr(QtCore, "qInstallMessageHandler"):
        pass


def app_icon_path() -> str:
    return os.path.join(project_root_from(__file__), UI_ASSETS_DIR, "bci_desk_icon.png")


def load_app_icon() -> QtGui.QIcon:
    path = app_icon_path()
    if os.path.isfile(path):
        return QtGui.QIcon(path)
    return QtGui.QIcon()

PLOT_SECONDS = 5                              # 波形窗口显示的秒数
PAD_SECONDS = 1.0                             # 左侧预热区秒数(吸收滤波边缘振铃; 500Hz 下约 1s)
UPDATE_INTERVAL_MS = 40                       # 界面刷新间隔(毫秒)
FILTER_ORDER = 4                              # 滤波器阶数 (Butterworth), 与 OpenBCI GUI 一致

# 带通滤波选项: 显示名 -> (低频, 高频), None 表示不滤波
BANDPASS_OPTIONS = {
    "无": None,
    "1-10 Hz": (1.0, 10.0),
    "1-28 Hz": (1.0, 28.0),
    "1-50 Hz": (1.0, 50.0),
    "5-50 Hz": (5.0, 50.0),
    "7-13 Hz": (7.0, 13.0),
}

# 陷波滤波选项: 显示名 -> 需要陷掉的中心频率列表, None 表示不滤波
NOTCH_OPTIONS = {
    "无": None,
    "50 Hz": [50.0],
    "60 Hz": [60.0],
    "50+60 Hz": [50.0, 60.0],
    "25+50 Hz": [25.0, 50.0],
}
NOTCH_HALF_WIDTH = 2.0                         # 陷波带宽的一半 (Hz)

# ---------------- FFT 设置(对应 OpenBCI GUI 的 FFT Plot) ----------------
FFT_SIZE_LEGACY = 256                          # 125 Hz 通道率下 GUI 默认 256 点
FFT_MIN_AMP = 0.01                             # 幅度下限(避免 log(0)), 与 GUI min_val 一致
FFT_TARGET_DF_HZ = 0.5                         # 目标频率分辨率 (Hz)；500Hz 时需 ~1024 点
FFT_MAX_SIZE = 4096
# 最大频率(X 轴)可选项；125 Hz 通道采样奈奎斯特 62.5 Hz，500 Hz 通道奈奎斯特 250 Hz
FFT_MAXFREQ_OPTIONS = [10, 20, 40, 60]
FFT_MAXFREQ_OPTIONS_HIGH_SR = [20, 40, 60, 100, 200]


def fft_maxfreq_options_for_sample_rate(sample_rate: int):
    """按当前通道采样率返回 FFT 横轴上限可选项。"""
    if int(sample_rate) >= 400:
        return list(FFT_MAXFREQ_OPTIONS_HIGH_SR)
    return list(FFT_MAXFREQ_OPTIONS)


def fft_size_for_sample_rate(sample_rate: int) -> int:
    """按通道采样率选取 FFT 点数，使低频分辨率与 125Hz×256 档相当。

    125 Hz / 256 ≈ 0.49 Hz/bin；503 Hz 若仍用 256 → 1.97 Hz/bin，5 Hz 测试信号会
    落在两 bin 之间，幅值被 Hamming 窗压低，看起来「频率/幅度都不对」。
    """
    sr = max(1, int(round(float(sample_rate))))
    need = int(np.ceil(sr / FFT_TARGET_DF_HZ))
    need = max(FFT_SIZE_LEGACY, min(FFT_MAX_SIZE, need))
    return int(DataFilter.get_nearest_power_of_two(need))

# ---------------- 示波器(实时监测页)设置 ----------------
NUM_SCOPES = 2                       # 双通道对比页的示波器数量
SCOPE_COLORS = [(22, 104, 192), (217, 119, 6)]   # 示波器1 蓝 / 示波器2 琥珀(时域与 FFT 同色)

# ---------------- 板卡硬件设置(对应 OpenBCI GUI 的 Hardware Settings) ----------------
# Cyton 通道配置命令格式:  x (通道) (掉电) (增益) (输入类型) (Bias) (SRB2) (SRB1) X
# 通道编号字符: 1-8 通道用 '1'-'8'; 9-16 通道(Daisy)用 'Q W E R T Y U I'
APPLY_HW_SETTINGS = True            # 连接时是否把下列设置写入板卡
CHANNEL_IDS = "12345678QWERTYUI"   # 16 个通道对应的命令字符
HW_POWER_DOWN = "0"   # 0=通道开启
HW_GAIN = "6"         # 增益: 0=1x 1=2x 2=4x 3=6x 4=8x 5=12x 6=24x  -> 6 即 x24
HW_INPUT_TYPE = "0"   # 0=Normal
HW_BIAS = "1"         # 1=计入 Bias (对应 GUI 的 Bias Include = Yes)
HW_SRB2 = "0"         # 0=Off (与 GUI 一致)
HW_SRB1 = "1"         # 1=On  (与 GUI 一致)


def build_channel_command(channel_char):
    """生成单个通道的硬件设置命令字符串, 例如通道1 -> 'x1060101X'。"""
    return ("x" + channel_char + HW_POWER_DOWN + HW_GAIN + HW_INPUT_TYPE
            + HW_BIAS + HW_SRB2 + HW_SRB1 + "X")


# ---------------- 阻抗检测设置(参考手册 2.2.2.1 导联脱落 / OpenBCI 公式) ----------------
# 阻抗命令格式:  z (通道) (P端电流源) (N端电流源) Z   例如通道1开P端 -> 'z110Z', 关闭 -> 'z100Z'
IMP_DRIVE_AMPS = 6.0e-9      # 阻抗测量驱动电流, ADS1299/OpenBCI 为 6 nA
IMP_SERIES_RESISTOR = 2200   # 板上串联电阻, 单位 Ω (OpenBCI 为 2.2kΩ)
IMP_WINDOW_SECONDS = 2.0     # 用于计算 RMS 的时间窗(秒)
IMP_UPDATE_MS = 500          # 阻抗数值刷新间隔(毫秒)
PROGRESS_BAR_WIDTH = 320     # 进度条最小宽度(可随布局拉伸)
# 满量程(饱和)阈值: ADS1299 在增益24下满量程 = Vref/gain = 4.5/24 = 187500 uV
# 超过满量程 90% 视为通道饱和(railed), 通常意味着电极开路/未佩戴, 此时无法测阻抗
ADS1299_FULLSCALE_UV = 4.5 / 24 * 1.0e6
RAIL_THRESHOLD_UV = 0.9 * ADS1299_FULLSCALE_UV

# 阻抗页 UI: 质量分级与图例（合格阈值 = ≤100 kΩ，含「可接受」）
IMP_PASS_KOHM = 100
IMP_QUALITY_LEGEND = [
    ("≤10 kΩ", "极好", "excellent"),
    ("≤50 kΩ", "良好", "good"),
    ("≤100 kΩ", "可接受（合格上限）", "acceptable"),
    ("≤200 kΩ", "偏高", "high"),
    (">200 kΩ", "差/未接好", "poor"),
]
IMP_QUALITY_OK_TIERS = {"excellent", "good", "acceptable"}  # 合格 = ≤ IMP_PASS_KOHM

# ---------------- 专注指标频谱分析设置 ----------------
FOCUS_WINDOW_SECONDS = 4.0           # PSD 分析窗口长度(秒)
FOCUS_UPDATE_MS = 50                 # 注意力指标刷新间隔(毫秒)
FOCUS_THRESHOLD_DEFAULT = 70.0       # 默认注意力阈值 (0-100)
FOCUS_FILTER_ORDER = 3               # 带通/陷波阶数 (timeseries_order)
FOCUS_RIPPLE = 0.1                   # 滤波器纹波 (ripple_value)
FOCUS_PLOT_SECONDS = 5.0             # 注意力曲线显示滑窗长度(秒)
# 陷波频段 (25/50/60 Hz)
FOCUS_NOTCH_BANDS = [
    (23.0, 27.0),   # 25 Hz
    (48.0, 52.0),   # 50 Hz
    (58.0, 62.0),   # 60 Hz
]
# 频带定义: (名称, 低频, 高频, 颜色)
FOCUS_BANDS = [
    ("Delta", 0.5, 3.0, (120, 120, 120)),
    ("Theta", 4.0, 7.0, (80, 140, 220)),
    ("Alpha", 8.0, 13.0, (40, 160, 90)),
    ("Beta", 14.0, 30.0, (220, 80, 60)),
    ("Gamma", 30.0, 50.0, (180, 60, 180)),
]

# ---------------- 专注指标页 (β/(α×4)×100 专注指数 + 对比指标) ----------------
FOCUS2_ARTIFACT_THRESHOLD_UV = 150.0   # 伪迹剔除幅度阈值 (focus1)
FOCUS2_CALIBRATION_SECONDS = 5.0       # 静息校准时长
FOCUS2_QUALITY_FLATLINE_STD = 0.8      # 平线判定 (µV)
FOCUS2_QUALITY_SATURATION_UV = 450.0   # 饱和判定 (µV)
FOCUS2_PERSONAL_THRESHOLD_SIGMA = 1.0  # 个体阈值 = 静息均值 + σ×标准差
FOCUS2_BANDPASS = (1.0, 28.0)        # 专注指标页带通频段
# focus1 四指标: (key, 显示名, 标签色 hex, 曲线 RGB)
FOCUS2_COG_SPECS = [
    ("attention", "Attention", "#da9100", (218, 145, 0)),    # 琥珀金, 与专注指数蓝区分
    ("relaxation", "Relaxation", "#2da44e", (45, 164, 78)),
    ("focus", "Focus", "#c05028", (192, 80, 40)),
    ("meditation", "Meditation", "#7b4fbb", (123, 79, 187)),
]
FOCUS2_BT_COLOR = (0, 120, 215)

# ---------------- 佩戴检测页 ----------------
WEARING_WINDOW_SECONDS = 5.0
WEARING_UPDATE_MS = 80
WEARING_ANALYSIS_INTERVAL_SECONDS = 0.5
WEARING_DISPLAY_POINTS = 150
WEARING_SCORE_THRESHOLD = 8
WEARING_Y_RANGE_UV = 300.0
WEARING_SQI_WINDOW_SECONDS = 3.0
WEARING_SQI_OK_THRESHOLD = 60.0
WEARING_SQI_MIN_STD_UV = 1.0
WEARING_SQI_MAX_STD_UV = 120.0
WEARING_SQI_MAX_PTP_UV = 600.0
WEARING_SQI_LINE_RATIO_WARN = 0.30
WEARING_SQI_ARTIFACT_UV = 180.0
WEARING_SQI_ARTIFACT_RATIO_WARN = 0.10
WEARING_SQI_EMA_ALPHA = 0.35
WEARING_SQI_OK_ON_THRESHOLD = 60.0
WEARING_SQI_OK_OFF_THRESHOLD = 40.0
# ---------------- 采集范式页 (ME 交互 session, 多范式) ----------------
ACQUIRE_TICK_MS = 40
_PROJECT_ROOT = project_root_from(__file__)
ACQUIRE_RECORD_DIR = TRAIN_RECORD_ROOT
ACQUIRE_PHASE_TEXT = {
    KIND_MANUAL_WAIT: ("按空格键继续", "#5c6bc0"),
    KIND_RELAX: ("请闭眼放松，保持静止", "#5c6bc0"),
    KIND_BREAK: ("段间休息，可眨眼活动", "#2da44e"),
    KIND_REST: ("注视「+」，保持静止", "#2da44e"),
    "prepare": ("准备", "#d8a200"),
    "iti": ("休息", "#2da44e"),
    "runbreak": ("本组结束, 放松休息", "#2da44e"),
}

# ---------------- 模型测试采集 (合并至采集范式页, 缩短版 ME session) ----------------
TEST_DEFAULT_TRIALS_PER_SEGMENT = 20
TEST_DEFAULT_SEGMENTS = 1
TEST_RECORD_DIR = TEST_RECORD_ROOT
TEST_TICK_MS = ACQUIRE_TICK_MS

# ---------------- 信号校验页 (固定时长原始流录制 + 单通道波形回看) ----------------
VALIDATE_DEFAULT_SECONDS = 10.0
VALIDATE_TICK_MS = 40
VALIDATE_RECORD_DIR = VALIDATION_RECORD_ROOT
VALIDATE_MAX_PLOT_POINTS = 2000       # 绘图降采样上限
VALIDATE_Y_RANGE_UV = 200.0        # 默认 Y 轴范围(µV), 正弦波可在此调
WEARING_COLORS = [
    (0, 120, 215), (0, 160, 80), (200, 60, 40), (160, 90, 200),
    (230, 140, 0), (80, 150, 180), (120, 120, 120), (180, 80, 120),
]
# 16 通道头顶示意图电极位置 (归一化 x/y, 鼻尖朝上, 受试者左/右视角)
WEARING_HEAD_ELECTRODES = [
    (0.38, 0.10),   # CH1  前额左
    (0.62, 0.10),   # CH2  前额右
    (0.40, 0.50),   # CH3  左区内
    (0.60, 0.50),   # CH4  右区内
    (0.18, 0.72),   # CH5  下颞左
    (0.82, 0.72),   # CH6  下颞右
    (0.38, 0.88),   # CH7  枕左
    (0.62, 0.88),   # CH8  枕右
    (0.18, 0.28),   # CH9  上颞左
    (0.82, 0.28),   # CH10 上颞右
    (0.35, 0.32),   # CH11 左内上
    (0.65, 0.32),   # CH12 右内上
    (0.08, 0.50),   # CH13 耳侧左
    (0.92, 0.50),   # CH14 耳侧右
    (0.35, 0.68),   # CH15 左内下
    (0.65, 0.68),   # CH16 右内下
]
WEARING_SQI_THRESHOLD_LABEL = f"{WEARING_SQI_OK_ON_THRESHOLD:.0f}"
WEARING_TAB_INFO = (
    "- **SQI 评分**: 饱和/平线/std/峰峰值/**50Hz 占比**/伪迹比例 → 每通道 **0-100 分**\n"
    f"- **显示**: 跟随顶部全局带通/陷波; 仿 OpenBCI GUI 横向 16 通道, 每通道最近 **{WEARING_DISPLAY_POINTS}** 点"
)


def remove_eye_blink_artifacts(data, threshold=FOCUS2_ARTIFACT_THRESHOLD_UV):
    """振幅超阈值的样本用线性插值替换 (参考 focus1/eeg_processor.py)。"""
    clean = np.ascontiguousarray(data, dtype=np.float64).copy()
    mask = np.abs(clean) > threshold
    if not np.any(mask):
        return clean
    indices = np.where(mask)[0]
    good = np.where(~mask)[0]
    if len(good) > 1:
        clean[mask] = np.interp(indices, good, clean[good])
    return clean


def estimate_channel_quality(channel_data):
    """单通道信号质量启发式评估 (参考 focus1)。"""
    ch = np.asarray(channel_data, dtype=np.float64)
    std = float(np.std(ch))
    peak = float(np.max(np.abs(ch)))
    flatline = std < FOCUS2_QUALITY_FLATLINE_STD
    saturated = peak > FOCUS2_QUALITY_SATURATION_UV
    if flatline or saturated:
        return {"ok": False, "quality": 0.0, "flatline": flatline, "saturated": saturated, "std": std}
    noise = float(np.median(np.abs(np.diff(ch))))
    q = 1.0 - min(noise / 50.0, 1.0)
    rms = float(np.sqrt(np.mean(ch ** 2)))
    if rms < 1.0 or rms > 300.0:
        q *= 0.5
    return {"ok": q >= 0.3, "quality": max(0.0, min(1.0, q)), "flatline": flatline,
            "saturated": saturated, "std": std}


def compute_focus1_cognitive_indices(band_powers):
    """
    focus1 四认知指标 (0-1), 参考 focus1/eeg_processor.py。
    band_powers: [delta, theta, alpha, beta, gamma] (gamma 作 low_gamma 使用)
    """
    delta = band_powers[0]
    theta = band_powers[1]
    alpha = band_powers[2]
    beta = band_powers[3]
    low_gamma = band_powers[4]

    att_raw = beta / max(alpha + theta, 1e-9)
    attention = 1.0 / (1.0 + np.exp(-2.0 * (att_raw - 1.0)))

    relaxation = alpha / max(alpha + beta, 1e-9)

    focus_raw = (beta + low_gamma) / max(delta + theta, 1e-9)
    focus = 1.0 / (1.0 + np.exp(-1.5 * (focus_raw - 1.0)))

    med_raw = theta / max(alpha + beta, 1e-9)
    meditation = 1.0 / (1.0 + np.exp(-2.0 * (med_raw - 0.5)))

    return {
        "attention": float(attention),
        "relaxation": float(relaxation),
        "focus": float(focus),
        "meditation": float(meditation),
        "attention_raw": float(att_raw),
    }


def _kurtosis_numpy(data):
    """等价于 scipy.stats.kurtosis 默认 fisher=True 的轻量实现。"""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    centered = arr - np.mean(arr)
    var = np.mean(centered ** 2)
    if var <= 1e-12:
        return 0.0
    return float(np.mean(centered ** 4) / (var ** 2) - 3.0)


def _welch_psd_numpy(data, fs, nperseg):
    """用于佩戴检测的 Welch PSD 简化实现, 避免额外依赖 scipy。"""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return np.array([]), np.array([])
    seg_len = int(min(max(1, nperseg), arr.size))
    step = max(1, seg_len // 2)
    window = np.hanning(seg_len)
    scale = fs * np.sum(window ** 2)
    if scale <= 0:
        return np.array([]), np.array([])

    psd_chunks = []
    for start in range(0, arr.size - seg_len + 1, step):
        segment = arr[start:start + seg_len]
        segment = segment - np.mean(segment)
        fft_vals = np.fft.rfft(segment * window)
        psd = (np.abs(fft_vals) ** 2) / scale
        if psd.size > 2:
            psd[1:-1] *= 2.0
        psd_chunks.append(psd)

    if not psd_chunks:
        return np.array([]), np.array([])
    freqs = np.fft.rfftfreq(seg_len, 1.0 / fs)
    return freqs, np.mean(psd_chunks, axis=0)




def check_wearing_sqi(eeg_data, fs, window_seconds=WEARING_SQI_WINDOW_SECONDS):
    """
    基于 SQI(Signal Quality Index) 的单通道质量检测。
    返回: (is_worn_placeholder, ok_channels, channels_info, avg_score)
    """
    eeg = np.asarray(eeg_data, dtype=np.float64)
    if eeg.ndim != 2 or eeg.shape[1] == 0:
        return False, 0, [], 0.0

    num_ch = min(eeg.shape[0], 16)
    samples = int(min(max(1, fs * window_seconds), eeg.shape[1]))
    channels_info = []
    ok_channels = 0
    total_score = 0.0

    for i in range(num_ch):
        raw = eeg[i, -samples:]
        raw_centered = np.ascontiguousarray(raw - np.mean(raw), dtype=np.float64)
        max_abs_uv = float(np.max(np.abs(raw)))
        railed = max_abs_uv > RAIL_THRESHOLD_UV

        # 标准差/峰峰值用预处理后的 1-40Hz 信号, 避免 DC 漂移把接上的通道误判为 std_high。
        processed = raw_centered.copy()
        if processed.size >= max(16, int(fs * 0.5)):
            DataFilter.detrend(processed, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                processed, fs, 1.0, 40.0,
                FILTER_ORDER, FilterTypes.BUTTERWORTH.value, 0.0,
            )
        std_uv = float(np.std(processed))
        ptp_uv = float(np.ptp(processed))
        artifact_ratio = float(np.mean(np.abs(processed) > WEARING_SQI_ARTIFACT_UV))
        flatline = std_uv < WEARING_SQI_MIN_STD_UV

        # 50Hz 占比仍用去均值后的原始信号算, 不先 notch, 这样才能看出工频污染。
        f, psd = _welch_psd_numpy(raw_centered, fs, int(fs * 2))
        if psd.size > 0:
            line_mask = (f > 49) & (f < 51)
            eeg_mask = (f >= 1) & (f <= 45)
            line_power = float(np.sum(psd[line_mask])) if np.any(line_mask) else 0.0
            eeg_power = float(np.sum(psd[eeg_mask])) if np.any(eeg_mask) else 0.0
            line_ratio = line_power / eeg_power if eeg_power > 1e-12 else 0.0
        else:
            line_ratio = 0.0

        score = 100.0
        reasons = []
        if railed:
            score = 0.0
            reasons.append("railed")
        else:
            if flatline:
                score -= 60.0
                reasons.append("flatline")
            if std_uv > WEARING_SQI_MAX_STD_UV:
                score -= 40.0
                reasons.append("std_high")
            if ptp_uv > WEARING_SQI_MAX_PTP_UV:
                score -= 10.0
                reasons.append("ptp_high")
            if line_ratio > WEARING_SQI_LINE_RATIO_WARN:
                score -= 15.0
                reasons.append("50hz_high")
            if artifact_ratio > WEARING_SQI_ARTIFACT_RATIO_WARN:
                score -= 30.0
                reasons.append("artifact")

        score = max(0.0, min(100.0, score))
        worn = score >= WEARING_SQI_OK_THRESHOLD
        ok_channels += 1 if worn else 0
        total_score += score

        channels_info.append({
            "channel": i,
            "score": round(score, 1),
            "std": round(std_uv, 2),
            "ptp": round(ptp_uv, 1),
            "raw_std": round(float(np.std(raw_centered)), 2),
            "line_ratio": round(line_ratio, 3),
            "artifact_ratio": round(artifact_ratio, 3),
            "worn": worn,
            "reason": "OK" if not reasons else ",".join(reasons),
        })

    avg_score = total_score / num_ch if num_ch > 0 else 0.0
    return ok_channels >= WEARING_SCORE_THRESHOLD, ok_channels, channels_info, avg_score


def build_impedance_command(channel_char, p_on, n_on):
    """生成阻抗设置命令, 例如通道1开P端 -> 'z110Z'。"""
    return "z" + channel_char + ("1" if p_on else "0") + ("1" if n_on else "0") + "Z"


def safe_config_board(board, cmd):
    """向板卡发送配置命令; 忽略 BrainFlow 解析响应时的 UTF-8 解码错误。"""
    if board is None:
        return
    try:
        board.config_board(cmd)
    except UnicodeDecodeError:
        pass
    except (BrainFlowError, SerialBoardError):
        pass


# ---------------- 界面主题(医疗软件风格) ----------------
# 仅用于美化外观, 不改变任何功能逻辑。冷静的临床蓝 + 洁净留白 + 卡片化分区。
MEDICAL_PALETTE = {
    "primary": "#1668c0",        # 主色: 临床蓝
    "primary_dark": "#0f4f97",   # 按下/悬停加深
    "primary_light": "#e7f0fb",  # 浅色高亮底
    "bg": "#eef2f7",             # 窗口背景: 柔和蓝灰
    "surface": "#ffffff",        # 卡片/面板底色
    "border": "#e2e8f0",         # 分隔线/边框(略淡, 少抢波形)
    "text": "#1f2d3d",           # 主文字
    "muted": "#64748b",          # 次要文字
    "ok": "#1a7f37",             # 正常/成功
    "warn": "#bf8700",           # 警告
    "danger": "#d1242f",         # 异常/危险
    "disabled_bg": "#e8eef4",    # 禁用按钮底
    "disabled_fg": "#64748b",    # 禁用按钮字(提高对比)
}

# 采集范式页背景（与 MEDICAL_PALETTE["bg"] 一致，供提示图合成用）
ACQUIRE_BG_COLOR = "#eef2f7"
# 大屏提示字号（px；须用 stylesheet 设置，否则会被全局 13px 覆盖）
ACQUIRE_CUE_FONT_PX = 72
ACQUIRE_HINT_FONT_PX = 27

# ---------------- ERD/ERS 检测页 ----------------
ERDERS_TICK_MS = 33                       # 范式时钟/取数刷新间隔(毫秒)
ERDERS_RECORD_DIR = ERDERS_RECORD_ROOT
ERDERS_DEFAULT_TRIALS_PER_DIR = 40        # 每侧(左/右手)默认试次数
ERDERS_CUE_COLOR_BASELINE = "#2da44e"     # 基线(绿)
ERDERS_CUE_COLOR_READY = "#d8a200"        # 准备(琥珀)
ERDERS_CUE_COLOR_MI = "#c0392b"           # 运动执行(红)
ERDERS_CUE_COLOR_RELAX = "#1668c0"        # 放松(蓝)
ERDERS_CUE_COLOR_ITI = "#888888"          # 试次间隔(灰)
ERDERS_MU_COLOR = (40, 160, 90)           # Mu 频带曲线
ERDERS_BETA_COLOR = (200, 60, 40)         # Beta 频带曲线
ERDERS_LEFT_COLOR = (22, 104, 192)        # 左手平均曲线
ERDERS_RIGHT_COLOR = (192, 80, 40)        # 右手平均曲线

MEDICAL_STYLESHEET = """
* {{
    font-family: "Microsoft YaHei UI", "Segoe UI", "Microsoft YaHei", sans-serif;
}}
QMainWindow {{
    background-color: {bg};
    color: {text};
    font-size: 13px;
}}
QWidget {{
    color: {text};
    font-size: 13px;
}}
QWidget#centralRoot {{
    background-color: {bg};
}}
QDialog {{
    background-color: {bg};
    color: {text};
}}

/* 顶部标题栏 */
QFrame#appHeader {{
    background-color: {surface};
    border: none;
    border-bottom: 1px solid {border};
}}
QLabel#appTitle {{
    color: {primary};
    font-size: 20px;
    font-weight: 700;
    background: transparent;
}}
QLabel#appSubtitle {{
    color: {muted};
    font-size: 12px;
    background: transparent;
}}
QLabel#appBadge {{
    color: {primary};
    background-color: {primary_light};
    border: 1px solid {primary};
    border-radius: 11px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
}}

/* 顶部设备条(连接 + 滤波合并为一张卡片) */
QFrame#deviceBar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QFrame#deviceBar QLabel {{
    background: transparent;
    color: {muted};
    font-size: 12px;
}}
QWidget#deviceInline, QWidget#connectProgressSlot {{
    background: transparent;
    border: none;
}}
QLabel#statusDot {{
    border-radius: 5px;
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
    background-color: {warn};
}}
QLabel#statusText {{
    background: transparent;
    color: {text};
    font-size: 13px;
    font-weight: 600;
}}
QLabel#deviceBarSection {{
    color: {muted};
    font-size: 12px;
    font-weight: 600;
    background: transparent;
    padding-right: 4px;
}}
QFrame#deviceBarDivider {{
    background-color: {border};
    max-height: 1px;
    min-height: 1px;
    border: none;
    margin: 2px 0;
}}

/* 兼容旧 objectName(若别处引用) */
QFrame#toolbar, QFrame#statusBar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}

/* 双通道页工具条 */
QFrame#monitorToolbar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QFrame#monitorToolbar QLabel {{
    background: transparent;
    color: {muted};
}}
QLabel#scopeColorDot {{
    border-radius: 5px;
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
}}
QLabel#emptyHint {{
    color: {muted};
    font-size: 13px;
    background: transparent;
    padding: 4px 8px;
}}

/* 选项卡 */
QTabWidget::pane {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
    top: -1px;
}}
QTabBar {{
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: transparent;
    color: {muted};
    padding: 9px 18px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    font-size: 13px;
}}
QTabBar::tab:hover {{
    color: {primary};
    background: {primary_light};
}}
QTabBar::tab:selected {{
    background: {surface};
    color: {primary};
    font-weight: 700;
    border: 1px solid {border};
    border-bottom: 2px solid {primary};
}}

/* 按钮 — 默认实心为主操作; #btnSecondary 为次要描边 */
QPushButton {{
    background-color: {primary};
    color: #ffffff;
    border: 1px solid {primary};
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {primary_dark};
    border-color: {primary_dark};
}}
QPushButton:pressed {{
    background-color: {primary_dark};
    border-color: {primary_dark};
}}
QPushButton:disabled {{
    background-color: {disabled_bg};
    color: {disabled_fg};
    border: 1px solid {border};
}}
QPushButton#btnSecondary {{
    background-color: {surface};
    color: {primary};
    border: 1px solid {primary};
}}
QPushButton#btnSecondary:hover {{
    background-color: {primary_light};
    color: {primary_dark};
    border-color: {primary_dark};
}}
QPushButton#btnSecondary:pressed {{
    background-color: {primary_light};
    color: {primary_dark};
}}
QPushButton#btnSecondary:disabled {{
    background-color: #f8fafc;
    color: {disabled_fg};
    border: 1px solid {border};
}}
QPushButton#btnPrimary {{
    background-color: {primary};
    color: #ffffff;
    border: 1px solid {primary};
    padding: 8px 20px;
}}

/* 下拉框 / 数字框 */
QComboBox, QDoubleSpinBox, QSpinBox {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 5px 10px;
    padding-right: 26px;
    min-height: 20px;
    color: {text};
    selection-background-color: {primary};
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
    border: 1px solid {primary};
}}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {primary};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid {border};
    border-bottom: 1px solid {border};
    background-color: {surface};
    border-top-right-radius: 5px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 1px solid {border};
    background-color: {surface};
    border-bottom-right-radius: 5px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {primary_light};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {surface};
    border: 1px solid {border};
    selection-background-color: {primary_light};
    selection-color: {primary};
    outline: none;
}}

/* 复选框 */
QCheckBox {{
    spacing: 7px;
    background: transparent;
    color: {text};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {border};
    border-radius: 4px;
    background: {surface};
}}
QCheckBox::indicator:hover {{
    border: 1px solid {primary};
}}
QCheckBox::indicator:checked {{
    background: {primary};
    border: 1px solid {primary};
    image: none;
}}

/* 分组框(卡片) */
QGroupBox {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px;
    font-weight: 600;
    color: {text};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 2px 10px;
    color: {primary};
    background-color: {primary_light};
    border-radius: 5px;
}}

/* 滚动条 */
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #c2cedd;
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {primary};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    width: 0px;
    border: none;
    background: none;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: #c2cedd;
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {primary};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    height: 0px;
    width: 0px;
    border: none;
    background: none;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* 阻抗检测页 */
QFrame#impHeader {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#impPageTitle {{
    color: {primary};
    font-size: 18px;
    font-weight: 700;
    background: transparent;
}}
QFrame#impControlBar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#impSummary {{
    color: {text};
    font-size: 13px;
    font-weight: 600;
    background: transparent;
}}
QFrame#impChannelCard {{
    background-color: {surface};
    border: 1px solid {border};
    border-left: 4px solid {border};
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QLabel#impChBadge {{
    background-color: {primary_light};
    color: {primary};
    border: 1px solid {primary};
    border-radius: 20px;
    font-size: 14px;
    font-weight: 700;
    font-family: Consolas, "Cascadia Mono", monospace;
}}
QLabel#impChNum {{
    color: {muted};
    font-size: 11px;
    background: transparent;
}}
QLabel#impValue {{
    color: {text};
    font-size: 22px;
    font-weight: 700;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QLabel#impUnit {{
    color: {muted};
    font-size: 12px;
    background: transparent;
}}
QLabel#impStatus {{
    font-size: 12px;
    font-weight: 600;
    background: transparent;
    border-radius: 10px;
    padding: 4px 12px;
}}
QFrame#impLegend {{
    background-color: {primary_light};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#impLegendTitle {{
    color: {primary};
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}}
QLabel#impLegendItem {{
    color: {muted};
    font-size: 11px;
    background: transparent;
}}
QLabel#impLegendDot {{
    border-radius: 5px;
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
}}

/* 佩戴检测页 - 右侧通道面板 */
QFrame#wearingRightPanel {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QTabWidget#wearingResultTabs::pane {{
    border: none;
    background: transparent;
    margin-top: 4px;
}}
QTabWidget#wearingResultTabs > QTabBar::tab {{
    background: {bg};
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 16px;
    margin-right: 4px;
    color: {muted};
    font-size: 13px;
    font-weight: 600;
}}
QTabWidget#wearingResultTabs > QTabBar::tab:selected {{
    background: {surface};
    color: {primary};
    border-bottom: 2px solid {primary};
}}
QFrame#wearingChCard {{
    background-color: {surface};
    border: 1px solid {border};
    border-left: 4px solid {border};
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QLabel#wearingChBadge {{
    border-radius: 16px;
    font-size: 13px;
    font-weight: 700;
    font-family: Consolas, "Cascadia Mono", monospace;
    color: white;
}}
QLabel#wearingChScore {{
    color: {text};
    font-size: 20px;
    font-weight: 700;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QLabel#wearingChMeta {{
    color: {muted};
    font-size: 12px;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QLabel#wearingChStatus {{
    font-size: 13px;
    font-weight: 600;
    background: transparent;
    border-radius: 8px;
    padding: 3px 10px;
}}

/* 专注指标页 */
QFrame#focus2Header {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#focus2PageTitle {{
    color: {primary};
    font-size: 18px;
    font-weight: 700;
    background: transparent;
}}
QFrame#focus2ControlBar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QFrame#focus2HeroCard {{
    background-color: {primary_light};
    border: 1px solid {border};
    border-left: 4px solid {primary};
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QFrame#focus2CompactBar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#focus2CompactLabel {{
    color: {muted};
    font-size: 11px;
    font-weight: 600;
    background: transparent;
}}
QLabel#focus2CompactValue {{
    color: {primary};
    font-size: 20px;
    font-weight: 700;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QLabel#focus2InlineValue {{
    font-size: 14px;
    font-weight: 700;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
    min-width: 42px;
}}

/* 信号校验页 */
QFrame#validateHeader {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#validatePageTitle {{
    color: {primary};
    font-size: 18px;
    font-weight: 700;
    background: transparent;
}}
QFrame#validateControlBar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QFrame#validateControlBar QLabel {{
    background: transparent;
    color: {muted};
    font-size: 12px;
}}
QFrame#validatePlotCard {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#validatePlotHint {{
    color: {muted};
    font-size: 13px;
    background: transparent;
    padding: 2px 4px;
}}
QLabel#validatePlotInfo {{
    color: {muted};
    font-size: 12px;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
    padding: 2px 4px;
}}
QFrame#validateSidePanel {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#validateSideTitle {{
    color: {primary};
    font-size: 13px;
    font-weight: 700;
    background: transparent;
}}
QLabel#validateMetaLabel {{
    color: {muted};
    font-size: 11px;
    font-weight: 600;
    background: transparent;
}}
QLabel#validateMetaValue {{
    color: {text};
    font-size: 13px;
    font-weight: 600;
    background: transparent;
}}
QLabel#validateMetaMono {{
    color: {muted};
    font-size: 12px;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QPlainTextEdit#validateStatsView {{
    background-color: {bg};
    border: 1px solid {border};
    border-radius: 6px;
    color: {text};
    font-family: Consolas, "Cascadia Mono", monospace;
    font-size: 11px;
    padding: 8px;
}}

QFrame#focus2InfoBar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#focus2HeroTitle {{
    color: {primary};
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}}
QLabel#focus2HeroValue {{
    color: {primary};
    font-size: 34px;
    font-weight: 700;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QLabel#focus2HeroSub {{
    color: {muted};
    font-size: 11px;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QLabel#focus2StatusBadge {{
    font-size: 12px;
    font-weight: 600;
    border-radius: 12px;
    padding: 5px 14px;
    background: transparent;
}}
QFrame#focus2MetricCard {{
    background-color: {surface};
    border: 1px solid {border};
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QLabel#focus2MetricTitle {{
    font-size: 11px;
    font-weight: 600;
    background: transparent;
}}
QLabel#focus2MetricValue {{
    font-size: 22px;
    font-weight: 700;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QFrame#focus2StatusBar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#focus2StatusLine {{
    color: {muted};
    font-size: 12px;
    background: transparent;
}}
QFrame#focus2AnalyticsCard {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#focus2SectionTitle {{
    color: {text};
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}}
QLabel#focus2VarLine {{
    color: {muted};
    font-size: 12px;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QLabel#focus2BandChip {{
    background-color: {primary_light};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 2px 8px;
    font-family: Consolas, "Cascadia Mono", monospace;
    font-size: 10px;
}}
QFrame#focus2ChartCard {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#focus2ChartTitle {{
    color: {text};
    font-size: 13px;
    font-weight: 600;
    background: transparent;
}}

/* ERD/ERS 检测页 */
QFrame#erdersStageCard {{
    background-color: {bg};
    border: 1px solid {border};
    border-radius: 10px;
}}
QFrame#erdersSidePanel {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QFrame#erdersSectionCard {{
    background-color: {bg};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#erdersPageTitle {{
    color: {primary};
    font-size: 16px;
    font-weight: 700;
    background: transparent;
}}
QLabel#erdersFieldLabel {{
    color: {muted};
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}}
QLabel#erdersMetaLabel {{
    color: {muted};
    font-size: 11px;
    font-weight: 600;
    background: transparent;
}}
QLabel#erdersMetaValue {{
    color: {text};
    font-size: 13px;
    font-weight: 600;
    background: transparent;
}}
QLabel#erdersMetaMono {{
    color: {muted};
    font-size: 12px;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QFrame#erdersPlotCard {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QTableWidget#erdersTable {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 6px;
    gridline-color: {border};
}}

/* 采集范式页 */
QFrame#acquireStageCard {{
    background-color: {bg};
    border: 1px solid {border};
    border-radius: 10px;
}}
QFrame#acquireSidePanel {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QFrame#acquireSectionCard {{
    background-color: {bg};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#acquireSectionTitle {{
    color: {primary};
    font-size: 14px;
    font-weight: 700;
    background: transparent;
}}
QLabel#acquireFieldLabel {{
    color: {muted};
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}}
QLabel#acquireNote {{
    color: {muted};
    font-size: 11px;
    background: transparent;
}}
QLabel#acquireMetaLabel {{
    color: {muted};
    font-size: 11px;
    font-weight: 600;
    background: transparent;
}}
QLabel#acquireMetaValue {{
    color: {text};
    font-size: 13px;
    font-weight: 600;
    background: transparent;
}}
QLabel#acquireMetaMono {{
    color: {muted};
    font-size: 12px;
    font-family: Consolas, "Cascadia Mono", monospace;
    background: transparent;
}}
QLineEdit {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 10px;
    color: {text};
    selection-background-color: {primary};
}}
QLineEdit:hover {{
    border: 1px solid {primary};
}}
QLineEdit:focus {{
    border: 1px solid {primary};
}}
QLineEdit:disabled {{
    background-color: {disabled_bg};
    color: {disabled_fg};
}}

/* 页面说明提示 (ⓘ 悬停显示) */
QToolButton#infoTipButton {{
    background-color: {primary_light};
    color: {primary};
    border: 1px solid {primary};
    border-radius: 10px;
    font-size: 12px;
    font-weight: 700;
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    padding: 0;
}}
QToolButton#infoTipButton:hover {{
    background-color: {primary};
    color: #ffffff;
}}

/* 全局原生 tooltip (下拉框等) 也统一风格 */
QToolTip {{
    background-color: {surface};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* 自定义 ⓘ 悬浮说明卡片 */
QFrame#infoTipPopup {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 10px;
}}
QFrame#infoTipPopupAccent {{
    background-color: {primary};
    border-top-left-radius: 9px;
    border-bottom-left-radius: 9px;
}}
QLabel#infoTipPopupTitle {{
    color: {primary};
    font-size: 13px;
    font-weight: 700;
    background: transparent;
}}
QLabel#infoTipPopupBody {{
    background: transparent;
}}
""".format(**MEDICAL_PALETTE)


def _emphasize_info_html(text):
    """将 **重点** 转为加粗主题色 HTML (输入需已 html.escape)。"""
    primary = MEDICAL_PALETTE["primary"]

    def repl(match):
        return f'<span style="color:{primary}; font-weight:700;">{match.group(1)}</span>'

    return re.sub(r"\*\*(.+?)\*\*", repl, text)


def format_info_tip_html(text):
    """把 - 分点 / **强调** 文本转为说明卡片 HTML。"""
    p = MEDICAL_PALETTE
    intro_parts = []
    bullet_rows = []
    html_chunks = []

    def flush_bullets():
        if not bullet_rows:
            return
        rows = "".join(bullet_rows)
        html_chunks.append(
            f'<table cellspacing="0" cellpadding="0" style="margin:0;">{rows}</table>'
        )
        bullet_rows.clear()

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        is_bullet = line.startswith(("- ", "• ", "* "))
        if is_bullet:
            line = line[2:].strip()
        escaped = _emphasize_info_html(html.escape(line))
        if is_bullet:
            bullet_rows.append(
                f'<tr>'
                f'<td valign="top" style="color:{p["primary"]}; font-size:13px; font-weight:700;'
                f' padding:3px 8px 3px 0;">•</td>'
                f'<td style="color:{p["text"]}; font-size:12px; line-height:1.6; padding:3px 0;">'
                f'{escaped}</td></tr>'
            )
        else:
            flush_bullets()
            intro_parts.append(escaped)

    flush_bullets()
    if intro_parts:
        html_chunks.insert(
            0,
            f'<p style="margin:0 0 8px 0; color:{p["muted"]}; font-size:12px; line-height:1.55;">'
            f'{" ".join(intro_parts)}</p>',
        )
    return "".join(html_chunks)


class InfoTipPopup(QtWidgets.QFrame):
    """与界面风格一致的悬浮说明卡片。"""

    def __init__(self, owner):
        super().__init__(None, QtCore.Qt.ToolTip | QtCore.Qt.FramelessWindowHint)
        self._owner = owner
        self.setObjectName("infoTipPopup")
        self.setMaximumWidth(420)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        accent = QtWidgets.QFrame()
        accent.setObjectName("infoTipPopupAccent")
        accent.setFixedWidth(4)
        accent.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding,
        )
        outer.addWidget(accent)

        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title = QtWidgets.QLabel("说明")
        title.setObjectName("infoTipPopupTitle")
        layout.addWidget(title)

        body = QtWidgets.QLabel(format_info_tip_html(owner._info_text))
        body.setObjectName("infoTipPopupBody")
        body.setWordWrap(True)
        body.setTextFormat(QtCore.Qt.RichText)
        body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        body.setOpenExternalLinks(False)
        layout.addWidget(body)

        outer.addWidget(content, 1)

    def enterEvent(self, event):
        if self._owner is not None:
            self._owner._cancel_hide()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._owner is not None:
            self._owner._schedule_hide()
        super().leaveEvent(event)


class InfoTipButton(QtWidgets.QToolButton):
    """ⓘ 按钮: 悬停显示自定义说明卡片 (非系统黄底 tooltip)。"""

    _open_button = None

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self._info_text = text
        self._popup = None
        self._anchor_provider = None
        self.setObjectName("infoTipButton")
        self.setText("i")
        self.setCursor(QtCore.Qt.WhatsThisCursor)
        self.setAutoRaise(True)
        self.setFixedSize(20, 20)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self._show_timer = QtCore.QTimer(self)
        self._show_timer.setSingleShot(True)
        self._show_timer.setInterval(280)
        self._show_timer.timeout.connect(self._show_popup)

        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(160)
        self._hide_timer.timeout.connect(self._hide_popup)

    def set_info_text(self, text):
        """更新说明文本；若弹层已创建则丢弃缓存以便下次悬停显示新内容。"""
        self._info_text = text
        if self._popup is not None:
            self._popup.hide()
            self._popup.deleteLater()
            self._popup = None

    def set_popup_anchor_provider(self, provider):
        """可选: 指定弹层锚点 (全局坐标), 用于选项卡等无固定按钮位置的场景。"""
        self._anchor_provider = provider

    def _popup_anchor_global(self):
        if callable(self._anchor_provider):
            return self._anchor_provider()
        return self.mapToGlobal(QtCore.QPoint(0, self.height() + 8))

    def enterEvent(self, event):
        self._cancel_hide()
        self._show_timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._show_timer.stop()
        self._schedule_hide()
        super().leaveEvent(event)

    def _cancel_hide(self):
        self._hide_timer.stop()

    def _schedule_hide(self):
        self._hide_timer.start()

    def _ensure_popup(self):
        if self._popup is None:
            self._popup = InfoTipPopup(self)
        return self._popup

    def _show_popup(self):
        if InfoTipButton._open_button is not None and InfoTipButton._open_button is not self:
            InfoTipButton._open_button._hide_popup()

        popup = self._ensure_popup()
        popup.adjustSize()

        anchor = self._popup_anchor_global()
        screen = QtWidgets.QApplication.desktop().availableGeometry(anchor)
        x = min(anchor.x(), screen.right() - popup.width() + 1)
        x = max(x, screen.left())
        y = anchor.y()
        if y + popup.height() > screen.bottom():
            y = anchor.y() - popup.height() - 16
        popup.move(x, y)
        popup.show()
        InfoTipButton._open_button = self

    def _hide_popup(self):
        self._show_timer.stop()
        self._hide_timer.stop()
        if self._popup is not None and self._popup.isVisible():
            self._popup.hide()
        if InfoTipButton._open_button is self:
            InfoTipButton._open_button = None


def pixmap_blend_bg(pixmap, bg_hex, white_threshold=235):
    """将接近白色的像素替换为程序背景色，使提示图与界面融合。"""
    if pixmap is None or pixmap.isNull():
        return pixmap
    img = pixmap.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
    bg = QtGui.QColor(bg_hex)
    w, h = img.width(), img.height()
    for y in range(h):
        for x in range(w):
            c = img.pixelColor(x, y)
            if (
                c.alpha() > 0
                and c.red() >= white_threshold
                and c.green() >= white_threshold
                and c.blue() >= white_threshold
            ):
                img.setPixelColor(x, y, bg)
    return QtGui.QPixmap.fromImage(img)


def ensure_spin_arrow_assets():
    """生成 SpinBox 上下箭头 PNG（Windows 下 CSS 三角形会渲染成黑条）。"""
    ui_dir = os.path.join(project_root_from(__file__), UI_ASSETS_DIR)
    os.makedirs(ui_dir, exist_ok=True)
    up_path = os.path.join(ui_dir, "spin_up.png")
    down_path = os.path.join(ui_dir, "spin_down.png")
    color = QtGui.QColor(MEDICAL_PALETTE["text"])

    def _draw(path, upward):
        pix = QtGui.QPixmap(16, 12)
        pix.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        if upward:
            pts = [QtCore.QPoint(8, 2), QtCore.QPoint(14, 10), QtCore.QPoint(2, 10)]
        else:
            pts = [QtCore.QPoint(8, 10), QtCore.QPoint(14, 2), QtCore.QPoint(2, 2)]
        painter.drawPolygon(QtGui.QPolygon(pts))
        painter.end()
        pix.save(path)

    if not os.path.isfile(up_path):
        _draw(up_path, True)
        _draw(down_path, False)
    return up_path.replace("\\", "/"), down_path.replace("\\", "/")


def spinbox_arrow_stylesheet():
    """返回 SpinBox 箭头 PNG 的 stylesheet 片段（须在 QApplication 创建后调用）。"""
    up_url, down_url = ensure_spin_arrow_assets()
    return f"""
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({up_url});
    width: 16px;
    height: 12px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({down_url});
    width: 16px;
    height: 12px;
}}
"""


class VirtualDebugBoard:
    """调试用虚拟板卡：驱动范式提示动画，不连接硬件，不落盘。"""

    def __init__(self, n_rows, sampling_rate, timestamp_channel, eeg_channels, tick_ms=40):
        self._n_rows = int(n_rows)
        self._sr = int(sampling_rate)
        self._ts_ch = int(timestamp_channel)
        self._eeg_chs = list(eeg_channels)
        self._tick_ms = int(tick_ms)
        self._streaming = False

    def prepare_session(self):
        pass

    def release_session(self):
        self._streaming = False

    def config_board(self, _cmd):
        pass

    def start_stream(self):
        self._streaming = True

    def stop_stream(self):
        self._streaming = False

    def get_board_data(self):
        if not self._streaming:
            return np.zeros((self._n_rows, 0), dtype=np.float64)
        n = max(1, int(self._sr * self._tick_ms / 1000.0))
        data = np.zeros((self._n_rows, n), dtype=np.float64)
        t_end = time.time()
        data[self._ts_ch, :] = np.linspace(t_end - n / self._sr, t_end, n, endpoint=False)
        return data


class ImpedanceBoardWorker(QtCore.QThread):
    """阻抗检测启停后台线程: 16 路 z 指令串口往返较慢, 避免卡住 UI。"""

    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(self, board, mode, p_on=False, parent=None):
        super().__init__(parent)
        self.board = board
        self.mode = mode          # "start" | "stop"
        self.p_on = p_on

    def run(self):
        try:
            n = len(CHANNEL_IDS)
            if self.mode == "start":
                for i, ch_char in enumerate(CHANNEL_IDS):
                    self.progress.emit(f"正在配置阻抗通道 {i + 1}/{n}...")
                    safe_config_board(
                        self.board,
                        build_impedance_command(ch_char, self.p_on, not self.p_on),
                    )
                    time.sleep(0.04)
                self.board.start_stream()
                try:
                    self.board.get_board_data()
                except (BrainFlowError, UnicodeDecodeError):
                    pass
            else:
                try:
                    self.board.stop_stream()
                except (BrainFlowError, UnicodeDecodeError):
                    pass
                for i, ch_char in enumerate(CHANNEL_IDS):
                    self.progress.emit(f"正在关闭电流源 {i + 1}/{n}...")
                    safe_config_board(
                        self.board, build_impedance_command(ch_char, False, False),
                    )
                    time.sleep(0.04)
            self.finished_ok.emit()
        except (BrainFlowError, UnicodeDecodeError) as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class BoardConnectWorker(QtCore.QThread):
    """板卡连接后台线程: prepare_session + 16 路硬件 x 指令, 避免卡住 UI。"""

    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, port, apply_hw, baud_rate=115200, parent=None):
        super().__init__(parent)
        self.port = port
        self.apply_hw = apply_hw
        self.baud_rate = int(baud_rate)

    def run(self):
        try:
            self.progress.emit(f"正在打开串口 ({self.baud_rate})...")
            board, backend = create_board(self.port, self.baud_rate, BOARD_ID)
            board.prepare_session()
            hw_msg = f"  |  {backend} @ {self.baud_rate}"
            if self.apply_hw:
                n = len(CHANNEL_IDS)
                for i, ch_char in enumerate(CHANNEL_IDS):
                    self.progress.emit(f"正在写入硬件设置 {i + 1}/{n}...")
                    safe_config_board(board, build_channel_command(ch_char))
                    time.sleep(0.06)
                hw_msg += "  |  硬件设置已写入(SRB2 Off/SRB1 On, x24)"
            self.finished_ok.emit(board, hw_msg)
        except (BrainFlowError, SerialBoardError, UnicodeDecodeError) as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class SquarePlotCard(QtWidgets.QFrame):
    """右侧图表卡: 子控件按卡片宽度做成正方形，铺满可用宽度，不两侧留白。"""

    def __init__(self, child, parent=None, min_side=120):
        super().__init__(parent)
        self.setObjectName("erdersPlotCard")
        self._child = child
        self._min_side = int(min_side)
        self._pad = 4
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(self._pad, self._pad, self._pad, self._pad)
        lay.setSpacing(0)
        child.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(child, 0, QtCore.Qt.AlignCenter)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 边长 = 本卡可用宽度 → 两图并排时各自铺满右栏的一半
        side = max(self._min_side, self.width() - 2 * self._pad)
        if self._child.width() != side or self._child.height() != side:
            self._child.setFixedSize(side, side)
        need_h = side + 2 * self._pad
        if self.height() != need_h:
            self.setFixedHeight(need_h)


class ImpSmoothProgressBar(QtWidgets.QWidget):
    """丝滑进度条: 缓动填充 + 不确定态流光。"""

    def __init__(self, parent=None, width=PROGRESS_BAR_WIDTH, stretch=False):
        super().__init__(parent)
        self.setObjectName("impSmoothProgress")
        if stretch:
            self.setFixedHeight(12)
            self.setMinimumWidth(width)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed,
            )
        else:
            self.setFixedSize(width, 12)
        self._value = 0.0
        self._target = 0.0
        self._indeterminate = False
        self._success = False
        self._pulse = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._animate)

    def set_progress(self, percent, indeterminate=False, success=False):
        self._target = max(0.0, min(100.0, float(percent)))
        self._indeterminate = indeterminate
        self._success = success
        if not self._timer.isActive():
            self._timer.start()

    def reset(self):
        self._value = 0.0
        self._target = 0.0
        self._indeterminate = False
        self._success = False
        self._timer.stop()
        self.update()

    def _animate(self):
        changed = False
        if self._indeterminate:
            self._pulse = (self._pulse + 0.028) % 1.0
            changed = True
        else:
            delta = self._target - self._value
            if abs(delta) >= 0.35:
                self._value += delta * 0.14
                changed = True
            elif abs(delta) > 0.01:
                self._value = self._target
                changed = True
            if abs(self._value - self._target) < 0.01:
                self._value = self._target
                self._timer.stop()
        if changed:
            self.update()

    def paintEvent(self, _event):
        pal = MEDICAL_PALETTE
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        radius = h / 2.0

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(pal["primary_light"]))
        painter.drawRoundedRect(QtCore.QRectF(0, 0, w, h), radius, radius)

        if self._indeterminate:
            chunk = w * 0.32
            x = (w + chunk) * self._pulse - chunk
            grad = QtGui.QLinearGradient(x, 0, x + chunk, 0)
            grad.setColorAt(0.0, QtGui.QColor(255, 255, 255, 0))
            grad.setColorAt(0.45, QtGui.QColor(pal["primary"]))
            grad.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
            painter.setBrush(grad)
            painter.drawRoundedRect(QtCore.QRectF(x, 0, chunk, h), radius, radius)
        elif self._value > 0.4:
            fill_w = w * self._value / 100.0
            grad = QtGui.QLinearGradient(0, 0, max(fill_w, 1.0), 0)
            if self._success:
                grad.setColorAt(0.0, QtGui.QColor("#2da44e"))
                grad.setColorAt(1.0, QtGui.QColor(pal["ok"]))
            else:
                grad.setColorAt(0.0, QtGui.QColor(pal["primary"]))
                grad.setColorAt(1.0, QtGui.QColor(pal["primary_dark"]))
            painter.setBrush(grad)
            painter.drawRoundedRect(QtCore.QRectF(0, 0, fill_w, h), radius, radius)
        painter.end()


class WearingHeadSchematicWidget(QtWidgets.QWidget):
    """16 通道佩戴质量头顶拓扑示意图 (结合阻抗页风格 + OpenBCI 16ch 布局)。"""

    _OK = QtGui.QColor("#2da44e")
    _BAD = QtGui.QColor("#d1242f")
    _IDLE = QtGui.QColor("#94a3b8")
    _HEAD_FILL = QtGui.QColor("#f0f4f8")
    _HEAD_STROKE = QtGui.QColor("#64748b")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 300)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding,
        )
        self.setMouseTracking(True)
        self._channels = [None] * 16
        self._hover_ch = -1

    def set_channels(self, channels_info):
        """channels_info: list[dict|None], 含 stable_score / stable_worn 等字段。"""
        self._channels = list(channels_info) if channels_info else [None] * 16
        while len(self._channels) < 16:
            self._channels.append(None)
        self.update()

    def _head_geometry(self):
        w, h = self.width(), self.height()
        margin_top, margin_bottom = 26, 36
        side = min(w - 12, h - margin_top - margin_bottom)
        cx = w / 2.0
        cy = margin_top + side / 2.0 + 4
        radius = side / 2.0
        return cx, cy, radius

    def _node_pos(self, ch, cx, cy, radius):
        nx, ny = WEARING_HEAD_ELECTRODES[ch]
        return cx + (nx - 0.5) * radius * 1.72, cy + (ny - 0.5) * radius * 1.72

    def _hit_test(self, pos):
        cx, cy, radius = self._head_geometry()
        node_r = max(14, radius * 0.088)
        for ch in range(15, -1, -1):
            px, py = self._node_pos(ch, cx, cy, radius)
            if (pos.x() - px) ** 2 + (pos.y() - py) ** 2 <= (node_r + 4) ** 2:
                return ch
        return -1

    def mouseMoveEvent(self, event):
        ch = self._hit_test(event.pos())
        if ch != self._hover_ch:
            self._hover_ch = ch
            self.update()
        if ch >= 0 and self._channels[ch]:
            info = self._channels[ch]
            worn = info.get("stable_worn", info.get("worn"))
            state = "OK" if worn else "BAD"
            tip = (
                f"CH{ch + 1}: {state}  SQI={info.get('stable_score', info.get('score', '---'))}\n"
                f"std={info.get('std', '---')}  50Hz={info.get('line_ratio', '---')}\n"
                f"{info.get('reason', '')}"
            )
            self.setToolTip(tip)
        else:
            self.setToolTip("")
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover_ch = -1
        self.setToolTip("")
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy, radius = self._head_geometry()
        node_r = max(14, radius * 0.088)

        painter.setPen(QtGui.QPen(self._HEAD_STROKE, 1.5))
        painter.setBrush(self._HEAD_FILL)
        painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

        ear_w, ear_h = radius * 0.10, radius * 0.16
        for sign in (-1, 1):
            ex = cx + sign * (radius + ear_w * 0.35)
            painter.drawArc(
                QtCore.QRectF(ex - ear_w, cy - ear_h / 2, ear_w * 2, ear_h),
                90 * 16 if sign < 0 else 270 * 16, 180 * 16,
            )

        dash = QtGui.QPen(QtGui.QColor("#b0bcc9"), 1, QtCore.Qt.DashLine)
        painter.setPen(dash)
        painter.drawLine(QtCore.QPointF(cx, cy - radius * 0.92), QtCore.QPointF(cx, cy + radius * 0.92))
        painter.drawLine(QtCore.QPointF(cx - radius * 0.92, cy), QtCore.QPointF(cx + radius * 0.92, cy))
        painter.drawEllipse(QtCore.QPointF(cx, cy), radius * 0.55, radius * 0.55)

        painter.setPen(QtGui.QPen(QtGui.QColor("#94a3b8"), 1))
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QtCore.QRectF(cx - radius, 4, radius, 18), QtCore.Qt.AlignRight, "受试者左")
        painter.drawText(QtCore.QRectF(cx, 4, radius, 18), QtCore.Qt.AlignLeft, "受试者右")

        ref_font = painter.font()
        ref_font.setPointSize(9)
        painter.setFont(ref_font)
        painter.setPen(QtGui.QPen(QtGui.QColor("#64748b")))
        painter.drawText(
            QtCore.QRectF(cx - 10, cy - 8, 20, 16),
            QtCore.Qt.AlignCenter, "R",
        )

        for ch, (nx, ny) in enumerate(WEARING_HEAD_ELECTRODES):
            px, py = self._node_pos(ch, cx, cy, radius)
            info = self._channels[ch] if ch < len(self._channels) else None
            if info is None:
                fill = self._IDLE
                value_text = "---"
            else:
                worn = info.get("stable_worn", info.get("worn"))
                fill = self._OK if worn else self._BAD
                score = info.get("stable_score", info.get("score"))
                value_text = f"{score:.0f}" if score is not None else "---"

            hover = ch == self._hover_ch
            draw_r = node_r + (2 if hover else 0)
            if hover:
                painter.setPen(QtGui.QPen(QtGui.QColor("#1668c0"), 2))
            else:
                painter.setPen(QtGui.QPen(QtGui.QColor("#1f2d3d"), 1.2))
            painter.setBrush(fill)
            painter.drawEllipse(QtCore.QPointF(px, py), draw_r, draw_r)

            num_font = painter.font()
            num_font.setPointSize(9)
            num_font.setBold(True)
            painter.setFont(num_font)
            painter.setPen(QtGui.QPen(QtCore.Qt.white if info else QtGui.QColor("#334155")))
            painter.drawText(
                QtCore.QRectF(px - draw_r, py - draw_r, draw_r * 2, draw_r * 2),
                QtCore.Qt.AlignCenter, str(ch + 1),
            )

            val_font = painter.font()
            val_font.setPointSize(9)
            val_font.setBold(False)
            painter.setFont(val_font)
            painter.setPen(QtGui.QPen(fill.darker(115) if info else self._IDLE))
            offset_y = draw_r + 2
            painter.drawText(
                QtCore.QRectF(px - 26, py + offset_y, 52, 16),
                QtCore.Qt.AlignCenter, value_text,
            )

        legend_y = h - 32
        painter.setPen(QtCore.Qt.NoPen)
        for i, (color, label) in enumerate([(self._BAD, "异常"), (self._OK, "正常")]):
            lx = 16 + i * 72
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPointF(lx, legend_y), 5, 5)
            painter.setPen(QtGui.QPen(QtGui.QColor("#64748b")))
            leg_font = painter.font()
            leg_font.setPointSize(10)
            painter.setFont(leg_font)
            painter.drawText(QtCore.QRectF(lx + 12, legend_y - 8, 56, 16), QtCore.Qt.AlignLeft, label)
            painter.setPen(QtCore.Qt.NoPen)

        thresh_font = painter.font()
        thresh_font.setPointSize(10)
        painter.setFont(thresh_font)
        painter.setPen(QtGui.QPen(QtGui.QColor("#64748b")))
        painter.drawText(
            QtCore.QRectF(w - 130, legend_y - 8, 118, 16),
            QtCore.Qt.AlignRight,
            f"SQI 阈值: {WEARING_SQI_THRESHOLD_LABEL}",
        )
        painter.end()


class SubjectCueWindow(QtWidgets.QDialog):
    """受试者全屏提示窗：只显示 cue 图与文字，操作员仍用主窗口右侧控件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} — 受试者提示")
        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.CustomizeWindowHint
            | QtCore.Qt.WindowCloseButtonHint
        )
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setStyleSheet(f"QDialog {{ background-color: {ACQUIRE_BG_COLOR}; }}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        self.cue_image = QtWidgets.QLabel()
        self.cue_image.setAlignment(QtCore.Qt.AlignCenter)
        self.cue_image.setMinimumSize(640, 480)
        self.cue_image.setStyleSheet(f"background-color: {ACQUIRE_BG_COLOR};")
        layout.addWidget(self.cue_image, 1)

        self.cue_label = QtWidgets.QLabel("")
        self.cue_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cue_label.setStyleSheet(
            f"font-size: 56px; font-weight: bold; color: {MEDICAL_PALETTE['primary']}; background: transparent;"
        )
        layout.addWidget(self.cue_label)

        self.hint_label = QtWidgets.QLabel("")
        self.hint_label.setAlignment(QtCore.Qt.AlignCenter)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            f"font-size: 28px; font-weight: bold; color: {MEDICAL_PALETTE['text']}; background: transparent;"
        )
        layout.addWidget(self.hint_label)

        exit_btn = QtWidgets.QPushButton("退出全屏")
        exit_btn.setMinimumHeight(44)
        exit_btn.setMinimumWidth(180)
        exit_btn.clicked.connect(self.exit_fullscreen)
        layout.addWidget(exit_btn, 0, QtCore.Qt.AlignHCenter)

        self._full_pixmap = None  # 未缩放原图，按窗口尺寸再缩放

    def exit_fullscreen(self):
        self.hide()
        self.close()

    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        # 首次全屏后尺寸才稳定，再按大图重绘
        QtCore.QTimer.singleShot(50, self._apply_full_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_full_pixmap()

    def _apply_full_pixmap(self):
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return
        # 预留下方文字与按钮区域，按可用高度缩放
        avail_w = max(200, self.cue_image.width())
        avail_h = max(200, self.cue_image.height())
        scaled = self._full_pixmap.scaled(
            avail_w,
            avail_h,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.cue_image.setPixmap(scaled)

    def sync_from(self, full_pixmap, cue_label, hint_label):
        """full_pixmap 应为未缩小的高清图；此处按全屏区域缩放。"""
        self._full_pixmap = full_pixmap
        if full_pixmap is None or full_pixmap.isNull():
            self.cue_image.clear()
        else:
            self._apply_full_pixmap()
        self.cue_label.setText(cue_label.text())
        self.cue_label.setStyleSheet(
            cue_label.styleSheet().replace(f"{ACQUIRE_CUE_FONT_PX}px", "56px")
        )
        self.hint_label.setText(hint_label.text())
        self.hint_label.setStyleSheet(
            hint_label.styleSheet().replace(f"{ACQUIRE_HINT_FONT_PX}px", "28px")
        )


class EEGMonitor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — {APP_NAME_CN}")
        self.setWindowIcon(load_app_icon())
        self.resize(1180, 900)

        # BrainFlow 相关状态
        self.board = None
        self.is_connected = False
        self.is_debug_virtual = False
        self.is_streaming = False
        self.imp_running = False             # 是否正在做阻抗检测
        self.imp_buffer = np.array([])       # 阻抗检测用的数据缓冲
        self._imp_board_worker = None        # 阻抗启停后台线程
        self._imp_p_on = True
        self._connect_board_worker = None    # 板卡连接后台线程
        self._connect_port = ""
        self.focus_plot_points = int(FOCUS_PLOT_SECONDS * 1000 / FOCUS_UPDATE_MS)
        self.focus_time_axis = np.linspace(-FOCUS_PLOT_SECONDS, 0, self.focus_plot_points)

        # 专注指标页状态
        self.focus2_running = False
        self.focus2_calibrating = False
        self.focus2_calib_until = 0.0
        self.focus2_calib_focus_vals = []
        self.focus2_calib_ch1_sum = 0.0
        self.focus2_calib_ch2_sum = 0.0
        self.focus2_calib_n_samples = 0
        self.focus2_rest_mean_ch1 = None
        self.focus2_rest_mean_ch2 = None
        self.focus2_baseline_mean = None
        self.focus2_baseline_std = None
        self.focus2_personal_threshold = FOCUS_THRESHOLD_DEFAULT
        self.focus2_value = 0.0
        self.focus2_run = False
        self.focus2_signal_ok = False
        self.focus2_history = deque(maxlen=self.focus_plot_points)
        self.focus2_cog_histories = {
            key: deque(maxlen=self.focus_plot_points)
            for key, _, _, _ in FOCUS2_COG_SPECS
        }
        self.wearing_exp_running = False
        self.wearing_exp_buffer = np.array([])
        self.wearing_exp_is_worn = False
        self.wearing_exp_score = 0
        self.wearing_exp_last_analysis_at = 0.0
        self.wearing_exp_score_ema = np.full(16, np.nan)
        self.wearing_exp_channel_ok = [False] * 16

        # 采集范式页状态
        self.acquire_running = False
        self.acquire_recorder = None
        self.acquire_schedule = None

        self.test_running = False
        self.test_recorder = None
        self.test_schedule = None
        self.test_last_csv = ""
        self.acquire_t0 = 0.0
        self.acquire_cue_pixmaps = {}
        self._current_cue_full_pixmap = None
        self._last_paradigm_display_key = None

        # 信号校验页状态
        self.validate_running = False
        self.validate_recorder = None
        self.validate_end_ms = 0.0
        self.validate_plot_eeg = None
        self.validate_plot_eeg_names = None
        self.validate_plot_sample_rate = SAMPLE_RATE

        # ERD/ERS 检测页状态
        self.erders_running = False
        self.erders_schedule = []          # 展开后的阶段列表(dict)
        self.erders_phase_idx = -1
        self.erders_phase_start_ms = 0.0
        self.erders_ts_chunks = []         # 时间戳(ms)片段
        self.erders_eeg_chunks = []        # (n,16) µV 片段(未滤波, 便于离线复算)
        self.erders_phase_marks = []       # 每个阶段 (trial, direction, phase, start_ms, end_ms)
        self.erders_results = []           # 每试次 TrialAnalysis
        self.erders_left_mu_curves = []    # 左手 MI 期 mu 功率时程(用于平均曲线)
        self.erders_right_mu_curves = []
        self.erders_config = erders.ErdErsConfig()

        # 板卡运行时档位（默认 115200 / 通道 125；921600 高速档通道 500）
        self.board_profile = BoardProfile.default()
        self.connect_baud_rate = self.board_profile.baud_rate
        self._sample_rate_verify_worker = None
        self._auto_verify_pending = False

        # 板卡参数(连接后填充)
        self.sampling_rate = self.board_profile.expected_sample_rate
        self.eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        self.timestamp_channel = BoardShim.get_timestamp_channel(BOARD_ID)   # 板卡时间戳通道(秒)
        self.ch1_index = self.eeg_channels[0]                        # 通道1在数据矩阵中的行号
        self.ch2_index = self.eeg_channels[1]                        # 通道2（专注指数默认用 CH1+CH2）
        self.focus_window_size = int(self.sampling_rate * FOCUS_WINDOW_SECONDS)
        self.focus_psd_size = DataFilter.get_nearest_power_of_two(self.sampling_rate)

        # 滚动缓冲区: 实际维护的原始数据比显示区多一段预热区(pad),
        # 滤波后只取内侧 buffer_size 个点显示, 把边缘振铃挡在屏幕外。
        self.buffer_size = int(self.sampling_rate * PLOT_SECONDS)
        self.pad_size = int(self.sampling_rate * PAD_SECONDS)
        self.raw_size = self._scope_raw_size()
        self.wearing_window_size = int(self.sampling_rate * WEARING_WINDOW_SECONDS)
        # 每个示波器各自维护缓冲, 见 _build_scope_panel / self.scopes

        self._build_menu()
        self._build_ui()
        self._subject_cue_window = None

        # 定时器: 周期性从板卡拉取数据并刷新界面
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)

        # 阻抗检测定时器
        self.imp_timer = QtCore.QTimer(self)
        self.imp_timer.timeout.connect(self.update_impedance)
        self.imp_window_size = int(self.sampling_rate * IMP_WINDOW_SECONDS)

        # 专注指标定时器
        self.focus2_timer = QtCore.QTimer()
        self.focus2_timer.timeout.connect(self.update_focus2)

        # 实时佩戴检测页定时器
        self.wearing_exp_timer = QtCore.QTimer()
        self.wearing_exp_timer.timeout.connect(self.update_wearing_exp)

        # 采集范式页定时器
        self.acquire_timer = QtCore.QTimer()
        self.acquire_timer.timeout.connect(self.update_acquire)

        # 模型测试页定时器
        self.test_timer = QtCore.QTimer()
        self.test_timer.timeout.connect(self.update_test)

        self.validate_timer = QtCore.QTimer()
        self.validate_timer.timeout.connect(self.update_validate)

        # ERD/ERS 检测页定时器
        self.erders_timer = QtCore.QTimer()
        self.erders_timer.timeout.connect(self.update_erders)

        self._paradigm_space_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        self._paradigm_space_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self._paradigm_space_shortcut.activated.connect(self._on_paradigm_space_continue)
        self._paradigm_space_shortcut.setEnabled(False)

        self.refresh_ports()
        self._load_ui_settings()
        self._sync_hw_setup_btn_state()

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._ensure_window_fits_screen)

    def _ensure_window_fits_screen(self):
        """窗口尺寸不超过当前屏幕可用区域, 避免 Qt 反复 setGeometry 导致闪烁。"""
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None or self.isMaximized():
            return
        avail = screen.availableGeometry()
        fg = self.frameGeometry()
        nw = min(fg.width(), avail.width())
        nh = min(fg.height(), avail.height())
        nx, ny = fg.x(), fg.y()
        if nx < avail.x():
            nx = avail.x()
        if ny < avail.y():
            ny = avail.y()
        if nx + nw > avail.right() + 1:
            nx = max(avail.x(), avail.right() + 1 - nw)
        if ny + nh > avail.bottom() + 1:
            ny = max(avail.y(), avail.bottom() + 1 - nh)
        if (nw, nh, nx, ny) != (fg.width(), fg.height(), fg.x(), fg.y()):
            self.setGeometry(nx, ny, nw, nh)

    # ---------------- 界面 ----------------
    def _build_menu(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("帮助(&H)")
        guide_act = help_menu.addAction("操作手册(&U)…")
        guide_act.triggered.connect(self._open_user_guide)
        help_menu.addSeparator()
        about_act = help_menu.addAction(f"关于 {APP_NAME}(&A)…")
        about_act.triggered.connect(self._show_about_dialog)
        notice_act = help_menu.addAction("免责声明(&N)…")
        notice_act.triggered.connect(self._show_notice_dialog)
        if is_debug_mode():
            debug_menu = menubar.addMenu("调试(&G)")
            debug_menu.addAction("调试模式已启用（虚拟端口可见）").setEnabled(False)

    def _open_user_guide(self):
        path = os.path.join(project_root_from(__file__), "docs", "USER_GUIDE.md")
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "操作手册", f"未找到手册文件:\n{path}")
            return
        url = QtCore.QUrl.fromLocalFile(path)
        if not QtGui.QDesktopServices.openUrl(url):
            # 回退：用记事本/对话框展示
            try:
                text = open(path, encoding="utf-8").read()
            except OSError as e:
                QtWidgets.QMessageBox.warning(self, "操作手册", str(e))
                return
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(f"{APP_NAME} 操作手册")
            dlg.resize(720, 560)
            v = QtWidgets.QVBoxLayout(dlg)
            browser = QtWidgets.QPlainTextEdit()
            browser.setReadOnly(True)
            browser.setPlainText(text)
            v.addWidget(browser)
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            btns.rejected.connect(dlg.reject)
            btns.accepted.connect(dlg.accept)
            btns.clicked.connect(dlg.accept)
            v.addWidget(btns)
            dlg.exec_()

    def _ui_settings(self) -> QtCore.QSettings:
        return QtCore.QSettings(APP_ORG, APP_NAME)

    def _load_ui_settings(self):
        s = self._ui_settings()
        baud = int(s.value("baud_rate", 115200))
        idx = self.baud_combo.findData(baud)
        if idx >= 0:
            self.baud_combo.blockSignals(True)
            self.baud_combo.setCurrentIndex(idx)
            self.baud_combo.blockSignals(False)
            self._on_baud_combo_changed()
        port = s.value("port", "")
        if port:
            pidx = self.port_combo.findData(port)
            if pidx >= 0:
                self.port_combo.setCurrentIndex(pidx)
        if hasattr(self, "hw_checkbox"):
            self.hw_checkbox.setChecked(s.value("apply_hw", APPLY_HW_SETTINGS, type=bool))
        if hasattr(self, "bandpass_combo"):
            bp = s.value("bandpass", "1-50 Hz")
            if self.bandpass_combo.findText(bp) >= 0:
                self.bandpass_combo.setCurrentText(bp)
            notch = s.value("notch", "50 Hz")
            if self.notch_combo.findText(notch) >= 0:
                self.notch_combo.setCurrentText(notch)
        if hasattr(self, "paradigm_combo"):
            pid = s.value("paradigm_id", "")
            if pid:
                for i in range(self.paradigm_combo.count()):
                    if self.paradigm_combo.itemData(i) == pid:
                        self.paradigm_combo.setCurrentIndex(i)
                        break

    def _save_ui_settings(self):
        s = self._ui_settings()
        if hasattr(self, "baud_combo"):
            data = self.baud_combo.currentData()
            if data is not None:
                s.setValue("baud_rate", int(data))
        if hasattr(self, "port_combo"):
            port = self.port_combo.currentData()
            if port and port != DEBUG_VIRTUAL_PORT:
                s.setValue("port", port)
        if hasattr(self, "hw_checkbox"):
            s.setValue("apply_hw", self.hw_checkbox.isChecked())
        if hasattr(self, "bandpass_combo"):
            s.setValue("bandpass", self.bandpass_combo.currentText())
            s.setValue("notch", self.notch_combo.currentText())
        if hasattr(self, "paradigm_combo"):
            pid = self.paradigm_combo.currentData()
            if pid:
                s.setValue("paradigm_id", str(pid))

    def _confirm_stop_activity(self, title: str, message: str) -> bool:
        reply = QtWidgets.QMessageBox.question(
            self,
            title,
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return reply == QtWidgets.QMessageBox.Yes

    def _session_busy(self) -> bool:
        return bool(
            getattr(self, "acquire_running", False)
            or getattr(self, "test_running", False)
            or getattr(self, "validate_running", False)
            or getattr(self, "erders_running", False)
            or getattr(self, "focus2_running", False)
            or getattr(self, "wearing_exp_running", False)
            or getattr(self, "imp_running", False)
            or getattr(self, "is_streaming", False)
        )

    def _open_subject_cue_window(self):
        if self._subject_cue_window is None:
            self._subject_cue_window = SubjectCueWindow(self)
            self._subject_cue_window.destroyed.connect(self._on_subject_cue_destroyed)
        self._subject_cue_window.showFullScreen()
        self._sync_subject_cue_window()
        self._subject_cue_window.raise_()
        self._subject_cue_window.activateWindow()
        self._subject_cue_window.setFocus(QtCore.Qt.ActiveWindowFocusReason)

    def _on_subject_cue_destroyed(self, *_args):
        self._subject_cue_window = None

    def _on_subject_cue_closed(self, *_args):
        self._subject_cue_window = None

    def _sync_subject_cue_window(self):
        win = getattr(self, "_subject_cue_window", None)
        if win is None or not win.isVisible():
            return
        if not hasattr(self, "acquire_cue_label"):
            return
        full = getattr(self, "_current_cue_full_pixmap", None)
        win.sync_from(full, self.acquire_cue_label, self.acquire_hint_label)

    def _show_about_dialog(self):
        tags = hashtags_line("#")
        QtWidgets.QMessageBox.about(
            self,
            f"关于 {APP_NAME}",
            (
                f"<h3>{APP_NAME}</h3>"
                f"<p>{APP_NAME_CN}</p>"
                f"<p>{APP_TAGLINE}</p>"
                f"<p>版本 <b>{__version__}</b> · 许可 {LICENSE_SPDX}</p>"
                f"<p><a href=\"{GITHUB_URL}\">{GITHUB_URL}</a></p>"
                f"<p style=\"color:#555;\">{tags}</p>"
                "<p>研究 / 教育用途 · <b>非医疗器械</b>。详见仓库 NOTICE.md 与 LICENSE。</p>"
                "<p>兼容 OpenBCI Cyton + Daisy；商标归各自权利人所有。</p>"
            ),
        )

    def _show_notice_dialog(self):
        QtWidgets.QMessageBox.information(
            self,
            "免责声明",
            (
                f"{APP_NAME} 是开源研究/教育软件，不是医疗器械。\n\n"
                "不得用于临床诊断或治疗决策。软件按「现状」提供，作者与贡献者"
                "不承担因使用本软件产生的任何责任。\n\n"
                "完整条款见仓库 NOTICE.md 与 LICENSE（MIT）。"
            ),
        )

    def _build_ui(self):
        central = QtWidgets.QWidget()
        central.setObjectName("centralRoot")
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 顶部标题栏 (品牌区)
        header = QtWidgets.QFrame()
        header.setObjectName("appHeader")
        header.setFixedHeight(56)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(14, 6, 14, 6)
        title_box = QtWidgets.QHBoxLayout()
        title_box.setSpacing(12)
        icon_lbl = QtWidgets.QLabel()
        icon_pix = load_app_icon().pixmap(36, 36)
        if not icon_pix.isNull():
            icon_lbl.setPixmap(icon_pix)
            title_box.addWidget(icon_lbl)
        text_box = QtWidgets.QVBoxLayout()
        text_box.setSpacing(1)
        app_title = QtWidgets.QLabel(APP_NAME)
        app_title.setObjectName("appTitle")
        app_subtitle = QtWidgets.QLabel("OpenBCI Cyton + Daisy · 16 通道运动执行工作台")
        app_subtitle.setObjectName("appSubtitle")
        text_box.addWidget(app_title)
        text_box.addWidget(app_subtitle)
        title_box.addLayout(text_box)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        self.header_badge = QtWidgets.QLabel()
        self.header_badge.setObjectName("appBadge")
        header_layout.addWidget(self.header_badge)
        self._update_rate_badge()
        layout.addWidget(header)

        # 设备条: 连接控件 + 状态/滤波合并为一张卡片(少占纵向空间)
        device_bar = QtWidgets.QFrame()
        device_bar.setObjectName("deviceBar")
        device_v = QtWidgets.QVBoxLayout(device_bar)
        device_v.setContentsMargins(16, 12, 16, 12)
        device_v.setSpacing(10)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(QtWidgets.QLabel("串口:"))
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setMinimumWidth(180)
        row1.addWidget(self.port_combo)

        self.refresh_btn = QtWidgets.QPushButton("刷新")
        self.refresh_btn.setObjectName("btnSecondary")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        row1.addWidget(self.refresh_btn)

        self.connect_btn = QtWidgets.QPushButton("连接")
        self.connect_btn.setObjectName("btnPrimary")
        self.connect_btn.clicked.connect(self.toggle_connect)
        row1.addWidget(self.connect_btn)

        row1.addWidget(QtWidgets.QLabel("波特率:"))
        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.addItem("115200", 115200)
        self.baud_combo.addItem("921600", 921600)
        self.baud_combo.setCurrentIndex(0)
        self.baud_combo.setToolTip(
            "须与板卡 SW4 / 蓝牙适配器当前波特率一致。"
            "默认 115200（BrainFlow · 通道 125 Hz）；921600 走 pyserial（通道 500 Hz）。"
        )
        self.baud_combo.currentIndexChanged.connect(self._on_baud_combo_changed)
        row1.addWidget(self.baud_combo)

        self.hw_setup_btn = QtWidgets.QPushButton("采样配置…")
        self.hw_setup_btn.setObjectName("btnSecondary")
        self.hw_setup_btn.setToolTip(
            "外部完成波特率/采样率设置后，在此向导中实测确认并应用。"
        )
        self.hw_setup_btn.clicked.connect(self._open_hardware_setup_wizard)
        row1.addWidget(self.hw_setup_btn)

        self.hw_checkbox = QtWidgets.QCheckBox("应用硬件设置")
        self.hw_checkbox.setChecked(APPLY_HW_SETTINGS)
        hw_row = QtWidgets.QHBoxLayout()
        hw_row.setContentsMargins(8, 4, 8, 4)
        hw_row.setSpacing(6)
        hw_row.addWidget(self.hw_checkbox)
        hw_row.addWidget(self._make_info_tip(
            "- 连接成功后写入 OpenBCI 硬件寄存器\n"
            "- **增益**: x24 · **输入**: Normal\n"
            "- **Bias**: 计入 · **SRB2**: Off · **SRB1**: On",
        ))
        hw_wrap = QtWidgets.QWidget()
        hw_wrap.setObjectName("deviceInline")
        hw_wrap.setLayout(hw_row)
        row1.addWidget(hw_wrap)

        self.connect_progress_slot = QtWidgets.QWidget()
        self.connect_progress_slot.setObjectName("connectProgressSlot")
        self.connect_progress_slot.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred,
        )
        connect_status = QtWidgets.QHBoxLayout(self.connect_progress_slot)
        connect_status.setContentsMargins(10, 6, 4, 6)
        connect_status.setSpacing(12)
        self.connect_progress_label = QtWidgets.QLabel("")
        self.connect_progress_label.setStyleSheet(
            f"color: {MEDICAL_PALETTE['muted']}; font-size: 12px; background: transparent;"
        )
        self.connect_progress_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.connect_progress_label.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred,
        )
        connect_status.addWidget(self.connect_progress_label)
        self.connect_progress_bar = ImpSmoothProgressBar(stretch=True)
        connect_status.addWidget(self.connect_progress_bar, 1)
        row1.addWidget(self.connect_progress_slot, 1)
        device_v.addLayout(row1)

        divider = QtWidgets.QFrame()
        divider.setObjectName("deviceBarDivider")
        divider.setFrameShape(QtWidgets.QFrame.NoFrame)
        device_v.addWidget(divider)

        row2 = QtWidgets.QHBoxLayout()
        row2.setContentsMargins(0, 2, 0, 2)
        row2.setSpacing(10)
        status_wrap = QtWidgets.QHBoxLayout()
        status_wrap.setContentsMargins(0, 0, 0, 0)
        status_wrap.setSpacing(8)
        self.status_dot = QtWidgets.QLabel()
        self.status_dot.setObjectName("statusDot")
        self.status_dot.setFixedSize(10, 10)
        status_wrap.addWidget(self.status_dot)
        self.status_label = QtWidgets.QLabel("状态: 未连接")
        self.status_label.setObjectName("statusText")
        status_wrap.addWidget(self.status_label)
        row2.addLayout(status_wrap)
        row2.addStretch()
        section = QtWidgets.QLabel("滤波 / FFT")
        section.setObjectName("deviceBarSection")
        row2.addWidget(section)
        self._add_global_filter_controls(row2)
        device_v.addLayout(row2)
        layout.addWidget(device_bar)
        self._set_app_status("状态: 未连接", tone="warn")

        # 选项卡 (各页包滚动区, 防止未激活 Tab 的最小高度撑大主窗口)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding,
        )
        layout.addWidget(self.tabs, 1)
        self.tabs.addTab(self._wrap_scroll_tab(self._build_monitor_tab()), "双通道对比")
        self.tabs.addTab(self._wrap_scroll_tab(self._build_impedance_tab()), "阻抗检测")
        self.tabs.addTab(self._wrap_scroll_tab(self._build_focus_enhanced_tab()), "专注指标")
        self.tabs.addTab(self._wrap_scroll_tab(self._build_wearing_experiment_tab()), "实时佩戴检测")
        self.tabs.addTab(self._wrap_scroll_tab(self._build_validate_tab()), "信号校验")
        self.tabs.addTab(self._wrap_scroll_tab(self._build_acquire_tab()), "采集范式")
        self.tabs.addTab(self._wrap_scroll_tab(self._build_erders_tab()), "ERD/ERS 检测")

        # x 轴时间序列(固定)
        self.time_axis = np.linspace(-PLOT_SECONDS, 0, self.buffer_size)
        self.wearing_time_axis = np.linspace(-WEARING_WINDOW_SECONDS, 0, self.wearing_window_size)

        # FFT 窗/频率轴（随通道采样率重建，见 _rebuild_fft_state）
        self._rebuild_fft_state()

    def _set_app_status(self, text, tone=None):
        """更新顶部状态文案与色点(仅外观, 不改业务逻辑)。"""
        self.status_label.setText(text)
        if not hasattr(self, "status_dot") or self.status_dot is None:
            return
        if tone is None:
            t = text or ""
            if any(k in t for k in ("失败", "未通过", "错误")):
                tone = "danger"
            elif any(k in t for k in ("未连接", "请先", "请选择")):
                tone = "warn"
            elif any(k in t for k in ("采集中", "检测中", "录制", "预热", "连接中", "启动")):
                tone = "primary"
            elif any(k in t for k in ("已连接", "已停止", "已保存", "确认", "完成")):
                tone = "ok"
            else:
                tone = "muted"
        color = {
            "ok": MEDICAL_PALETTE["ok"],
            "warn": MEDICAL_PALETTE["warn"],
            "danger": MEDICAL_PALETTE["danger"],
            "primary": MEDICAL_PALETTE["primary"],
            "muted": MEDICAL_PALETTE["muted"],
        }.get(tone, MEDICAL_PALETTE["muted"])
        self.status_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 5px;"
        )
    def _make_info_tip(self, text):
        """圆形 ⓘ 提示按钮, 悬停显示风格化说明卡片。"""
        return InfoTipButton(text, self)

    def _wrap_scroll_tab(self, widget):
        """Tab 内容外包 QScrollArea, 仅按当前视口约束最小高度。"""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Ignored,
        )
        scroll.setWidget(widget)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding,
        )
        return scroll

    def _add_page_title(self, layout, title, info_text=None, object_name=None, font_size=14):
        """紧凑页标题 + 可选 ⓘ 说明 (不占正文空间)。若含说明则返回 InfoTipButton。"""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        lbl = QtWidgets.QLabel(title)
        if object_name:
            lbl.setObjectName(object_name)
        font = lbl.font()
        font.setPointSize(font_size)
        font.setBold(True)
        lbl.setFont(font)
        row.addWidget(lbl)
        tip_btn = None
        if info_text:
            tip_btn = self._make_info_tip(info_text)
            row.addWidget(tip_btn)
        row.addStretch()
        layout.addLayout(row)
        return tip_btn

    def _add_global_filter_controls(self, row):
        """在状态栏右侧添加全局带通/陷波与 FFT 显示设置。"""
        row.addWidget(QtWidgets.QLabel("带通:"))
        self.bandpass_combo = QtWidgets.QComboBox()
        self.bandpass_combo.addItems(list(BANDPASS_OPTIONS.keys()))
        self.bandpass_combo.setCurrentText("1-50 Hz")
        self.bandpass_combo.setToolTip("全局带通: 双通道对比 + 佩戴检测波形 + 采集/测试/校验 CSV 落盘")
        self.bandpass_combo.currentIndexChanged.connect(self._on_global_filter_changed)
        row.addWidget(self.bandpass_combo)

        row.addWidget(QtWidgets.QLabel("陷波:"))
        self.notch_combo = QtWidgets.QComboBox()
        self.notch_combo.addItems(list(NOTCH_OPTIONS.keys()))
        self.notch_combo.setCurrentText("50 Hz")
        self.notch_combo.setToolTip("全局陷波: 双通道对比 + 佩戴检测波形 + 采集/测试/校验 CSV 落盘")
        self.notch_combo.currentIndexChanged.connect(self._on_global_filter_changed)
        row.addWidget(self.notch_combo)

        row.addSpacing(10)
        row.addWidget(QtWidgets.QLabel("FFT 最大频率:"))
        self.fft_maxfreq_combo = QtWidgets.QComboBox()
        self.fft_maxfreq_combo.addItems([f"{f} Hz" for f in FFT_MAXFREQ_OPTIONS])
        self.fft_maxfreq_combo.setCurrentText("60 Hz")
        self.fft_maxfreq_combo.currentIndexChanged.connect(self.update_fft_xrange)
        row.addWidget(self.fft_maxfreq_combo)
        self._fft_maxfreq_options = fft_maxfreq_options_for_sample_rate(self.sampling_rate)

        self.fft_log_checkbox = QtWidgets.QCheckBox("FFT 对数 Y 轴")
        self.fft_log_checkbox.setChecked(True)
        self.fft_log_checkbox.stateChanged.connect(self.update_fft_logmode)
        row.addWidget(self.fft_log_checkbox)

    def _build_monitor_tab(self):
        pg.setConfigOptions(antialias=True)
        self.scopes = []

        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 控制条贴住波形区: 开始采集 + 通道选择
        toolbar = QtWidgets.QFrame()
        toolbar.setObjectName("monitorToolbar")
        row = QtWidgets.QHBoxLayout(toolbar)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        self.start_btn = QtWidgets.QPushButton("开始采集")
        self.start_btn.setObjectName("btnPrimary")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.toggle_stream)
        row.addWidget(self.start_btn)
        row.addSpacing(12)

        for i in range(NUM_SCOPES):
            self.scopes.append(self._create_scope_controls(i))
            scope = self.scopes[i]
            color = SCOPE_COLORS[i % len(SCOPE_COLORS)]
            dot = QtWidgets.QLabel()
            dot.setObjectName("scopeColorDot")
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(
                f"background-color: rgb({color[0]}, {color[1]}, {color[2]});"
                "border-radius: 5px;"
            )
            row.addWidget(dot)
            row.addWidget(QtWidgets.QLabel(f"示波器 {i + 1}"))
            row.addWidget(QtWidgets.QLabel("通道:"))
            row.addWidget(scope["combo"])
            row.addWidget(scope["value_label"])
            if i < NUM_SCOPES - 1:
                row.addSpacing(14)
        row.addStretch()
        layout.addWidget(toolbar)

        self.monitor_empty_hint = QtWidgets.QLabel(
            "连接设备后点击「开始采集」，即可在下方查看时域波形与 FFT 频谱"
        )
        self.monitor_empty_hint.setObjectName("emptyHint")
        self.monitor_empty_hint.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.monitor_empty_hint)

        for i in range(NUM_SCOPES):
            layout.addWidget(self._build_scope_panel(i), 1)

        return tab

    def _create_scope_controls(self, idx):
        combo = QtWidgets.QComboBox()
        combo.addItems([f"CH{c + 1}" for c in range(len(self.eeg_channels))])
        combo.setCurrentIndex(min(idx, len(self.eeg_channels) - 1))
        combo.setMinimumWidth(72)
        combo.currentIndexChanged.connect(lambda _, i=idx: self._reset_scope(i))

        value_label = QtWidgets.QLabel("--- uV")
        f = value_label.font()
        f.setPointSize(14)
        f.setBold(True)
        value_label.setFont(f)
        color = SCOPE_COLORS[idx % len(SCOPE_COLORS)]
        value_label.setStyleSheet(
            f"color: rgb({color[0]}, {color[1]}, {color[2]}); background: transparent;"
        )
        return {
            "combo": combo,
            "value_label": value_label,
            "raw_buffer": np.zeros(self.raw_size),
            "has_data": False,
            "fill_count": 0,
        }

    def _build_scope_panel(self, idx):
        """构建示波器面板: 时域波形 + FFT (通道选择在顶部工具行)。"""
        scope = self.scopes[idx]
        color = SCOPE_COLORS[idx % len(SCOPE_COLORS)]
        box = QtWidgets.QGroupBox(f"示波器 {idx + 1}")
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(6, 8, 6, 6)
        v.setSpacing(4)

        plots = QtWidgets.QHBoxLayout()
        plots.setSpacing(6)
        time_plot = pg.PlotWidget()
        time_plot.setBackground("w")
        time_plot.setTitle("时域波形")
        time_plot.setLabel("left", "电压", units="uV")
        time_plot.setLabel("bottom", "时间", units="s")
        time_plot.showGrid(x=True, y=True, alpha=0.18)
        time_curve = time_plot.plot(pen=pg.mkPen(color=color, width=1.5))
        plots.addWidget(time_plot, 3)

        fft_plot = pg.PlotWidget()
        fft_plot.setBackground("w")
        fft_plot.setTitle("FFT 频谱")
        fft_plot.setLabel("left", "振幅", units="uV")
        fft_plot.setLabel("bottom", "频率", units="Hz")
        fft_plot.showGrid(x=True, y=True, alpha=0.18)
        fft_plot.setLogMode(False, self.fft_log_checkbox.isChecked())
        fft_plot.getPlotItem().enableAutoRange(x=False, y=False)
        fft_plot.setXRange(0, FFT_MAXFREQ_OPTIONS[self.fft_maxfreq_combo.currentIndex()])
        fft_curve = fft_plot.plot(pen=pg.mkPen(color=color, width=1.5))
        plots.addWidget(fft_plot, 2)
        v.addLayout(plots)

        scope["time_curve"] = time_curve
        scope["fft_plot"] = fft_plot
        scope["fft_curve"] = fft_curve
        return box

    def _set_monitor_empty_hint_visible(self, visible):
        hint = getattr(self, "monitor_empty_hint", None)
        if hint is not None:
            hint.setVisible(bool(visible))
    def _reset_scope(self, idx):
        """切换通道时重置该示波器的缓冲, 避免新旧通道数据混在一起。"""
        scope = self.scopes[idx]
        scope["raw_buffer"] = np.zeros(self.raw_size)
        scope["has_data"] = False
        scope["fill_count"] = 0
        scope["value_label"].setText("--- uV")

    def _imp_tier_from_z(self, z_kohm):
        """将阻抗值映射为 UI 质量分级。"""
        if z_kohm <= 10:
            return "excellent"
        if z_kohm <= 50:
            return "good"
        if z_kohm <= 100:
            return "acceptable"
        if z_kohm <= 200:
            return "high"
        return "poor"

    @staticmethod
    def _imp_tier_colors(tier):
        """返回 (左边框色, 卡片底色, 文字色, 状态徽章底色)。"""
        p = MEDICAL_PALETTE
        return {
            "idle": (p["border"], p["surface"], p["muted"], p["primary_light"]),
            "measuring": (p["primary"], "#f0f6fd", p["primary"], p["primary_light"]),
            "open": (p["danger"], "#fef2f2", p["danger"], "#fde8e8"),
            "excellent": (p["ok"], "#f0fdf4", p["ok"], "#dcfce7"),
            "good": ("#2da44e", "#f0fdf4", "#2da44e", "#dcfce7"),
            "acceptable": (p["warn"], "#fffbeb", p["warn"], "#fef3c7"),
            "high": (p["danger"], "#fef2f2", p["danger"], "#fde8e8"),
            "poor": ("#a40e26", "#fef2f2", "#a40e26", "#fde8e8"),
        }.get(tier, (p["border"], p["surface"], p["muted"], p["primary_light"]))

    def _imp_card_stylesheet(self, tier):
        accent, bg, text, badge_bg = self._imp_tier_colors(tier)
        border = MEDICAL_PALETTE["border"]
        return (
            f"QFrame#impChannelCard {{"
            f"background-color: {bg}; border: 1px solid {border};"
            f"border-left: 4px solid {accent};"
            f"border-top-left-radius: 0px; border-bottom-left-radius: 0px;"
            f"border-top-right-radius: 8px; border-bottom-right-radius: 8px;"
            f"}}"
            f"QLabel#impChBadge {{ background-color: {badge_bg}; color: {text};"
            f"border: 1px solid {accent}; }}"
            f"QLabel#impValue {{ color: {text}; }}"
            f"QLabel#impStatus {{ color: {text}; background-color: {badge_bg}; }}"
        )

    def _build_imp_channel_card(self, ch):
        frame = QtWidgets.QFrame()
        frame.setObjectName("impChannelCard")
        frame.setFixedHeight(72)
        row = QtWidgets.QHBoxLayout(frame)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(12)

        badge = QtWidgets.QLabel(f"{ch + 1:02d}")
        badge.setObjectName("impChBadge")
        badge.setFixedSize(40, 40)
        badge.setAlignment(QtCore.Qt.AlignCenter)
        row.addWidget(badge)

        value_col = QtWidgets.QVBoxLayout()
        value_col.setSpacing(0)
        ch_num = QtWidgets.QLabel(f"通道 {ch + 1}")
        ch_num.setObjectName("impChNum")
        value_label = QtWidgets.QLabel("---")
        value_label.setObjectName("impValue")
        unit_label = QtWidgets.QLabel("kΩ")
        unit_label.setObjectName("impUnit")
        value_col.addWidget(ch_num)
        value_col.addWidget(value_label)
        value_col.addWidget(unit_label)
        row.addLayout(value_col, 1)

        status_label = QtWidgets.QLabel("待检测")
        status_label.setObjectName("impStatus")
        status_label.setAlignment(QtCore.Qt.AlignCenter)
        row.addWidget(status_label)

        frame.setStyleSheet(self._imp_card_stylesheet("idle"))
        return {
            "frame": frame,
            "value_label": value_label,
            "status_label": status_label,
        }

    def _build_wearing_channel_card(self, ch):
        frame = QtWidgets.QFrame()
        frame.setObjectName("wearingChCard")
        frame.setFixedHeight(72)
        row = QtWidgets.QHBoxLayout(frame)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        badge = QtWidgets.QLabel(f"{ch + 1}")
        badge.setObjectName("wearingChBadge")
        badge.setFixedSize(32, 32)
        badge.setAlignment(QtCore.Qt.AlignCenter)
        color = WEARING_COLORS[ch % len(WEARING_COLORS)]
        badge.setStyleSheet(
            f"QLabel#wearingChBadge {{ background-color: rgb{color}; }}"
        )
        row.addWidget(badge)

        mid = QtWidgets.QVBoxLayout()
        mid.setSpacing(0)
        score_lbl = QtWidgets.QLabel("---")
        score_lbl.setObjectName("wearingChScore")
        meta_lbl = QtWidgets.QLabel("std:---  50Hz:---")
        meta_lbl.setObjectName("wearingChMeta")
        mid.addWidget(score_lbl)
        mid.addWidget(meta_lbl)
        row.addLayout(mid, 1)

        status_lbl = QtWidgets.QLabel("---")
        status_lbl.setObjectName("wearingChStatus")
        status_lbl.setAlignment(QtCore.Qt.AlignCenter)
        row.addWidget(status_lbl)

        card = {
            "frame": frame,
            "score_label": score_lbl,
            "meta_label": meta_lbl,
            "status_label": status_lbl,
        }
        self._set_wearing_channel_ui(card, ch, "idle")
        return card

    def _wearing_channel_tier(self, state):
        return {
            "ok": ("#1a7f37", "#dcfce7", "#1a7f37"),
            "bad": ("#d1242f", "#fde8e8", "#d1242f"),
            "idle": ("#64748b", "#f6f8fa", "#d3dde8"),
            "measuring": ("#bf8700", "#fff8e6", "#bf8700"),
        }.get(state, ("#64748b", "#f6f8fa", "#d3dde8"))

    def _set_wearing_channel_ui(self, card, ch, state, info=None):
        border_color, bg, text_color = self._wearing_channel_tier(state)
        card["frame"].setStyleSheet(
            f"QFrame#wearingChCard {{ background-color: {bg}; border: 1px solid #d3dde8; "
            f"border-left: 4px solid {border_color}; "
            f"border-top-left-radius: 0px; border-bottom-left-radius: 0px; "
            f"border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}"
        )
        if state == "idle":
            card["score_label"].setText("SQI ---")
            card["meta_label"].setText("std:---  50Hz:---")
            card["status_label"].setText("待检测")
        elif state == "measuring":
            card["score_label"].setText("SQI ...")
            card["meta_label"].setText("数据累积中")
            card["status_label"].setText("...")
        else:
            worn = info.get("stable_worn", info.get("worn", False))
            score = info.get("stable_score", info.get("score", 0))
            card["score_label"].setText(f"SQI {score:.1f}")
            card["meta_label"].setText(
                f"std:{info.get('std', 0):>5.1f}  50Hz:{info.get('line_ratio', 0):.3f}"
            )
            card["status_label"].setText("OK" if worn else "BAD")
        card["status_label"].setStyleSheet(
            f"QLabel#wearingChStatus {{ color: {text_color}; background: transparent; }}"
        )

    def _update_wearing_exp_channels_ui(self, channels_info=None, measuring=False):
        if measuring:
            for ch, card in enumerate(self.wearing_exp_channel_cards):
                self._set_wearing_channel_ui(card, ch, "measuring")
            self.wearing_exp_head_map.set_channels([None] * 16)
            return
        if channels_info is None:
            for ch, card in enumerate(self.wearing_exp_channel_cards):
                self._set_wearing_channel_ui(card, ch, "idle")
            self.wearing_exp_head_map.set_channels([None] * 16)
            return
        for idx, info in enumerate(channels_info):
            worn = info.get("stable_worn", info.get("worn", False))
            state = "ok" if worn else "bad"
            self._set_wearing_channel_ui(self.wearing_exp_channel_cards[idx], idx, state, info)
        self.wearing_exp_head_map.set_channels(channels_info)

    def _set_imp_channel_ui(self, ch, tier, value_text, status_text):
        card = self.imp_cards[ch]
        card["value_label"].setText(value_text)
        card["status_label"].setText(status_text)
        card["frame"].setStyleSheet(self._imp_card_stylesheet(tier))

    def _reset_imp_channels(self):
        for ch in range(len(self.eeg_channels)):
            self._set_imp_channel_ui(ch, "idle", "---", "待检测")
        self.imp_summary_label.setText("待检测 · 0/16 通道合格")
        self._update_imp_progress(0)

    def _update_imp_progress(self, percent, indeterminate=False, success=False):
        if hasattr(self, "imp_progress_bar"):
            self.imp_progress_bar.set_progress(percent, indeterminate, success)

    def _parse_fraction_progress_msg(self, msg):
        """从「X/Y」文案解析进度百分比。"""
        match = re.search(r"(\d+)\s*/\s*(\d+)", msg)
        if match:
            cur, tot = int(match.group(1)), int(match.group(2))
            if tot > 0:
                return cur / tot * 100.0
        return None

    def _update_connect_progress(self, percent, indeterminate=False):
        if hasattr(self, "connect_progress_bar"):
            self.connect_progress_bar.set_progress(percent, indeterminate)

    def _show_connect_progress(self, msg, indeterminate=True):
        self.connect_progress_label.setText(msg)
        pct = self._parse_fraction_progress_msg(msg)
        if pct is not None:
            self._update_connect_progress(pct)
        else:
            self._update_connect_progress(0, indeterminate=indeterminate)

    def _hide_connect_progress(self):
        self.connect_progress_label.setText("")
        self.connect_progress_bar.reset()

    def _connect_worker_busy(self):
        return self._connect_board_worker is not None and self._connect_board_worker.isRunning()

    def _on_connect_progress(self, msg):
        self._set_app_status(f"状态: {msg}")
        self._show_connect_progress(msg, indeterminate=("/" not in msg))

    def _on_connect_ok(self, board, hw_msg):
        self._connect_board_worker = None
        port = self._connect_port
        self.board = board
        self.is_debug_virtual = False
        self.is_connected = True
        self.connect_btn.setText("断开")
        self.connect_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.imp_btn.setEnabled(True)
        self.focus2_btn.setEnabled(True)
        self.focus2_calib_btn.setEnabled(True)
        self.wearing_exp_btn.setEnabled(True)
        self.acquire_btn.setEnabled(True)
        if hasattr(self, "test_acquire_btn"):
            self.test_acquire_btn.setEnabled(True)
        if hasattr(self, "erders_btn"):
            self.erders_btn.setEnabled(True)
            self.erders_status_label.setText("已连接, 可开始 ERD/ERS 实验")
        self.validate_btn.setEnabled(True)
        self.port_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.hw_checkbox.setEnabled(False)
        self.baud_combo.setEnabled(False)
        self._hide_connect_progress()
        self.board_profile = BoardProfile.profile_for_baud_rate(self.connect_baud_rate)
        self.sampling_rate = self.board_profile.expected_sample_rate
        self._rebuild_rate_dependent_state()
        self._set_app_status(
            f"状态: 已连接 {port}  |  通道 {self.sampling_rate} Hz（待实测确认）  |  "
            f"通道1 行号 {self.ch1_index}{hw_msg}"
        )
        self._sync_hw_setup_btn_state()

    def _on_connect_failed(self, err):
        self._connect_board_worker = None
        self.board = None
        self.is_debug_virtual = False
        self.is_connected = False
        self.connect_btn.setText("连接")
        self.connect_btn.setEnabled(True)
        self.port_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.hw_checkbox.setEnabled(True)
        self.baud_combo.setEnabled(True)
        self._hide_connect_progress()
        if "PermissionError" in err or "拒绝访问" in err:
            hint = (
                "（COM 口被占用：请关闭 OpenBCI GUI、串口助手等；"
                "完全退出本程序后重开；仍不行则拔插 USB 或重启电脑）"
            )
        else:
            hint = (
                f"（当前 {self.connect_baud_rate} 波特率；须与板卡 SW4/适配器一致："
                "115200→BrainFlow，921600→pyserial。默认档为 115200，不是 961200）"
            )
        self._set_app_status(f"状态: 连接失败 - {err} {hint}")

    def _on_baud_combo_changed(self):
        if self.is_connected:
            return
        data = self.baud_combo.currentData()
        if data is not None:
            self.connect_baud_rate = int(data)
            if not self.board_profile.confirmed:
                self.board_profile = BoardProfile.profile_for_baud_rate(self.connect_baud_rate)
                self.sampling_rate = self.board_profile.expected_sample_rate
                self._rebuild_rate_dependent_state()

    def _set_baud_combo_rate(self, baud_rate: int):
        baud_rate = int(baud_rate)
        for i in range(self.baud_combo.count()):
            if int(self.baud_combo.itemData(i)) == baud_rate:
                blocked = self.baud_combo.blockSignals(True)
                self.baud_combo.setCurrentIndex(i)
                self.baud_combo.blockSignals(blocked)
                break
        self.connect_baud_rate = baud_rate

    def _wearing_analysis_samples(self) -> int:
        return max(
            8,
            int(self.sampling_rate * WEARING_ANALYSIS_INTERVAL_SECONDS),
        )

    def _validate_page_intro_text(self) -> str:
        sr = int(self.sampling_rate)
        return (
            "- **取数**: 与「采集范式」相同采集路径, 连续录制固定时长 **µV** 数据\n"
            "- **滤波**: 落盘与波形均使用顶部 **带通/陷波**\n"
            f"- **横轴**: 按 **{sr} Hz** 等间隔 (样本序号)"
        )

    def _refresh_fft_maxfreq_options(self):
        """通道采样率变更后刷新 FFT 横轴上限选项。"""
        if not hasattr(self, "fft_maxfreq_combo"):
            return
        opts = fft_maxfreq_options_for_sample_rate(self.sampling_rate)
        self._fft_maxfreq_options = opts
        prev = self.fft_maxfreq_combo.currentText()
        blocked = self.fft_maxfreq_combo.blockSignals(True)
        self.fft_maxfreq_combo.clear()
        self.fft_maxfreq_combo.addItems([f"{f} Hz" for f in opts])
        if prev in [f"{f} Hz" for f in opts]:
            self.fft_maxfreq_combo.setCurrentText(prev)
        else:
            self.fft_maxfreq_combo.setCurrentText(f"{opts[-1]} Hz")
        self.fft_maxfreq_combo.blockSignals(blocked)
        self.update_fft_xrange()

    def _update_rate_badge(self):
        sr = int(self.sampling_rate)
        frame = int(getattr(self.board_profile, "frame_sample_rate", sr * 2))
        suffix = ""
        if self.board_profile.confirmed:
            suffix = f" · {self.board_profile.display_label.split('(')[0].strip()}"
        elif not self.board_profile.confirmed:
            suffix = " · 未确认"
        self.header_badge.setText(
            f"通道 {sr} Hz · 帧 {frame} Hz · 16 CH{suffix}"
        )

    def _board_profile_meta(self) -> dict:
        return dict(self.board_profile.meta_fields())

    def _rebuild_rate_dependent_state(self):
        """采样率变更后重建缓冲、时间轴与滤波相关尺寸。"""
        self.focus_window_size = int(self.sampling_rate * FOCUS_WINDOW_SECONDS)
        self.focus_psd_size = DataFilter.get_nearest_power_of_two(self.sampling_rate)
        self.buffer_size = int(self.sampling_rate * PLOT_SECONDS)
        self.pad_size = int(self.sampling_rate * PAD_SECONDS)
        self.raw_size = self._scope_raw_size()
        self.wearing_window_size = int(self.sampling_rate * WEARING_WINDOW_SECONDS)
        self.imp_window_size = int(self.sampling_rate * IMP_WINDOW_SECONDS)

        if hasattr(self, "time_axis"):
            self.time_axis = np.linspace(-PLOT_SECONDS, 0, self.buffer_size)
        if hasattr(self, "wearing_time_axis"):
            self.wearing_time_axis = np.linspace(
                -WEARING_WINDOW_SECONDS, 0, self.wearing_window_size,
            )
        self._rebuild_fft_state()
        self._refresh_fft_maxfreq_options()

        if hasattr(self, "scopes"):
            self._reset_scope_buffers()

        if hasattr(self, "wearing_exp_buffer"):
            self.wearing_exp_buffer = np.zeros((len(self.eeg_channels), 0))
        if hasattr(self, "imp_buffer"):
            self.imp_buffer = np.array([])

        if hasattr(self, "validate_info_tip") and self.validate_info_tip is not None:
            self.validate_info_tip.set_info_text(self._validate_page_intro_text())

        self._update_rate_badge()

    def _rebuild_fft_state(self):
        """重建 FFT 点数、Hamming 窗与频率轴（500Hz 通道率下自动加大 N）。"""
        fs = int(round(float(getattr(self, "sampling_rate", SAMPLE_RATE))))
        self.fft_size = fft_size_for_sample_rate(fs)
        self.fft_window = np.hamming(self.fft_size)
        self.fft_cg = float(np.mean(self.fft_window))
        self.fft_freqs = np.fft.rfftfreq(self.fft_size, 1.0 / fs)

    def _apply_board_profile(self, profile: BoardProfile, *, reset_scope_buffers: bool = True):
        self.board_profile = profile
        new_sr = profile.active_sample_rate
        prev_sr = int(round(float(getattr(self, "sampling_rate", new_sr))))
        sr_changed = abs(new_sr - prev_sr) > max(8, int(prev_sr * 0.05))
        self.sampling_rate = new_sr
        if not self.is_connected:
            self._set_baud_combo_rate(profile.baud_rate)
        if sr_changed:
            self._rebuild_rate_dependent_state()
        else:
            self._update_rate_badge()
        if reset_scope_buffers and self.is_streaming and hasattr(self, "scopes") and sr_changed:
            self._reset_scope_buffers()
        if self.is_connected and not self.is_debug_virtual and self.board is not None:
            self._set_app_status(
                f"状态: 已应用 {profile.display_label} · "
                f"通道 {self.sampling_rate} Hz（帧 {profile.frame_sample_rate} Hz）"
            )

    def _open_hardware_setup_wizard(self):
        if self.acquire_running or self.test_running:
            QtWidgets.QMessageBox.warning(
                self,
                "采样配置",
                "请先停止采集范式/模型测试，再打开采样配置向导。",
            )
            return
        if self.erders_running or self.validate_running:
            QtWidgets.QMessageBox.warning(
                self,
                "采样配置",
                "请先停止 ERD/ERS 或信号校验，再打开采样配置向导。",
            )
            return
        if self.imp_running or self._imp_worker_busy():
            QtWidgets.QMessageBox.warning(
                self,
                "采样配置",
                "请先停止阻抗检测，再打开采样配置向导。",
            )
            return
        if not self.is_connected or self.board is None:
            QtWidgets.QMessageBox.information(
                self,
                "采样配置",
                "请先连接串口，并确保工具栏「波特率」与板卡一致。",
            )
            return
        if not self.is_streaming:
            QtWidgets.QMessageBox.information(
                self,
                "采样配置",
                "请先点击「开始采集」，再打开采样配置向导进行实测确认。",
            )
            return

        profile = HardwareSetupDialog.run_wizard(
            parent=self,
            board=self.board,
            eeg_row=self.ch1_index,
            is_connected=self.is_connected,
            is_streaming=self.is_streaming,
        )
        if profile is not None:
            self._apply_board_profile(profile)

    def _sync_hw_setup_btn_state(self):
        if not hasattr(self, "hw_setup_btn"):
            return
        busy = (
            getattr(self, "acquire_running", False)
            or getattr(self, "test_running", False)
            or getattr(self, "erders_running", False)
            or getattr(self, "validate_running", False)
            or self.imp_running
            or self._imp_worker_busy()
        )
        self.hw_setup_btn.setEnabled(self.is_connected and not busy)

    def _build_impedance_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        header = QtWidgets.QFrame()
        header.setObjectName("impHeader")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        self._add_page_title(
            header_layout,
            "电极接触质量（阻抗）",
            "- 检查电极与头皮接触是否良好，便于后续采集\n"
            f"- **合格标准**: 阻抗 ≤ **{IMP_PASS_KOHM} kΩ**（含「可接受」档）\n"
            "- 测量时头皮注入微弱电流；高级命令细节见 ⓘ",
            object_name="impPageTitle",
            font_size=18,
        )
        layout.addWidget(header)

        control = QtWidgets.QFrame()
        control.setObjectName("impControlBar")
        ctrl_row = QtWidgets.QHBoxLayout(control)
        ctrl_row.setContentsMargins(14, 10, 14, 10)
        ctrl_row.setSpacing(10)
        ctrl_row.addWidget(QtWidgets.QLabel("测量端:"))
        self.imp_end_combo = QtWidgets.QComboBox()
        self.imp_end_combo.addItem("正极侧 (推荐)", True)
        self.imp_end_combo.addItem("负极侧", False)
        self.imp_end_combo.setMinimumWidth(160)
        self.imp_end_combo.setToolTip(
            "选择测量电流注入端。一般保持「正极侧」即可。\n"
            "对应 OpenBCI z 指令：P 端 z..10Z / N 端 z..01Z。"
        )
        ctrl_row.addWidget(self.imp_end_combo)
        ctrl_row.addWidget(self._make_info_tip(
            "- **合格**: ≤100 kΩ（极好 ≤10 / 良好 ≤50 / 可接受 ≤100）\n"
            "- **偏高/差**: 检查电极凝胶、接触与参考电极\n"
            "- 底层仍使用 OpenBCI 阻抗测量电流与 RMS 公式"
        ))

        self.imp_btn = QtWidgets.QPushButton("开始阻抗检测")
        self.imp_btn.setEnabled(False)
        self.imp_btn.setMinimumWidth(140)
        self.imp_btn.clicked.connect(self.toggle_impedance)
        ctrl_row.addWidget(self.imp_btn)

        self.imp_progress_slot = QtWidgets.QWidget()
        self.imp_progress_slot.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred,
        )
        imp_status = QtWidgets.QHBoxLayout(self.imp_progress_slot)
        imp_status.setContentsMargins(0, 0, 0, 0)
        imp_status.setSpacing(10)
        self.imp_summary_label = QtWidgets.QLabel("待检测 · 0/16 通道合格")
        self.imp_summary_label.setObjectName("impSummary")
        self.imp_summary_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.imp_summary_label.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred,
        )
        imp_status.addWidget(self.imp_summary_label)
        self.imp_progress_bar = ImpSmoothProgressBar(stretch=True)
        imp_status.addWidget(self.imp_progress_bar, 1)
        ctrl_row.addWidget(self.imp_progress_slot, 1)
        layout.addWidget(control)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(14)
        self.imp_cards = []
        for col_idx, (col_title, ch_start, ch_end) in enumerate([
            ("前额 / 左区 (CH 1–8)", 0, 8),
            ("顶枕 / 右区 (CH 9–16)", 8, 16),
        ]):
            group = QtWidgets.QGroupBox(col_title)
            group_layout = QtWidgets.QVBoxLayout(group)
            group_layout.setSpacing(8)
            for ch in range(ch_start, ch_end):
                card = self._build_imp_channel_card(ch)
                self.imp_cards.append(card)
                group_layout.addWidget(card["frame"])
            columns.addWidget(group, 1)
        layout.addLayout(columns, 1)

        legend = QtWidgets.QFrame()
        legend.setObjectName("impLegend")
        legend_row = QtWidgets.QHBoxLayout(legend)
        legend_row.setContentsMargins(14, 8, 14, 8)
        legend_row.setSpacing(16)
        legend_title = QtWidgets.QLabel("质量标准")
        legend_title.setObjectName("impLegendTitle")
        legend_row.addWidget(legend_title)
        tier_dot_colors = {name: self._imp_tier_colors(name)[0] for _, _, name in IMP_QUALITY_LEGEND}
        for range_text, label, tier in IMP_QUALITY_LEGEND:
            item = QtWidgets.QHBoxLayout()
            item.setSpacing(6)
            dot = QtWidgets.QLabel()
            dot.setObjectName("impLegendDot")
            dot.setStyleSheet(
                f"background-color: {tier_dot_colors[tier]}; border-radius: 5px;"
            )
            item.addWidget(dot)
            text = QtWidgets.QLabel(f"{range_text}  {label}")
            text.setObjectName("impLegendItem")
            item.addWidget(text)
            legend_row.addLayout(item)
        legend_row.addStretch()
        layout.addWidget(legend)

        return tab

    def _apply_focus2_status_badge(self, state, text):
        """state: idle / measuring / calibrating / ok / bad / stopped"""
        styles = {
            "idle": ("#666", "#f6f8fa"),
            "measuring": ("#bf8700", "#fff8e6"),
            "calibrating": ("#bf8700", "#fff8e6"),
            "ok": ("#1a7f37", "#dcfce7"),
            "bad": ("#d1242f", "#fde8e8"),
            "stopped": ("#666", "#f6f8fa"),
        }
        color, bg = styles.get(state, styles["idle"])
        self.focus2_status_label.setText(text)
        self.focus2_status_label.setStyleSheet(
            f"color: {color}; background-color: {bg};"
            f"font-size: 12px; font-weight: 600; border-radius: 12px; padding: 5px 14px;"
        )

    def _build_focus_enhanced_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        header = QtWidgets.QFrame()
        header.setObjectName("focus2Header")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        self._add_page_title(
            header_layout,
            "专注指标",
            "- **核心算法**: **1-28Hz** 带通 + 陷波 → 专注指数 β/(α×4)×100 (**CH1+CH2**)\n"
            "- **增强项**: 伪迹剔除、信号质量门控、静息个体化阈值\n"
            "- **并行指标**: Attention / Relaxation / Focus / Meditation (0-100%)",
            object_name="focus2PageTitle",
            font_size=18,
        )
        layout.addWidget(header)

        control = QtWidgets.QFrame()
        control.setObjectName("focus2ControlBar")
        ctrl_row = QtWidgets.QHBoxLayout(control)
        ctrl_row.setContentsMargins(14, 10, 14, 10)
        ctrl_row.setSpacing(10)
        self.focus2_btn = QtWidgets.QPushButton("开始专注力检测")
        self.focus2_btn.setEnabled(False)
        self.focus2_btn.setMinimumWidth(140)
        self.focus2_btn.clicked.connect(self.toggle_focus2)
        ctrl_row.addWidget(self.focus2_btn)

        self.focus2_calib_btn = QtWidgets.QPushButton("静息校准 (5s)")
        self.focus2_calib_btn.setEnabled(False)
        self.focus2_calib_btn.clicked.connect(self.start_focus2_calibration)
        ctrl_row.addWidget(self.focus2_calib_btn)

        ctrl_row.addSpacing(16)
        self.focus2_personal_cb = QtWidgets.QCheckBox("使用个体化阈值")
        self.focus2_personal_cb.setChecked(False)
        self.focus2_personal_cb.stateChanged.connect(self._on_focus2_personal_threshold_toggle)
        ctrl_row.addWidget(self.focus2_personal_cb)

        ctrl_row.addWidget(QtWidgets.QLabel("固定阈值:"))
        self.focus2_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.focus2_threshold_spin.setRange(0.0, 100.0)
        self.focus2_threshold_spin.setDecimals(1)
        self.focus2_threshold_spin.setValue(FOCUS_THRESHOLD_DEFAULT)
        self.focus2_threshold_spin.valueChanged.connect(self._on_focus2_threshold_changed)
        ctrl_row.addWidget(self.focus2_threshold_spin)
        ctrl_row.addStretch()
        layout.addWidget(control)

        compact = QtWidgets.QFrame()
        compact.setObjectName("focus2CompactBar")
        compact.setFixedHeight(50)
        compact_row = QtWidgets.QHBoxLayout(compact)
        compact_row.setContentsMargins(12, 4, 12, 4)
        compact_row.setSpacing(10)

        bt_title = QtWidgets.QLabel("专注指数")
        bt_title.setObjectName("focus2CompactLabel")
        self.focus2_bt_label = QtWidgets.QLabel("---")
        self.focus2_bt_label.setObjectName("focus2CompactValue")
        compact_row.addWidget(bt_title)
        compact_row.addWidget(self.focus2_bt_label)

        self.focus2_bt_sub = QtWidgets.QLabel("CH1:---  CH2:---")
        self.focus2_bt_sub.setObjectName("focus2HeroSub")
        compact_row.addWidget(self.focus2_bt_sub)

        self.focus2_status_label = QtWidgets.QLabel("未检测")
        self.focus2_status_label.setObjectName("focus2StatusBadge")
        self._apply_focus2_status_badge("idle", "未检测")
        compact_row.addWidget(self.focus2_status_label)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setStyleSheet(f"color: {MEDICAL_PALETTE['border']};")
        compact_row.addWidget(sep)

        self.focus2_cog_labels = {}
        for key, title, color, _ in FOCUS2_COG_SPECS:
            title_lbl = QtWidgets.QLabel(f"{title}:")
            title_lbl.setObjectName("focus2CompactLabel")
            title_lbl.setStyleSheet(f"color: {color};")
            value_lbl = QtWidgets.QLabel("---")
            value_lbl.setObjectName("focus2InlineValue")
            value_lbl.setStyleSheet(f"color: {color};")
            self.focus2_cog_labels[key] = value_lbl
            compact_row.addWidget(title_lbl)
            compact_row.addWidget(value_lbl)
        compact_row.addStretch()
        layout.addWidget(compact)

        info = QtWidgets.QFrame()
        info.setObjectName("focus2InfoBar")
        info.setFixedHeight(34)
        info_row = QtWidgets.QHBoxLayout(info)
        info_row.setContentsMargins(12, 4, 12, 4)
        info_row.setSpacing(12)
        self.focus2_calib_label = QtWidgets.QLabel("个体化阈值: 未校准 (默认 70.0)")
        self.focus2_calib_label.setObjectName("focus2StatusLine")
        self.focus2_quality_label = QtWidgets.QLabel("信号质量: ---")
        self.focus2_quality_label.setObjectName("focus2StatusLine")
        self.focus2_var_label = QtWidgets.QLabel(
            f"σ:---  Δ:---  趋势:--- ({FOCUS_PLOT_SECONDS:.0f}s)"
        )
        self.focus2_var_label.setObjectName("focus2VarLine")
        info_row.addWidget(self.focus2_calib_label)
        info_row.addWidget(self.focus2_quality_label)
        info_row.addWidget(self.focus2_var_label)
        info_row.addStretch()
        self.focus2_band_labels = []
        for name, _, _, color in FOCUS_BANDS:
            lbl = QtWidgets.QLabel(f"{name}: ---")
            lbl.setObjectName("focus2BandChip")
            lbl.setStyleSheet(f"QLabel#focus2BandChip {{ color: rgb{color}; }}")
            self.focus2_band_labels.append(lbl)
            info_row.addWidget(lbl)
        layout.addWidget(info)

        chart_card = QtWidgets.QFrame()
        chart_card.setObjectName("focus2ChartCard")
        chart_layout = QtWidgets.QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(12, 10, 12, 10)
        chart_layout.setSpacing(8)

        chart_header = QtWidgets.QHBoxLayout()
        chart_title = QtWidgets.QLabel(
            f"指数历史曲线 (专注指数 + 四指标, {FOCUS_PLOT_SECONDS:.0f}s 滑窗)"
        )
        chart_title.setObjectName("focus2ChartTitle")
        chart_header.addWidget(chart_title)
        chart_header.addStretch()
        chart_header.addWidget(QtWidgets.QLabel("显示曲线:"))
        self.focus2_plot = pg.PlotWidget()
        self.focus2_plot.setBackground("w")
        self.focus2_plot.setLabel("left", "指数", units="%")
        self.focus2_plot.setLabel("bottom", "时间", units="s")
        self.focus2_plot.showGrid(x=True, y=True, alpha=0.25)
        self.focus2_plot.setYRange(0, 100, padding=0)
        self.focus2_plot.setXRange(-FOCUS_PLOT_SECONDS, 0, padding=0)
        self.focus2_plot.getViewBox().setLimits(yMin=0, yMax=100)
        self.focus2_plot.getViewBox().enableAutoRange(axis="y", enable=False)
        self.focus2_plot.addLegend(offset=(10, 10))
        self.focus2_threshold_line = pg.InfiniteLine(
            pos=FOCUS_THRESHOLD_DEFAULT, angle=0,
            pen=pg.mkPen(color=(200, 60, 40), width=1.5, style=QtCore.Qt.DashLine),
        )
        self.focus2_plot.addItem(self.focus2_threshold_line)
        self.focus2_bt_curve = self.focus2_plot.plot(
            name="专注指数", pen=pg.mkPen(color=FOCUS2_BT_COLOR, width=2.5),
        )
        self.focus2_cog_curves = {}
        for key, legend, _, rgb in FOCUS2_COG_SPECS:
            self.focus2_cog_curves[key] = self.focus2_plot.plot(
                name=legend, pen=pg.mkPen(color=rgb, width=1.5),
            )

        self.focus2_curve_checks = {}
        curve_check_specs = [("专注指数", "#0078d7", self.focus2_bt_curve)]
        curve_check_specs += [
            (legend, color_hex, self.focus2_cog_curves[key])
            for key, legend, color_hex, _ in FOCUS2_COG_SPECS
        ]
        for name, color_hex, curve in curve_check_specs:
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {color_hex}; font-weight: bold; }}")
            cb.toggled.connect(lambda visible, c=curve: c.setVisible(visible))
            self.focus2_curve_checks[name] = cb
            chart_header.addWidget(cb)
        chart_header.addWidget(
            self._make_info_tip(
                "- 勾选「显示曲线」可显隐各指标曲线\n"
                "- 也可点击图表左上角图例的色块/文字"
            )
        )
        chart_layout.addLayout(chart_header)
        self.focus2_plot.setMinimumHeight(420)
        chart_layout.addWidget(self.focus2_plot, 1)
        layout.addWidget(chart_card, 1)

        return tab

    def _build_wearing_experiment_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        plots_panel = QtWidgets.QFrame()
        plots_panel.setObjectName("focus2ChartCard")
        plots_grid = QtWidgets.QGridLayout(plots_panel)
        plots_grid.setContentsMargins(10, 8, 10, 8)
        plots_grid.setHorizontalSpacing(6)
        plots_grid.setVerticalSpacing(1)
        self.wearing_exp_plots = []
        self.wearing_exp_curves = []
        wearing_plot_h = 40
        for ch in range(16):
            ch_label = QtWidgets.QLabel(f"{ch + 1}")
            ch_label.setAlignment(QtCore.Qt.AlignCenter)
            ch_label.setFixedSize(28, 24)
            label_color = WEARING_COLORS[ch % len(WEARING_COLORS)]
            ch_label.setStyleSheet(
                "border-radius: 12px; color: white; font-weight: bold; "
                f"background: rgb{label_color};"
            )

            plot = pg.PlotWidget()
            plot.setBackground("w")
            plot.setFixedHeight(wearing_plot_h)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.setXRange(-WEARING_WINDOW_SECONDS, 0, padding=0)
            plot.setYRange(-WEARING_Y_RANGE_UV, WEARING_Y_RANGE_UV)
            plot.getPlotItem().hideButtons()
            plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
            plot.getAxis("left").setWidth(0)
            plot.getAxis("left").setStyle(showValues=False)
            plot.getAxis("bottom").setStyle(showValues=(ch == 15))
            if ch != 15:
                plot.getAxis("bottom").setHeight(0)
            color = WEARING_COLORS[ch % len(WEARING_COLORS)]
            curve = plot.plot(pen=pg.mkPen(color=color, width=1.0))
            self.wearing_exp_plots.append(plot)
            self.wearing_exp_curves.append(curve)
            plots_grid.addWidget(ch_label, ch, 0)
            plots_grid.addWidget(plot, ch, 1)
        left_layout.addWidget(plots_panel, 1)
        layout.addWidget(left, 3)

        right = QtWidgets.QFrame()
        right.setObjectName("wearingRightPanel")
        right.setMinimumWidth(480)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        self.wearing_exp_summary_label = QtWidgets.QLabel("通道质量: 未检测")
        sf = self.wearing_exp_summary_label.font()
        sf.setPointSize(15)
        sf.setBold(True)
        self.wearing_exp_summary_label.setFont(sf)
        self.wearing_exp_summary_label.setStyleSheet("color: #666; background: transparent;")

        self.wearing_exp_score_label = QtWidgets.QLabel("OK通道: 0 / 16")
        score_font = self.wearing_exp_score_label.font()
        score_font.setPointSize(14)
        score_font.setBold(True)
        self.wearing_exp_score_label.setFont(score_font)
        self.wearing_exp_score_label.setStyleSheet("background: transparent;")

        self.wearing_exp_btn = QtWidgets.QPushButton("开始佩戴检测")
        self.wearing_exp_btn.setEnabled(False)
        self.wearing_exp_btn.setMinimumWidth(120)
        self.wearing_exp_btn.clicked.connect(self.toggle_wearing_exp)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(10)
        header_row.addWidget(
            self._make_info_tip(WEARING_TAB_INFO), 0, QtCore.Qt.AlignTop,
        )
        status_col = QtWidgets.QVBoxLayout()
        status_col.setSpacing(2)
        status_col.addWidget(self.wearing_exp_summary_label)
        status_col.addWidget(self.wearing_exp_score_label)
        header_row.addLayout(status_col, 1)
        header_row.addWidget(self.wearing_exp_btn, 0, QtCore.Qt.AlignVCenter)
        right_layout.addLayout(header_row)

        self.wearing_exp_result_tabs = QtWidgets.QTabWidget()
        self.wearing_exp_result_tabs.setObjectName("wearingResultTabs")

        map_tab = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(map_tab)
        map_layout.setContentsMargins(0, 4, 0, 0)
        self.wearing_exp_head_map = WearingHeadSchematicWidget()
        map_layout.addWidget(self.wearing_exp_head_map, 1)
        self.wearing_exp_result_tabs.addTab(map_tab, "拓扑示意图")

        list_tab = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_tab)
        list_layout.setContentsMargins(0, 4, 0, 0)
        list_scroll = QtWidgets.QScrollArea()
        list_scroll.setWidgetResizable(True)
        list_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        list_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        list_inner = QtWidgets.QWidget()
        list_grid = QtWidgets.QGridLayout(list_inner)
        list_grid.setContentsMargins(0, 0, 0, 0)
        list_grid.setHorizontalSpacing(8)
        list_grid.setVerticalSpacing(6)
        self.wearing_exp_channel_cards = []
        for ch in range(16):
            card = self._build_wearing_channel_card(ch)
            self.wearing_exp_channel_cards.append(card)
            list_grid.addWidget(card["frame"], ch // 2, ch % 2)
        list_scroll.setWidget(list_inner)
        list_layout.addWidget(list_scroll)
        self.wearing_exp_result_tabs.addTab(list_tab, "数值列表")

        right_layout.addWidget(self.wearing_exp_result_tabs, 1)
        layout.addWidget(right, 2)

        return tab

    def _build_acquire_tab(self):
        """采集范式(ME): 多范式可选, 放松 → Segment 交互 → 段间休息, 全程打标签落盘。"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        stage = QtWidgets.QFrame()
        stage.setObjectName("acquireStageCard")
        left_layout = QtWidgets.QVBoxLayout(stage)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(10)

        self.acquire_cue_image = QtWidgets.QLabel()
        self.acquire_cue_image.setAlignment(QtCore.Qt.AlignCenter)
        self.acquire_cue_image.setMinimumHeight(360)
        self.acquire_cue_image.setStyleSheet("background: transparent;")
        left_layout.addWidget(self.acquire_cue_image, 1)

        self.acquire_cue_label = QtWidgets.QLabel("未开始")
        self.acquire_cue_label.setAlignment(QtCore.Qt.AlignCenter)
        self.acquire_cue_label.setObjectName("acquireCueLabel")
        left_layout.addWidget(self.acquire_cue_label)

        self.acquire_hint_label = QtWidgets.QLabel(
            "请连接板卡，选择范式后点击「开始训练采集」或「开始测试采集」"
        )
        self.acquire_hint_label.setAlignment(QtCore.Qt.AlignCenter)
        self.acquire_hint_label.setWordWrap(True)
        self.acquire_hint_label.setObjectName("acquireHintLabel")
        self._set_acquire_cue_style(MEDICAL_PALETTE["muted"])
        self._set_acquire_hint_style(MEDICAL_PALETTE["muted"])
        left_layout.addWidget(self.acquire_hint_label)

        layout.addWidget(stage, 3)

        right = QtWidgets.QFrame()
        right.setObjectName("acquireSidePanel")
        right.setMinimumWidth(320)
        right.setMaximumWidth(380)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(12)

        paradigm_card = QtWidgets.QFrame()
        paradigm_card.setObjectName("acquireSectionCard")
        paradigm_v = QtWidgets.QVBoxLayout(paradigm_card)
        paradigm_v.setContentsMargins(12, 10, 12, 10)
        paradigm_v.setSpacing(6)
        paradigm_lbl = QtWidgets.QLabel("范式类型")
        paradigm_lbl.setObjectName("acquireFieldLabel")
        paradigm_v.addWidget(paradigm_lbl)
        self.paradigm_combo = QtWidgets.QComboBox()
        self._populate_paradigm_combo()
        self.paradigm_combo.currentIndexChanged.connect(self._on_paradigm_changed)
        paradigm_v.addWidget(self.paradigm_combo)
        right_layout.addWidget(paradigm_card)

        train_card = QtWidgets.QFrame()
        train_card.setObjectName("acquireSectionCard")
        train_v = QtWidgets.QVBoxLayout(train_card)
        train_v.setContentsMargins(12, 12, 12, 12)
        train_v.setSpacing(8)

        train_title_row = QtWidgets.QHBoxLayout()
        train_title_row.setContentsMargins(0, 0, 0, 0)
        train_title_row.setSpacing(8)
        train_lbl = QtWidgets.QLabel("训练采集")
        train_lbl.setObjectName("acquireSectionTitle")
        train_title_row.addWidget(train_lbl)
        self.acquire_train_info_tip = self._make_info_tip("")
        train_title_row.addWidget(self.acquire_train_info_tip)
        train_title_row.addStretch()
        train_v.addLayout(train_title_row)

        sess_lbl = QtWidgets.QLabel("Session ID")
        sess_lbl.setObjectName("acquireFieldLabel")
        train_v.addWidget(sess_lbl)
        sess_row = QtWidgets.QHBoxLayout()
        sess_row.setSpacing(8)
        self.acquire_session_edit = QtWidgets.QLineEdit()
        self.acquire_session_edit.setPlaceholderText("自动新建，如 FINGER2_20260618_143052")
        self.acquire_session_edit.textChanged.connect(self._refresh_acquire_info_tips)
        sess_row.addWidget(self.acquire_session_edit, 1)
        self.acquire_new_session_btn = QtWidgets.QPushButton("新建")
        self.acquire_new_session_btn.setObjectName("btnSecondary")
        self.acquire_new_session_btn.setMaximumWidth(64)
        self.acquire_new_session_btn.clicked.connect(self._new_acquire_session)
        sess_row.addWidget(self.acquire_new_session_btn)
        train_v.addLayout(sess_row)

        self._new_acquire_session()

        self.acquire_btn = QtWidgets.QPushButton("开始训练采集")
        self.acquire_btn.setObjectName("btnPrimary")
        self.acquire_btn.setEnabled(False)
        self.acquire_btn.setMinimumHeight(36)
        self.acquire_btn.clicked.connect(self.toggle_acquire)
        train_v.addWidget(self.acquire_btn)

        self.subject_cue_btn = QtWidgets.QPushButton("受试者全屏提示")
        self.subject_cue_btn.setObjectName("btnSecondary")
        self.subject_cue_btn.setMinimumHeight(34)
        self.subject_cue_btn.setToolTip(
            "另开全屏窗口仅显示提示图与文字，便于受试者观看；操作员继续使用本侧控件。"
        )
        self.subject_cue_btn.clicked.connect(self._open_subject_cue_window)
        train_v.addWidget(self.subject_cue_btn)

        self.acquire_status_label = QtWidgets.QLabel("未连接")
        self.acquire_status_label.setObjectName("acquireMetaValue")
        train_v.addLayout(self._acquire_meta_row("状态", self.acquire_status_label))

        self.acquire_progress_label = QtWidgets.QLabel("---")
        self.acquire_progress_label.setObjectName("acquireMetaMono")
        self.acquire_progress_label.setWordWrap(True)
        train_v.addLayout(self._acquire_meta_row("进度", self.acquire_progress_label))

        self.acquire_result_label = QtWidgets.QLabel("")
        self.acquire_result_label.setObjectName("acquireMetaMono")
        self.acquire_result_label.setWordWrap(True)
        train_v.addWidget(self.acquire_result_label)

        right_layout.addWidget(train_card)
        self._add_test_acquire_controls(right_layout)
        right_layout.addStretch(1)
        layout.addWidget(right, 1)

        self._refresh_paradigm_ui()
        return tab

    def _acquire_meta_row(self, label_text, value_widget):
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        lbl = QtWidgets.QLabel(label_text)
        lbl.setObjectName("acquireMetaLabel")
        lbl.setFixedWidth(36)
        row.addWidget(lbl)
        row.addWidget(value_widget, 1)
        return row

    def _populate_paradigm_combo(self):
        self.paradigm_combo.clear()
        default_idx = 0
        for group_name, ids in PARADIGM_COMBO_GROUPS:
            self.paradigm_combo.addItem(f"── {group_name} ──")
            sep_idx = self.paradigm_combo.count() - 1
            sep_item = self.paradigm_combo.model().item(sep_idx)
            if sep_item is not None:
                sep_item.setEnabled(False)
            for pid in ids:
                spec = get_paradigm(pid)
                self.paradigm_combo.addItem(spec.name_cn, pid)
                if pid == DEFAULT_PARADIGM_ID:
                    default_idx = self.paradigm_combo.count() - 1
        self.paradigm_combo.setCurrentIndex(default_idx)

    def _selected_paradigm(self):
        pid = self.paradigm_combo.currentData()
        if not pid:
            return default_paradigm()
        return get_paradigm(str(pid))

    def _on_paradigm_changed(self):
        if not self.paradigm_combo.currentData():
            return
        if self.acquire_running or self.test_running:
            return
        self.acquire_cue_pixmaps.clear()
        self._new_acquire_session()
        self._refresh_paradigm_ui()

    def _refresh_paradigm_ui(self):
        if not hasattr(self, "paradigm_combo"):
            return
        spec = self._selected_paradigm()
        if hasattr(self, "test_segments_spin"):
            self.test_segments_spin.setValue(spec.test_segments)
            self.test_trials_spin.setValue(spec.test_trials)
        self._refresh_acquire_info_tips()

    def _refresh_acquire_info_tips(self):
        """根据当前范式、Session ID、测试参数刷新训练/测试 ⓘ 说明。"""
        if not hasattr(self, "paradigm_combo"):
            return
        spec = self._selected_paradigm()
        project_dir = os.path.dirname(os.path.abspath(__file__))
        sid = self.acquire_session_edit.text().strip() if hasattr(self, "acquire_session_edit") else ""
        if not sid:
            sid = f"{spec.session_prefix}_YYYYMMDD_HHMMSS"
        train_base = os.path.join(project_dir, ACQUIRE_RECORD_DIR)
        train_path = mi_session_csv_path(train_base, sid, purpose="train", paradigm_subdir=spec.record_subdir)
        if hasattr(self, "acquire_train_info_tip"):
            self.acquire_train_info_tip.set_info_text(spec.train_info_tip(save_path=train_path))
        if hasattr(self, "test_acquire_info_tip"):
            timing = self._test_mi_timing() if hasattr(self, "test_trials_spin") else spec.train_timing
            test_dir = os.path.join(project_dir, TEST_RECORD_DIR, spec.record_subdir)
            self.test_acquire_info_tip.set_info_text(spec.test_info_tip(timing, save_dir=test_dir))

    def _paradigm_schedule_kwargs(self, spec):
        return {
            "conditions": list(spec.conditions),
            "condition_to_label": dict(spec.condition_to_label),
            "condition_to_marker": dict(spec.condition_to_marker),
            "label_names_cn": dict(spec.label_names_cn),
        }

    def _set_acquire_cue_style(self, color):
        self.acquire_cue_label.setStyleSheet(
            f"font-size: {ACQUIRE_CUE_FONT_PX}px; font-weight: bold; color: {color}; background: transparent;"
        )

    def _set_acquire_hint_style(self, color):
        self.acquire_hint_label.setStyleSheet(
            f"font-size: {ACQUIRE_HINT_FONT_PX}px; font-weight: bold; color: {color}; background: transparent;"
        )

    def _new_acquire_session(self):
        """新建训练 session。"""
        spec = self._selected_paradigm()
        sid = new_session_id(spec.session_prefix)
        self.acquire_session_edit.setText(sid)

    def _default_acquire_mi_timing(self):
        return self._selected_paradigm().train_timing

    def _acquire_cue_pixmap(self, condition):
        """按条件名加载并缓存提示图（保留原分辨率，显示时再缩放）。"""
        spec = self._selected_paradigm()
        cache_key = (spec.id, condition)
        if cache_key in self.acquire_cue_pixmaps:
            return self.acquire_cue_pixmaps[cache_key]
        path = resolve_cue_image(_PROJECT_ROOT, spec.cue_image_subdir, condition)
        pix = None
        if path is not None:
            pm = QtGui.QPixmap(path)
            if not pm.isNull():
                pix = pixmap_blend_bg(pm, ACQUIRE_BG_COLOR)
            else:
                logging.getLogger("bci_desk").warning("提示图无法解码: %s", path)
        else:
            logging.getLogger("bci_desk").warning(
                "未找到提示图: paradigm=%s condition=%s", spec.id, condition
            )
        self.acquire_cue_pixmaps[cache_key] = pix
        return pix

    def _acquire_fixation_pixmap(self):
        """试次间休息注视点（保留原分辨率）。"""
        cache_key = ("__fixation_rest__",)
        if cache_key in self.acquire_cue_pixmaps:
            return self.acquire_cue_pixmaps[cache_key]
        path = resolve_cue_image(_PROJECT_ROOT, "common", "fixation_rest")
        pix = None
        if path is not None:
            pm = QtGui.QPixmap(path)
            if not pm.isNull():
                pix = pixmap_blend_bg(pm, ACQUIRE_BG_COLOR)
        self.acquire_cue_pixmaps[cache_key] = pix
        return pix

    def toggle_acquire(self):
        if not self.acquire_running:
            self.start_acquire()
        else:
            if not self._confirm_stop_activity(
                "停止训练采集",
                "确定停止当前训练采集吗？已录数据仍会尝试保存。",
            ):
                return
            self.stop_acquire(finished=False)

    def toggle_test(self):
        if not self.test_running:
            self.start_test()
        else:
            if not self._confirm_stop_activity(
                "停止测试采集",
                "确定停止当前测试采集吗？已录数据仍会尝试保存。",
            ):
                return
            self.stop_test(finished=False)

    def _show_acquire_intro(self):
        """开始采集前的流程说明页。"""
        spec = self._selected_paradigm()
        timing = self._default_acquire_mi_timing()

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("采集流程说明")
        dlg.setMinimumSize(720, 560)
        dlg.setStyleSheet(f"QDialog {{ background-color: {ACQUIRE_BG_COLOR}; }}")
        v = QtWidgets.QVBoxLayout(dlg)
        v.setSpacing(14)
        v.setContentsMargins(24, 20, 24, 20)

        title = QtWidgets.QLabel(f"{spec.name_cn} 采集流程")
        tf = title.font()
        tf.setPointSize(24)
        tf.setBold(True)
        title.setFont(tf)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(f"color: {MEDICAL_PALETTE['primary']}; background: transparent;")
        v.addWidget(title)

        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setFrameShape(QtWidgets.QFrame.NoFrame)
        browser.setStyleSheet(
            f"QTextBrowser {{ background: {MEDICAL_PALETTE['surface']}; border: 1px solid {MEDICAL_PALETTE['border']}; "
            f"border-radius: 8px; padding: 16px; }}"
        )
        browser.setHtml(
            paradigm_intro_html(spec, timing, palette_text=MEDICAL_PALETTE["text"])
        )
        v.addWidget(browser, 1)

        btns = QtWidgets.QDialogButtonBox()
        start_btn = btns.addButton("开始", QtWidgets.QDialogButtonBox.AcceptRole)
        btns.addButton("取消", QtWidgets.QDialogButtonBox.RejectRole)
        start_btn.setDefault(True)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)

        return dlg.exec_() == QtWidgets.QDialog.Accepted

    def _build_mi_schedule_for_acquire(self):
        spec = self._selected_paradigm()
        return build_mi_session_schedule(
            timing=self._default_acquire_mi_timing(),
            manual_waits=True,
            **self._paradigm_schedule_kwargs(spec),
        )

    def _on_paradigm_space_continue(self):
        """范式 manual_wait 阶段：空格确认进入下一阶段。"""
        now_ms = time.time() * 1000.0
        if self.acquire_running and self.acquire_recorder is not None:
            phase = self.acquire_recorder.current_phase
            if phase is not None and phase.is_manual_wait:
                done, nxt, remaining = self.acquire_recorder.confirm_manual_wait(now_ms)
                if done:
                    self.stop_acquire(finished=True)
                else:
                    self._update_acquire_display(nxt, remaining)
                return
        if self.test_running and self.test_recorder is not None:
            phase = self.test_recorder.current_phase
            if phase is not None and phase.is_manual_wait:
                done, nxt, remaining = self.test_recorder.confirm_manual_wait(now_ms)
                if done:
                    self.stop_test(finished=True)
                else:
                    self._update_test_display(nxt, remaining)

    def _paradigm_display_key(self, phase, recorder):
        return (recorder.phase_index, phase.kind, phase.condition, phase.marker)

    def _set_cue_image_pixmap(self, cue_image, pix, *, panel_max=460):
        """切换提示图：缓存为原图，按面板尺寸缩放显示。"""
        cue_image.clear()
        self._current_cue_full_pixmap = pix
        if pix is not None and not pix.isNull():
            tw = cue_image.width() if cue_image.width() > 40 else panel_max
            th = cue_image.height() if cue_image.height() > 40 else panel_max
            # 主界面预览默认不超过 panel_max
            tw = min(tw, panel_max) if tw <= panel_max * 1.2 else tw
            th = min(th, panel_max) if th <= panel_max * 1.2 else th
            if tw <= panel_max and th <= panel_max:
                tw = th = panel_max
            scaled = pix.scaled(
                tw,
                th,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            cue_image.setPixmap(scaled)
        cue_image.update()

    def _precache_paradigm_cues(self):
        """采集开始前预加载当前范式全部提示图，避免试次切换时在主线程卡顿。"""
        spec = self._selected_paradigm()
        for cond in spec.conditions:
            self._acquire_cue_pixmap(cond)
        if spec.family == "finger":
            self._acquire_fixation_pixmap()

    def _update_mi_paradigm_display(self, phase, remaining, *, cue_image, cue_label, hint_label, progress_label, recorder, set_cue_style, set_hint_style, label_hint_template):
        """ME 范式大屏显示（采集/测试共用）。"""
        phase_key = self._paradigm_display_key(phase, recorder)
        visual_changed = phase_key != self._last_paradigm_display_key
        if visual_changed:
            self._last_paradigm_display_key = phase_key

        kind = phase.kind
        if visual_changed:
            if kind == KIND_LABEL:
                pix = self._acquire_cue_pixmap(phase.condition)
                self._set_cue_image_pixmap(cue_image, pix)
            elif kind == KIND_MANUAL_WAIT:
                self._current_cue_full_pixmap = None
                cue_image.clear()
            elif kind == KIND_REST:
                pix = self._acquire_fixation_pixmap()
                if pix is not None:
                    self._set_cue_image_pixmap(cue_image, pix)
                else:
                    self._current_cue_full_pixmap = None
                    cue_image.clear()
            else:
                self._current_cue_full_pixmap = None
                cue_image.clear()

        if kind == KIND_LABEL:
            cue_label.setText(str(phase.marker))
            set_cue_style("#1f6feb")
            set_hint_style(MEDICAL_PALETTE["text"])
            hint_label.setText(label_hint_template.format(name=phase.condition_cn))
        elif kind == KIND_MANUAL_WAIT:
            if phase.segment_num == 0:
                cue_label.setText("准备")
                hint_label.setText("按空格键开始 Segment 1")
            else:
                cue_label.setText("休息")
                hint_label.setText(f"段间休息 · 按空格键继续 Segment {phase.segment_num + 1}")
            set_cue_style(ACQUIRE_PHASE_TEXT[KIND_MANUAL_WAIT][1])
            set_hint_style(MEDICAL_PALETTE["text"])
        elif kind == KIND_REST:
            if self._acquire_fixation_pixmap() is None:
                text, color = ACQUIRE_PHASE_TEXT[KIND_REST]
                cue_label.setText("+")
                set_cue_style(color)
                set_hint_style(MEDICAL_PALETTE["text"])
                hint_label.setText(text)
            else:
                cue_label.setText("")
                hint_label.setText("")
        else:
            text, color = ACQUIRE_PHASE_TEXT.get(kind, ("", "#666"))
            if kind == KIND_RELAX:
                cue_label.setText("放松")
            elif kind == KIND_BREAK:
                cue_label.setText("休息")
            else:
                cue_label.setText("+")
            set_cue_style(color)
            set_hint_style(MEDICAL_PALETTE["text"])
            hint_label.setText(text)

        if kind == KIND_RELAX:
            progress_label.setText(
                f"初始放松 · marker {phase.marker} · 剩余 {remaining:.1f}s · 已录 {recorder.n_samples} 点"
            )
        elif kind == KIND_BREAK:
            progress_label.setText(
                f"段间休息 · marker {phase.marker} · Segment {phase.segment_num}/{phase.total_segments} · "
                f"剩余 {remaining:.1f}s · 已录 {recorder.n_samples} 点"
            )
        elif kind in (KIND_LABEL, KIND_REST):
            progress_label.setText(
                f"Segment {phase.segment_num}/{phase.total_segments} · "
                f"试次 {phase.trial_num}/{phase.total_trials} · {kind} · marker {phase.marker} · "
                f"剩余 {remaining:.1f}s · 已录 {recorder.n_samples} 点"
            )
        else:
            progress_label.setText(
                f"{kind} · 剩余 {remaining:.1f}s · 已录 {recorder.n_samples} 点"
            )

    def start_acquire(self):
        if not self.is_connected or self.board is None:
            self.acquire_status_label.setText("请先连接板卡")
            return
        # 开始前展示流程说明页, 受试者确认后再开始
        if not self._show_acquire_intro():
            self.acquire_status_label.setText("已取消")
            return
        # 与其他采集/检测活动互斥
        if self.is_streaming:
            self.stop_stream()
        if self.imp_running:
            self.stop_impedance()
        if self.focus2_running:
            self.stop_focus2()
        if self.wearing_exp_running:
            self.stop_wearing_exp()
        if self.test_running:
            self.stop_test(finished=False)

        self.acquire_schedule = self._build_mi_schedule_for_acquire()
        self.acquire_recorder = ParadigmRecorder(
            self.acquire_schedule, num_channels=len(self.eeg_channels), sample_rate=self.sampling_rate
        )

        try:
            self.board.start_stream()
            self.board.get_board_data()   # 流开始后清空板内既有缓冲, 从干净状态录制
        except (BrainFlowError, SerialBoardError) as e:
            self.acquire_status_label.setText(f"启动采集失败 - {e}")
            return

        self.acquire_running = True
        self.acquire_t0 = time.time() * 1000.0
        self._last_paradigm_display_key = None
        self._precache_paradigm_cues()
        self.acquire_recorder.start(self.acquire_t0)
        self._paradigm_space_shortcut.setEnabled(True)
        init_phase = self.acquire_recorder.current_phase
        if init_phase is not None:
            self._update_acquire_display(init_phase, 0.0)

        self.acquire_btn.setText("停止")
        self.connect_btn.setEnabled(False)
        self.acquire_session_edit.setEnabled(False)
        self.acquire_new_session_btn.setEnabled(False)
        self.paradigm_combo.setEnabled(False)
        if hasattr(self, "test_acquire_btn"):
            self.test_acquire_btn.setEnabled(False)
        # 采集期间禁用其他活动按钮, 避免抢占板卡
        self.start_btn.setEnabled(False)
        self.imp_btn.setEnabled(False)
        self.focus2_btn.setEnabled(False)
        self.focus2_calib_btn.setEnabled(False)
        self.wearing_exp_btn.setEnabled(False)
        self.acquire_result_label.setText("")
        total_s = total_duration_s(self.acquire_schedule)
        timing = self._default_acquire_mi_timing()
        spec = self._selected_paradigm()
        dbg = "[调试] " if self.is_debug_virtual else ""
        self.acquire_status_label.setText(
            f"{dbg}{spec.name_cn} · {timing.n_segments} Segment × {timing.n_trials_per_segment} 试次 · "
            f"约 {total_s / 60.0:.0f}min"
        )
        self.acquire_timer.start(ACQUIRE_TICK_MS)

    def update_acquire(self):
        if not self.acquire_running or self.board is None:
            return

        # 1) 取板卡新数据, 喂给录制器
        data = self.board.get_board_data()
        if data.shape[1] > 0:
            ts_ms = data[self.timestamp_channel] * 1000.0     # 秒 -> 毫秒
            eeg = self._eeg_from_board_filtered(data)
            self.acquire_recorder.add_samples(ts_ms, eeg)

        # 2) 推进范式时钟
        now_ms = time.time() * 1000.0
        done, phase, remaining = self.acquire_recorder.update(now_ms)

        if done:
            self.stop_acquire(finished=True)
            return

        self._update_acquire_display(phase, remaining)

    def _update_acquire_display(self, phase, remaining):
        spec = self._selected_paradigm()
        self._update_mi_paradigm_display(
            phase,
            remaining,
            cue_image=self.acquire_cue_image,
            cue_label=self.acquire_cue_label,
            hint_label=self.acquire_hint_label,
            progress_label=self.acquire_progress_label,
            recorder=self.acquire_recorder,
            set_cue_style=self._set_acquire_cue_style,
            set_hint_style=self._set_acquire_hint_style,
            label_hint_template=spec.cue_hint,
        )
        self._sync_subject_cue_window()

    def stop_acquire(self, finished=False):
        if not self.acquire_running:
            return
        self.acquire_timer.stop()
        try:
            if self.board is not None:
                self.board.stop_stream()
        except (BrainFlowError, SerialBoardError):
            pass
        self.acquire_running = False
        self._last_paradigm_display_key = None
        self._paradigm_space_shortcut.setEnabled(self.test_running)

        self.acquire_btn.setText("开始训练采集")
        self.connect_btn.setEnabled(True)
        self.acquire_session_edit.setEnabled(True)
        self.acquire_new_session_btn.setEnabled(True)
        self.paradigm_combo.setEnabled(True)
        if hasattr(self, "test_acquire_btn") and self.is_connected:
            self.test_acquire_btn.setEnabled(True)
        # 恢复其他活动按钮（虚拟调试端口仅开放范式预览）
        if self.is_connected and not self.is_debug_virtual:
            self.start_btn.setEnabled(True)
            self.imp_btn.setEnabled(True)
            self.focus2_btn.setEnabled(True)
            self.focus2_calib_btn.setEnabled(True)
            self.wearing_exp_btn.setEnabled(True)
        self.acquire_cue_image.clear()

        if finished:
            self.acquire_cue_label.setText("完成")
            self._set_acquire_cue_style("#1a7f37")
            if self.is_debug_virtual:
                self.acquire_hint_label.setText("调试模式: 未保存任何数据")
                self._set_acquire_hint_style(MEDICAL_PALETTE["muted"])
                self.acquire_status_label.setText("[调试] 范式预览完成 · 未落盘")
                self.acquire_result_label.setText(
                    "虚拟端口仅预览训练提示动画，不写入 CSV / manifest / session.json。"
                )
            else:
                self.acquire_hint_label.setText("采集结束, 正在保存...")
                self._set_acquire_hint_style(MEDICAL_PALETTE["muted"])
                self._save_acquire()
        else:
            self.acquire_cue_label.setText("已停止")
            self._set_acquire_cue_style("#888")
            self.acquire_hint_label.setText("采集被手动停止")
            self._set_acquire_hint_style(MEDICAL_PALETTE["muted"])
            if (
                not self.is_debug_virtual
                and self.acquire_recorder is not None
                and self.acquire_recorder.events
            ):
                self._save_acquire()
            elif self.is_debug_virtual:
                self.acquire_status_label.setText("[调试] 已停止 · 未落盘")
            else:
                self.acquire_status_label.setText("已停止")

    def _save_acquire(self):
        if self.is_debug_virtual:
            return
        spec = self._selected_paradigm()
        session_id = self.acquire_session_edit.text().strip()
        if not session_id:
            session_id = new_session_id(spec.session_prefix)
            self.acquire_session_edit.setText(session_id)

        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ACQUIRE_RECORD_DIR)
        out_path = mi_session_csv_path(
            base_dir, session_id, purpose="train", paradigm_subdir=spec.record_subdir,
        )
        timing = self._default_acquire_mi_timing()

        extra_meta = {
            "session_id": session_id,
            "purpose": "train",
            "paradigm_id": spec.id,
            "paradigm": spec.id,
            "family": spec.family,
            "classes": list(spec.labels),
            "conditions": list(spec.conditions),
            "session_timing": {
                "relax_s": timing.relax_s,
                "break_s": timing.break_s,
                "label_s": timing.label_s,
                "rest_s": timing.rest_s,
                "n_trials_per_segment": timing.n_trials_per_segment,
                "n_segments": timing.n_segments,
            },
            **self._recording_meta_filter_note(),
        }

        try:
            stats = self.acquire_recorder.finish(out_path, extra_meta=extra_meta)
        except Exception as e:  # noqa: BLE001
            self.acquire_status_label.setText("保存失败")
            self.acquire_result_label.setText(f"错误: {e}")
            return

        folder = os.path.dirname(stats["output_path"])

        counts = stats.get("label_counts", {})
        counts_str = ", ".join(f"{k}:{v}" for k, v in counts.items())
        self.acquire_status_label.setText(f"已保存 {spec.name_cn} ({stats.get('n_rows')} 行)")

        verify_msg = ""
        try:
            X, y = load_epochs_from_csv(stats["output_path"], input_unit=stats.get("unit", "microvolt"))
            verify_msg = f"\n校验通过: epoch X={tuple(X.shape)}, {len(set(y.tolist()))} 类"
        except Exception as e:  # noqa: BLE001
            verify_msg = f"\n校验警告: {e}"

        train_hint = (
            f"\n\n上传训练: 将 session 目录拷到服务器\n"
            f"  python train.py --data {folder}"
        )
        self.acquire_result_label.setText(
            f"文件: {stats['output_path']}\n"
            f"标签分布: {counts_str}\n"
            f"单位: {stats.get('unit')}{verify_msg}{train_hint}"
        )

    def _add_test_acquire_controls(self, right_layout):
        """模型测试采集控件（合并至采集范式页右侧，与训练采集共用左侧大屏提示）。"""
        test_card = QtWidgets.QFrame()
        test_card.setObjectName("acquireSectionCard")
        test_v = QtWidgets.QVBoxLayout(test_card)
        test_v.setContentsMargins(12, 12, 12, 12)
        test_v.setSpacing(8)

        test_title_row = QtWidgets.QHBoxLayout()
        test_title_row.setContentsMargins(0, 0, 0, 0)
        test_title_row.setSpacing(8)
        test_lbl = QtWidgets.QLabel("测试集采集")
        test_lbl.setObjectName("acquireSectionTitle")
        test_title_row.addWidget(test_lbl)
        self.test_acquire_info_tip = self._make_info_tip("")
        test_title_row.addWidget(self.test_acquire_info_tip)
        test_title_row.addStretch()
        test_v.addLayout(test_title_row)

        param_form = QtWidgets.QGridLayout()
        param_form.setHorizontalSpacing(10)
        param_form.setVerticalSpacing(8)
        seg_lbl = QtWidgets.QLabel("Segment 数")
        seg_lbl.setObjectName("acquireFieldLabel")
        param_form.addWidget(seg_lbl, 0, 0)
        self.test_segments_spin = QtWidgets.QSpinBox()
        self.test_segments_spin.setRange(1, 3)
        self.test_segments_spin.setValue(TEST_DEFAULT_SEGMENTS)
        self.test_segments_spin.valueChanged.connect(self._refresh_acquire_info_tips)
        param_form.addWidget(self.test_segments_spin, 0, 1)

        trials_lbl = QtWidgets.QLabel("每 Segment 试次")
        trials_lbl.setObjectName("acquireFieldLabel")
        param_form.addWidget(trials_lbl, 1, 0)
        self.test_trials_spin = QtWidgets.QSpinBox()
        self.test_trials_spin.setRange(5, 300)
        self.test_trials_spin.setValue(TEST_DEFAULT_TRIALS_PER_SEGMENT)
        self.test_trials_spin.valueChanged.connect(self._refresh_acquire_info_tips)
        param_form.addWidget(self.test_trials_spin, 1, 1)
        test_v.addLayout(param_form)

        self.test_acquire_btn = QtWidgets.QPushButton("开始测试采集")
        self.test_acquire_btn.setObjectName("btnSecondary")
        self.test_acquire_btn.setEnabled(False)
        self.test_acquire_btn.setMinimumHeight(36)
        self.test_acquire_btn.clicked.connect(self.toggle_test)
        test_v.addWidget(self.test_acquire_btn)

        self.test_status_label = QtWidgets.QLabel("未连接")
        self.test_status_label.setObjectName("acquireMetaValue")
        test_v.addLayout(self._acquire_meta_row("状态", self.test_status_label))

        self.test_progress_label = QtWidgets.QLabel("---")
        self.test_progress_label.setObjectName("acquireMetaMono")
        self.test_progress_label.setWordWrap(True)
        test_v.addLayout(self._acquire_meta_row("进度", self.test_progress_label))

        self.test_result_label = QtWidgets.QLabel("")
        self.test_result_label.setObjectName("acquireMetaMono")
        self.test_result_label.setWordWrap(True)
        self.test_result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        test_v.addWidget(self.test_result_label)

        right_layout.addWidget(test_card)

    def _test_mi_timing(self):
        base = self._selected_paradigm().train_timing
        return replace(
            base,
            n_trials_per_segment=int(self.test_trials_spin.value()),
            n_segments=int(self.test_segments_spin.value()),
        )

    def _build_mi_schedule_for_test(self):
        spec = self._selected_paradigm()
        timing = self._test_mi_timing()
        return build_mi_session_schedule(
            timing=timing,
            manual_waits=True,
            **self._paradigm_schedule_kwargs(spec),
        )

    def _stop_other_activities_for_test(self):
        if self.is_streaming:
            self.stop_stream()
        if self.imp_running:
            self.stop_impedance()
        if self.focus2_running:
            self.stop_focus2()
        if self.wearing_exp_running:
            self.stop_wearing_exp()
        if self.acquire_running:
            self.stop_acquire(finished=False)

    def start_test(self):
        if not self.is_connected or self.board is None:
            self.test_status_label.setText("请先连接板卡")
            return
        self._stop_other_activities_for_test()

        self.test_schedule = self._build_mi_schedule_for_test()
        self.test_recorder = ParadigmRecorder(
            self.test_schedule, num_channels=len(self.eeg_channels), sample_rate=self.sampling_rate
        )

        try:
            self.board.start_stream()
            self.board.get_board_data()
        except (BrainFlowError, SerialBoardError) as e:
            self.test_status_label.setText(f"启动采集失败 - {e}")
            return

        self.test_running = True
        self._last_paradigm_display_key = None
        self._precache_paradigm_cues()
        t0 = time.time() * 1000.0
        self.test_recorder.start(t0)
        self._paradigm_space_shortcut.setEnabled(True)
        init_phase = self.test_recorder.current_phase
        if init_phase is not None:
            self._update_test_display(init_phase, 0.0)

        self.test_acquire_btn.setText("停止")
        self.connect_btn.setEnabled(False)
        self.test_segments_spin.setEnabled(False)
        self.test_trials_spin.setEnabled(False)
        self.acquire_btn.setEnabled(False)
        self.acquire_session_edit.setEnabled(False)
        self.acquire_new_session_btn.setEnabled(False)
        self.paradigm_combo.setEnabled(False)
        total_s = total_duration_s(self.test_schedule)
        timing = self._test_mi_timing()
        self.test_result_label.setText("")
        self.test_status_label.setText(
            f"测试采集中 · {timing.n_segments} Segment × {timing.n_trials_per_segment} 试次 · "
            f"约 {total_s / 60.0:.1f}min"
        )
        self.test_timer.start(TEST_TICK_MS)

    def update_test(self):
        if not self.test_running or self.board is None:
            return
        data = self.board.get_board_data()
        if data.shape[1] > 0:
            ts_ms = data[self.timestamp_channel] * 1000.0
            eeg = self._eeg_from_board_filtered(data)
            self.test_recorder.add_samples(ts_ms, eeg)

        now_ms = time.time() * 1000.0
        done, phase, remaining = self.test_recorder.update(now_ms)
        if done:
            self.stop_test(finished=True)
            return
        self._update_test_display(phase, remaining)

    def _update_test_display(self, phase, remaining):
        spec = self._selected_paradigm()
        self._update_mi_paradigm_display(
            phase,
            remaining,
            cue_image=self.acquire_cue_image,
            cue_label=self.acquire_cue_label,
            hint_label=self.acquire_hint_label,
            progress_label=self.test_progress_label,
            recorder=self.test_recorder,
            set_cue_style=self._set_acquire_cue_style,
            set_hint_style=self._set_acquire_hint_style,
            label_hint_template=spec.cue_hint,
        )
        self._sync_subject_cue_window()

    def stop_test(self, finished=False):
        if not self.test_running:
            return
        self.test_timer.stop()
        try:
            if self.board is not None:
                self.board.stop_stream()
        except (BrainFlowError, SerialBoardError):
            pass
        self.test_running = False
        self._last_paradigm_display_key = None
        self._paradigm_space_shortcut.setEnabled(self.acquire_running)

        self.test_acquire_btn.setText("开始测试采集")
        self.connect_btn.setEnabled(True)
        self.test_segments_spin.setEnabled(True)
        self.test_trials_spin.setEnabled(True)
        self.acquire_session_edit.setEnabled(True)
        self.acquire_new_session_btn.setEnabled(True)
        self.paradigm_combo.setEnabled(True)
        if self.is_connected:
            self.acquire_btn.setEnabled(True)
            self.test_acquire_btn.setEnabled(True)
        self.acquire_cue_image.clear()

        if finished:
            self.acquire_cue_label.setText("完成")
            self._set_acquire_cue_style("#1a7f37")
            if self.is_debug_virtual:
                self.acquire_hint_label.setText("调试模式: 未保存任何数据")
                self._set_acquire_hint_style(MEDICAL_PALETTE["muted"])
                self.test_status_label.setText("[调试] 测试预览完成 · 未落盘")
                self.test_result_label.setText("虚拟端口仅预览动画，不写入测试 CSV。")
            else:
                self.acquire_hint_label.setText("测试采集结束, 正在保存...")
                self._set_acquire_hint_style(MEDICAL_PALETTE["muted"])
                self._save_test()
        else:
            self.acquire_cue_label.setText("已停止")
            self._set_acquire_cue_style("#888")
            if (
                not self.is_debug_virtual
                and self.test_recorder is not None
                and self.test_recorder.events
            ):
                self._save_test()
            elif self.is_debug_virtual:
                self.test_status_label.setText("[调试] 已停止 · 未落盘")
            else:
                self.test_status_label.setText("已停止")

    def _save_test(self):
        if self.is_debug_virtual:
            return
        spec = self._selected_paradigm()
        session_id = new_session_id(f"{spec.session_prefix}_test")
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), TEST_RECORD_DIR)
        out_path = mi_session_csv_path(
            base_dir, session_id, purpose="test", paradigm_subdir=spec.record_subdir,
        )
        timing = self._test_mi_timing()
        extra_meta = {
            "session_id": session_id,
            "purpose": "test",
            "paradigm_id": spec.id,
            "paradigm": spec.id,
            "family": spec.family,
            "classes": list(spec.labels),
            "conditions": list(spec.conditions),
            "session_timing": {
                "relax_s": timing.relax_s,
                "break_s": timing.break_s,
                "label_s": timing.label_s,
                "rest_s": timing.rest_s,
                "n_trials_per_segment": timing.n_trials_per_segment,
                "n_segments": timing.n_segments,
            },
            **self._recording_meta_filter_note(),
        }
        try:
            stats = self.test_recorder.finish(out_path, extra_meta=extra_meta)
        except Exception as e:  # noqa: BLE001
            self.test_status_label.setText("保存失败")
            self.test_result_label.setText(f"错误: {e}")
            return

        self.test_last_csv = stats["output_path"]
        counts = stats.get("label_counts", {})
        counts_str = ", ".join(f"{k}:{v}" for k, v in counts.items())
        self.test_status_label.setText(f"已保存 ({stats.get('n_rows')} 行)")

        eval_hint = (
            "\n\n评估(服务器):\n"
            f"  python evaluate.py --model models/eegnet_mi.pth --data {os.path.basename(self.test_last_csv)}"
        )
        self.test_result_label.setText(
            f"测试文件: {self.test_last_csv}\n"
            f"{spec.name_cn}\n"
            f"标签分布: {counts_str}\n"
            f"单位: {stats.get('unit')}{eval_hint}"
        )

    # ==================== ERD/ERS 检测页 ====================
    def _erders_default_timing(self):
        return self.erders_config.timing

    def _build_erders_tab(self):
        """ERD/ERS 检测: 左右手运动执行范式 + 在线 Mu/Beta 频带功率 ERD/ERS 分析。

        单试次: 基线2s → 准备1s → 运动执行(握拳)4s(左/右随机) → 放松2s → 间隔1s。
        分析: 基线功率为参考, 任务期(cue 后 0.5~4.0s)算 ERD, 整段 Relax 算 ERS(Beta 反弹)。
        落盘: 完整原始 EEG + 事件 marker(便于离线复算), 以及 trial_results.csv 与平均 ERD 曲线。
        """
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        stage = QtWidgets.QFrame()
        stage.setObjectName("erdersStageCard")
        left_layout = QtWidgets.QVBoxLayout(stage)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(6)

        self.erders_cue_label = QtWidgets.QLabel("未开始")
        self.erders_cue_label.setAlignment(QtCore.Qt.AlignCenter)
        self.erders_cue_label.setMinimumHeight(160)
        left_layout.addWidget(self.erders_cue_label, 1)

        self.erders_hint_label = QtWidgets.QLabel(
            "请连接板卡，设置左右手试次数后点击「开始 ERD/ERS 实验」"
        )
        self.erders_hint_label.setAlignment(QtCore.Qt.AlignCenter)
        self.erders_hint_label.setWordWrap(True)
        left_layout.addWidget(self.erders_hint_label)

        self.erders_progress_label = QtWidgets.QLabel("Trial 0 / 0")
        self.erders_progress_label.setAlignment(QtCore.Qt.AlignCenter)
        self.erders_progress_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {MEDICAL_PALETTE['primary']}; background: transparent;"
        )
        left_layout.addWidget(self.erders_progress_label)

        self._erders_set_cue_style(MEDICAL_PALETTE["muted"])
        self._erders_set_hint_style(MEDICAL_PALETTE["muted"])
        # 左侧略压缩；右侧整体不超过一半
        layout.addWidget(stage, 55)

        right = QtWidgets.QFrame()
        right.setObjectName("erdersSidePanel")
        right.setMinimumWidth(380)
        right.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding,
        )
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(6)

        cfg_card = QtWidgets.QFrame()
        cfg_card.setObjectName("erdersSectionCard")
        cfg_v = QtWidgets.QVBoxLayout(cfg_card)
        cfg_v.setContentsMargins(8, 6, 8, 6)
        cfg_v.setSpacing(4)

        title_row = QtWidgets.QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title_lbl = QtWidgets.QLabel("ERD/ERS 检测")
        title_lbl.setObjectName("erdersPageTitle")
        title_row.addWidget(title_lbl)
        self.erders_info_tip = self._make_info_tip("")
        title_row.addWidget(self.erders_info_tip)
        title_row.addStretch()
        cfg_v.addLayout(title_row)

        param_form = QtWidgets.QGridLayout()
        param_form.setHorizontalSpacing(8)
        param_form.setVerticalSpacing(4)

        def _field(row, col, text, widget):
            lbl = QtWidgets.QLabel(text)
            lbl.setObjectName("erdersFieldLabel")
            param_form.addWidget(lbl, row, col * 2)
            param_form.addWidget(widget, row, col * 2 + 1)

        self.erders_left_spin = QtWidgets.QSpinBox()
        self.erders_left_spin.setRange(1, 200)
        self.erders_left_spin.setValue(ERDERS_DEFAULT_TRIALS_PER_DIR)
        self.erders_left_spin.valueChanged.connect(self._refresh_erders_info)
        _field(0, 0, "左手试次", self.erders_left_spin)

        self.erders_right_spin = QtWidgets.QSpinBox()
        self.erders_right_spin.setRange(1, 200)
        self.erders_right_spin.setValue(ERDERS_DEFAULT_TRIALS_PER_DIR)
        self.erders_right_spin.valueChanged.connect(self._refresh_erders_info)
        _field(0, 1, "右手试次", self.erders_right_spin)

        ch_items = [f"CH{c + 1}" for c in range(len(self.eeg_channels))]
        n_ch = len(ch_items)
        self.erders_c3_combo = QtWidgets.QComboBox()
        self.erders_c3_combo.addItems(ch_items)
        self.erders_c3_combo.setCurrentIndex(min(2, n_ch - 1))
        _field(1, 0, "C3(左脑)", self.erders_c3_combo)

        self.erders_c4_combo = QtWidgets.QComboBox()
        self.erders_c4_combo.addItems(ch_items)
        self.erders_c4_combo.setCurrentIndex(min(3, n_ch - 1))
        _field(1, 1, "C4(右脑)", self.erders_c4_combo)
        cfg_v.addLayout(param_form)

        self.erders_info_label = QtWidgets.QLabel("")
        self.erders_info_label.hide()

        self.erders_btn = QtWidgets.QPushButton("开始 ERD/ERS 实验")
        self.erders_btn.setObjectName("btnPrimary")
        self.erders_btn.setEnabled(False)
        self.erders_btn.setMinimumHeight(32)
        self.erders_btn.clicked.connect(self.toggle_erders)
        cfg_v.addWidget(self.erders_btn)

        self.erders_status_label = QtWidgets.QLabel("未连接")
        self.erders_status_label.setObjectName("erdersMetaValue")
        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        status_lbl = QtWidgets.QLabel("状态")
        status_lbl.setObjectName("erdersMetaLabel")
        status_lbl.setFixedWidth(36)
        status_row.addWidget(status_lbl)
        status_row.addWidget(self.erders_status_label, 1)
        cfg_v.addLayout(status_row)
        right_layout.addWidget(cfg_card)

        plots_row = QtWidgets.QHBoxLayout()
        plots_row.setContentsMargins(0, 0, 0, 0)
        plots_row.setSpacing(8)

        self.erders_tc_plot = pg.PlotWidget()
        self.erders_tc_plot.setBackground("w")
        self.erders_tc_plot.setTitle("试次 Mu/Beta 时程")
        self.erders_tc_plot.setLabel("left", "功率", units="µV²")
        self.erders_tc_plot.setLabel("bottom", "相对 MI(s)")
        self.erders_tc_plot.showGrid(x=True, y=True, alpha=0.18)
        self.erders_tc_plot.addLegend(offset=(6, 6))
        self.erders_mi_line = pg.InfiniteLine(
            pos=0.0, angle=90, pen=pg.mkPen(color=(150, 150, 150), style=QtCore.Qt.DashLine)
        )
        self.erders_tc_plot.addItem(self.erders_mi_line)
        self.erders_tc_mu_curve = self.erders_tc_plot.plot(
            name="Mu", pen=pg.mkPen(color=ERDERS_MU_COLOR, width=2)
        )
        self.erders_tc_beta_curve = self.erders_tc_plot.plot(
            name="Beta", pen=pg.mkPen(color=ERDERS_BETA_COLOR, width=2)
        )
        plots_row.addWidget(SquarePlotCard(self.erders_tc_plot), 1)

        self.erders_avg_plot = pg.PlotWidget()
        self.erders_avg_plot.setBackground("w")
        self.erders_avg_plot.setTitle("平均 ERD (Mu)")
        self.erders_avg_plot.setLabel("left", "ERD", units="%")
        self.erders_avg_plot.setLabel("bottom", "MI 期(s)")
        self.erders_avg_plot.showGrid(x=True, y=True, alpha=0.18)
        self.erders_avg_plot.addLegend(offset=(6, 6))
        self.erders_zero_line = pg.InfiniteLine(
            pos=0.0, angle=0, pen=pg.mkPen(color=(150, 150, 150), style=QtCore.Qt.DashLine)
        )
        self.erders_avg_plot.addItem(self.erders_zero_line)
        self.erders_avg_left_curve = self.erders_avg_plot.plot(
            name="左手", pen=pg.mkPen(color=ERDERS_LEFT_COLOR, width=2)
        )
        self.erders_avg_right_curve = self.erders_avg_plot.plot(
            name="右手", pen=pg.mkPen(color=ERDERS_RIGHT_COLOR, width=2)
        )
        plots_row.addWidget(SquarePlotCard(self.erders_avg_plot), 1)
        right_layout.addLayout(plots_row, 0)

        self.erders_table = QtWidgets.QTableWidget(0, 6)
        self.erders_table.setObjectName("erdersTable")
        self.erders_table.setHorizontalHeaderLabels(
            ["Trial", "方向", "ERD Mu%", "ERD Beta%", "ERS Mu%", "ERS Beta%"]
        )
        self.erders_table.horizontalHeader().setStretchLastSection(True)
        self.erders_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.erders_table.setMaximumHeight(96)
        self.erders_table.setMinimumHeight(80)
        right_layout.addWidget(self.erders_table, 0)

        self.erders_result_label = QtWidgets.QLabel("")
        self.erders_result_label.setObjectName("erdersMetaMono")
        self.erders_result_label.setWordWrap(True)
        self.erders_result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        right_layout.addWidget(self.erders_result_label, 0)
        right_layout.addStretch(1)

        layout.addWidget(right, 45)
        self._refresh_erders_info()
        return tab

    def _erders_set_cue_style(self, color):
        self.erders_cue_label.setStyleSheet(
            f"font-size: 48px; font-weight: bold; color: {color}; background: transparent;"
        )

    def _erders_set_hint_style(self, color):
        self.erders_hint_label.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {color}; background: transparent;"
        )

    def _refresh_erders_info(self):
        if not hasattr(self, "erders_left_spin"):
            return
        n_left = int(self.erders_left_spin.value())
        n_right = int(self.erders_right_spin.value())
        t = self._erders_default_timing()
        total = n_left + n_right
        total_min = total * t.trial_s / 60.0
        schedule = (
            f"- **规模**: 共 {total} 试次 (左 {n_left} / 右 {n_right})\n"
            f"- **时长**: 单试次 {t.trial_s:.0f}s · 预计约 {total_min:.1f} min\n"
            f"- **单试次结构**: 基线 {t.baseline_s:.0f}s + 提示 {t.ready_s:.0f}s + "
            f"想象 {t.mi_s:.0f}s + 放松 {t.relax_s:.0f}s + 间隔 {t.iti_s:.0f}s\n"
            f"- **频带**: Mu 8-13Hz / Beta 13-30Hz\n"
            f"- **任务分析窗**: cue 后 {t.mi_analysis_delay_s:.1f}~{t.mi_s:.1f}s"
        )
        tip = (
            "- **任务**: 左右手随机握拳, 在线计算 **Mu (8-13Hz)** / **Beta (13-30Hz)** 功率\n"
            "- **ERD**: (任务−基线)/基线×100% (**负值** 表示抑制)\n"
            "- **ERS**: (放松−基线)/基线×100% (**正值** 表示增强)\n"
            "- **落盘**: 完整原始 EEG + 事件 marker, 可离线换 Welch/Hilbert/小波/EEGNet\n"
            f"{schedule}"
        )
        if hasattr(self, "erders_info_tip") and self.erders_info_tip is not None:
            self.erders_info_tip.set_info_text(tip)
        if hasattr(self, "erders_info_label") and self.erders_info_label is not None:
            self.erders_info_label.setText("")

    def toggle_erders(self):
        if not self.erders_running:
            self.start_erders()
        else:
            self.stop_erders(finished=False)

    def _build_erders_schedule(self, directions):
        """把方向序列展开成逐阶段列表。"""
        t = self._erders_default_timing()
        phases = [
            (erders.PHASE_BASELINE, t.baseline_s),
            (erders.PHASE_READY, t.ready_s),
            (erders.PHASE_MI, t.mi_s),
            (erders.PHASE_RELAX, t.relax_s),
            (erders.PHASE_ITI, t.iti_s),
        ]
        schedule = []
        for i, direction in enumerate(directions, start=1):
            for kind, dur in phases:
                schedule.append({"trial": i, "direction": direction, "phase": kind, "dur": dur})
        return schedule

    def start_erders(self):
        if not self.is_connected or self.board is None:
            self.erders_status_label.setText("请先连接板卡")
            return
        if self.is_debug_virtual:
            self.erders_status_label.setText("虚拟调试端口不支持 ERD/ERS(需真实 EEG)")
            return

        # 与其他活动互斥
        for stop_fn, flag in (
            (self.stop_stream, "is_streaming"),
            (self.stop_impedance, "imp_running"),
            (self.stop_focus2, "focus2_running"),
            (self.stop_wearing_exp, "wearing_exp_running"),        ):
            if getattr(self, flag, False):
                stop_fn()
        if self.acquire_running:
            self.stop_acquire(finished=False)
        if self.test_running:
            self.stop_test(finished=False)

        n_left = int(self.erders_left_spin.value())
        n_right = int(self.erders_right_spin.value())
        self.erders_config = erders.ErdErsConfig(n_trials_left=n_left, n_trials_right=n_right)
        directions = erders.generate_directions(n_left, n_right)
        self.erders_schedule = self._build_erders_schedule(directions)

        # 重置缓冲与结果
        self.erders_ts_chunks = []
        self.erders_eeg_chunks = []
        self.erders_phase_marks = []
        self.erders_results = []
        self.erders_left_mu_curves = []
        self.erders_right_mu_curves = []
        self.erders_table.setRowCount(0)
        self.erders_tc_mu_curve.setData([], [])
        self.erders_tc_beta_curve.setData([], [])
        self.erders_avg_left_curve.setData([], [])
        self.erders_avg_right_curve.setData([], [])
        self.erders_result_label.setText("")

        try:
            self.board.start_stream()
            self.board.get_board_data()   # 清空既有缓冲
        except (BrainFlowError, SerialBoardError) as e:
            self.erders_status_label.setText(f"启动失败 - {e}")
            return

        self.erders_running = True
        self.erders_phase_idx = 0
        now_ms = time.time() * 1000.0
        self.erders_phase_start_ms = now_ms
        self._erders_enter_phase(now_ms)

        self.erders_btn.setText("停止")
        self.connect_btn.setEnabled(False)
        self.erders_left_spin.setEnabled(False)
        self.erders_right_spin.setEnabled(False)
        self.erders_c3_combo.setEnabled(False)
        self.erders_c4_combo.setEnabled(False)
        self._erders_set_other_buttons(False)

        total = len(directions)
        total_min = total * self._erders_default_timing().trial_s / 60.0
        self.erders_status_label.setText(
            f"实验进行中 · {total} 试次 · 约 {total_min:.1f} min"
        )
        self.erders_timer.start(ERDERS_TICK_MS)

    def _erders_set_other_buttons(self, enabled):
        """统一启停其他活动按钮(避免抢占板卡)。"""
        for name in (
            "start_btn", "imp_btn", "focus2_btn", "focus2_calib_btn",
            "wearing_exp_btn", "acquire_btn", "test_acquire_btn", "validate_btn",
        ):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(enabled)

    def _erders_enter_phase(self, now_ms):
        """进入当前阶段: 登记 mark, 刷新显示。"""
        ph = self.erders_schedule[self.erders_phase_idx]
        self.erders_phase_marks.append({
            "trial": ph["trial"],
            "direction": ph["direction"],
            "phase": ph["phase"],
            "start_ms": now_ms,
            "end_ms": None,
        })
        self._erders_update_display(ph, ph["dur"])

    def update_erders(self):
        if not self.erders_running or self.board is None:
            return

        # 1) 取板卡新数据(原始 µV, 不滤波, 便于离线复算), 累积
        data = self.board.get_board_data()
        if data.shape[1] > 0:
            ts_ms = data[self.timestamp_channel] * 1000.0
            eeg = data[self.eeg_channels, :].T
            self.erders_ts_chunks.append(np.asarray(ts_ms, dtype=np.float64))
            self.erders_eeg_chunks.append(np.asarray(eeg, dtype=np.float64))

        # 2) 推进阶段时钟
        now_ms = time.time() * 1000.0
        cur = self.erders_schedule[self.erders_phase_idx]
        elapsed = now_ms - self.erders_phase_start_ms
        if elapsed >= cur["dur"] * 1000.0:
            # 结束当前阶段
            if self.erders_phase_marks:
                self.erders_phase_marks[-1]["end_ms"] = now_ms
            # 若刚结束的是 relax, 该试次数据已齐, 立即分析并刷新曲线
            if cur["phase"] == erders.PHASE_RELAX:
                self._erders_analyze_trial(cur["trial"], cur["direction"])
            self.erders_phase_idx += 1
            if self.erders_phase_idx >= len(self.erders_schedule):
                self.stop_erders(finished=True)
                return
            self.erders_phase_start_ms = now_ms
            self._erders_enter_phase(now_ms)
            cur = self.erders_schedule[self.erders_phase_idx]
            elapsed = 0.0

        remaining = max(0.0, cur["dur"] - elapsed / 1000.0)
        self._erders_update_display(cur, remaining)

    def _erders_update_display(self, ph, remaining):
        kind = ph["phase"]
        direction = ph["direction"]
        trial = ph["trial"]
        total = self.erders_config.n_trials
        if kind == erders.PHASE_BASELINE:
            self.erders_cue_label.setText("请放松")
            self._erders_set_cue_style(ERDERS_CUE_COLOR_BASELINE)
            self.erders_hint_label.setText("保持不动，记录基线")
            self._erders_set_hint_style(MEDICAL_PALETTE["muted"])
        elif kind == erders.PHASE_READY:
            self.erders_cue_label.setText("+")
            self._erders_set_cue_style(ERDERS_CUE_COLOR_READY)
            self.erders_hint_label.setText("请准备")
            self._erders_set_hint_style(MEDICAL_PALETTE["text"])
        elif kind == erders.PHASE_MI:
            if direction == erders.DIR_LEFT:
                self.erders_cue_label.setText("← 左手")
            else:
                self.erders_cue_label.setText("右手 →")
            self._erders_set_cue_style(ERDERS_CUE_COLOR_MI)
            self.erders_hint_label.setText("请持续握拳")
            self._erders_set_hint_style(MEDICAL_PALETTE["text"])
        elif kind == erders.PHASE_RELAX:
            self.erders_cue_label.setText("Relax")
            self._erders_set_cue_style(ERDERS_CUE_COLOR_RELAX)
            self.erders_hint_label.setText("请放松，停止握拳")
            self._erders_set_hint_style(MEDICAL_PALETTE["muted"])
        else:  # ITI
            self.erders_cue_label.setText("+")
            self._erders_set_cue_style(ERDERS_CUE_COLOR_ITI)
            self.erders_hint_label.setText("准备下一试次")
            self._erders_set_hint_style(MEDICAL_PALETTE["muted"])

        dir_cn = erders.DIRECTION_NAMES_CN.get(direction, direction)
        self.erders_progress_label.setText(
            f"Trial {trial} / {total} · 当前: {dir_cn} · 倒计时 {remaining:.1f}s"
        )

    def _erders_concat_buffers(self):
        if not self.erders_ts_chunks:
            return np.array([]), np.zeros((0, len(self.eeg_channels)))
        ts = np.concatenate(self.erders_ts_chunks)
        eeg = np.concatenate(self.erders_eeg_chunks, axis=0)
        return ts, eeg

    def _erders_marks_for(self, trial, phase):
        for m in self.erders_phase_marks:
            if m["trial"] == trial and m["phase"] == phase:
                return m
        return None

    def _erders_analysis_channel(self, direction):
        """对侧分析通道列索引: 右手→C3(左脑), 左手→C4(右脑)。"""
        if direction == erders.DIR_RIGHT:
            return int(self.erders_c3_combo.currentIndex())
        return int(self.erders_c4_combo.currentIndex())

    def _erders_analyze_trial(self, trial, direction):
        ts, eeg = self._erders_concat_buffers()
        if ts.size == 0:
            return
        t = self._erders_default_timing()
        ch = self._erders_analysis_channel(direction)
        if ch >= eeg.shape[1]:
            ch = 0

        m_base = self._erders_marks_for(trial, erders.PHASE_BASELINE)
        m_mi = self._erders_marks_for(trial, erders.PHASE_MI)
        m_relax = self._erders_marks_for(trial, erders.PHASE_RELAX)
        if not (m_base and m_mi and m_relax):
            return

        base_sig = erders.slice_by_time(ts, eeg, m_base["start_ms"], m_base["end_ms"] or m_base["start_ms"])
        # MI 分析窗: cue 后延迟 ~ MI 结束
        mi_start = m_mi["start_ms"] + t.mi_analysis_delay_s * 1000.0
        mi_end = m_mi["end_ms"] or (m_mi["start_ms"] + t.mi_s * 1000.0)
        task_sig = erders.slice_by_time(ts, eeg, mi_start, mi_end)
        relax_sig = erders.slice_by_time(ts, eeg, m_relax["start_ms"], m_relax["end_ms"] or m_relax["start_ms"])

        base_ch = base_sig[:, ch] if base_sig.shape[0] else np.array([])
        task_ch = task_sig[:, ch] if task_sig.shape[0] else np.array([])
        relax_ch = relax_sig[:, ch] if relax_sig.shape[0] else np.array([])

        res = erders.analyze_trial(
            trial, direction, base_ch, task_ch, relax_ch, self.sampling_rate, self.erders_config
        )
        self.erders_results.append(res)
        self._erders_append_table_row(res)

        # 当前试次频带功率时程(整段, 对侧通道), x 相对 MI 起点
        full = erders.slice_by_time(ts, eeg, m_base["start_ms"], m_relax["end_ms"] or m_relax["start_ms"])
        if full.shape[0] > 0:
            full_ch = full[:, ch]
            offset = -(t.baseline_s + t.ready_s)
            tt_mu, vv_mu = erders.band_power_timecourse(full_ch, self.sampling_rate, self.erders_config.mu_band)
            tt_be, vv_be = erders.band_power_timecourse(full_ch, self.sampling_rate, self.erders_config.beta_band)
            if tt_mu.size:
                self.erders_tc_mu_curve.setData(tt_mu + offset, vv_mu)
            if tt_be.size:
                self.erders_tc_beta_curve.setData(tt_be + offset, vv_be)

        # MI 期 mu ERD 时程, 累积到左/右平均
        if task_ch.size:
            tt, vv = erders.band_power_timecourse(task_ch, self.sampling_rate, self.erders_config.mu_band)
            if tt.size:
                erd_curve = np.array([erders.erd_ers_percent(res.baseline_mu, p) for p in vv])
                tt_rel = tt + t.mi_analysis_delay_s
                if direction == erders.DIR_LEFT:
                    self.erders_left_mu_curves.append((tt_rel, erd_curve))
                else:
                    self.erders_right_mu_curves.append((tt_rel, erd_curve))
        self._erders_refresh_avg_curves()

    def _erders_append_table_row(self, res):
        row = self.erders_table.rowCount()
        self.erders_table.insertRow(row)
        dir_cn = erders.DIRECTION_NAMES_CN.get(res.direction, res.direction)
        vals = [
            str(res.trial), dir_cn,
            f"{res.erd_mu:.1f}", f"{res.erd_beta:.1f}",
            f"{res.ers_mu:.1f}", f"{res.ers_beta:.1f}",
        ]
        for col, v in enumerate(vals):
            item = QtWidgets.QTableWidgetItem(v)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.erders_table.setItem(row, col, item)
        self.erders_table.scrollToBottom()

    def _erders_refresh_avg_curves(self):
        lt, lv = erders.average_erd_curve(self.erders_left_mu_curves)
        rt, rv = erders.average_erd_curve(self.erders_right_mu_curves)
        self.erders_avg_left_curve.setData(lt, lv)
        self.erders_avg_right_curve.setData(rt, rv)

    def stop_erders(self, finished=False):
        if not self.erders_running:
            return
        self.erders_timer.stop()
        try:
            if self.board is not None:
                self.board.stop_stream()
        except (BrainFlowError, SerialBoardError):
            pass
        self.erders_running = False

        self.erders_btn.setText("开始 ERD/ERS 实验")
        self.connect_btn.setEnabled(True)
        self.erders_left_spin.setEnabled(True)
        self.erders_right_spin.setEnabled(True)
        self.erders_c3_combo.setEnabled(True)
        self.erders_c4_combo.setEnabled(True)
        if self.is_connected and not self.is_debug_virtual:
            self._erders_set_other_buttons(True)

        if finished:
            self.erders_cue_label.setText("完成")
            self._erders_set_cue_style("#1a7f37")
            self.erders_hint_label.setText("实验结束，正在保存...")
            self._erders_set_hint_style(MEDICAL_PALETTE["muted"])
        else:
            self.erders_cue_label.setText("已停止")
            self._erders_set_cue_style("#888")
            self.erders_hint_label.setText("实验被手动停止")
            self._erders_set_hint_style(MEDICAL_PALETTE["muted"])

        if self.erders_results:
            self._save_erders(finished=finished)
        else:
            self.erders_status_label.setText("已停止 · 无完整试次, 未保存")

    def _save_erders(self, finished=True):
        session_id = new_session_id("ERDERS")
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ERDERS_RECORD_DIR)
        folder = os.path.join(base_dir, session_id)
        os.makedirs(folder, exist_ok=True)

        # 1) 完整原始 EEG + 事件 marker(逐样本标注 trial/direction/phase), 便于离线复算
        raw_path = os.path.join(folder, f"{session_id}_raw.csv")
        try:
            self._erders_write_raw_csv(raw_path)
        except Exception as e:  # noqa: BLE001
            self.erders_status_label.setText("原始数据保存失败")
            self.erders_result_label.setText(f"错误: {e}")
            return

        # 2) trial_results.csv
        results_path = os.path.join(folder, "trial_results.csv")
        self._erders_write_results_csv(results_path)

        # 3) 平均 ERD 曲线 CSV
        avg_path = os.path.join(folder, "avg_erd_mu_curve.csv")
        self._erders_write_avg_curve_csv(avg_path)

        # 4) 图表 PNG(matplotlib 兜底, 缺失则跳过)
        plot_msg = self._erders_export_plots(folder, session_id)

        # 5) meta
        meta = {
            "session_id": session_id,
            "paradigm": "ERD_ERS_motor_execution",
            "unit": UNIT_MICROVOLT,
            "sample_rate": self.sampling_rate,
            "num_channels": len(self.eeg_channels),
            **self._board_profile_meta(),
            "n_trials": len(self.erders_results),
            "finished": bool(finished),
            "timing": {
                "baseline_s": self.erders_config.timing.baseline_s,
                "ready_s": self.erders_config.timing.ready_s,
                "mi_s": self.erders_config.timing.mi_s,
                "relax_s": self.erders_config.timing.relax_s,
                "iti_s": self.erders_config.timing.iti_s,
                "mi_analysis_delay_s": self.erders_config.timing.mi_analysis_delay_s,
            },
            "bands": {"mu": list(self.erders_config.mu_band), "beta": list(self.erders_config.beta_band)},
            "c3_channel": f"CH{self.erders_c3_combo.currentIndex() + 1}",
            "c4_channel": f"CH{self.erders_c4_combo.currentIndex() + 1}",
            "phase_markers": erders.PHASE_MARKERS,
            "mi_direction_markers": erders.MI_DIRECTION_MARKERS,
        }
        with open(os.path.join(folder, f"{session_id}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        n = len(self.erders_results)
        mean_erd_mu = float(np.mean([r.erd_mu for r in self.erders_results])) if n else 0.0
        mean_ers_beta = float(np.mean([r.ers_beta for r in self.erders_results])) if n else 0.0
        self.erders_status_label.setText(f"已保存 {n} 试次 → {folder}")
        self.erders_result_label.setText(
            f"目录: {folder}\n"
            f"完整试次: {n} · 平均 ERD(Mu): {mean_erd_mu:.1f}% · 平均 ERS(Beta): {mean_ers_beta:.1f}%\n"
            f"文件: {os.path.basename(raw_path)} / trial_results.csv / avg_erd_mu_curve.csv\n"
            f"{plot_msg}\n"
            "提示: 原始 EEG + marker 已完整保存, 可离线换算法重新分析。"
        )

    def _erders_write_raw_csv(self, path):
        ts, eeg = self._erders_concat_buffers()
        if ts.size == 0:
            raise RuntimeError("没有录到任何 EEG 样本")
        n, n_ch = eeg.shape
        # 逐样本标注 trial/direction/phase/marker(按时间落在哪个阶段窗内)
        trial_arr = np.zeros(n, dtype=int)
        dir_arr = np.array([""] * n, dtype=object)
        phase_arr = np.array([""] * n, dtype=object)
        marker_arr = np.zeros(n, dtype=int)
        for m in self.erders_phase_marks:
            end_ms = m["end_ms"] if m["end_ms"] is not None else ts[-1] + 1
            mask = (ts >= m["start_ms"]) & (ts < end_ms)
            if not np.any(mask):
                continue
            trial_arr[mask] = m["trial"]
            dir_arr[mask] = m["direction"]
            phase_arr[mask] = m["phase"]
            if m["phase"] == erders.PHASE_MI:
                marker_arr[mask] = erders.MI_DIRECTION_MARKERS.get(m["direction"], 0)
            else:
                marker_arr[mask] = erders.PHASE_MARKERS.get(m["phase"], 0)

        eeg_cols = [f"EEG Channel {i + 1}" for i in range(n_ch)]
        header = ["timestamp", "trial", "direction", "phase", "marker"] + eeg_cols
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(n):
                w.writerow(
                    [f"{ts[i]:.4f}", int(trial_arr[i]), dir_arr[i], phase_arr[i], int(marker_arr[i])]
                    + [f"{eeg[i, c]:.4f}" for c in range(n_ch)]
                )

    def _erders_write_results_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=erders.TRIAL_RESULTS_COLUMNS)
            w.writeheader()
            for res in self.erders_results:
                row = res.to_row()
                # 数值保留 4 位
                for k, v in row.items():
                    if isinstance(v, float):
                        row[k] = round(v, 4)
                w.writerow(row)

    def _erders_write_avg_curve_csv(self, path):
        lt, lv = erders.average_erd_curve(self.erders_left_mu_curves)
        rt, rv = erders.average_erd_curve(self.erders_right_mu_curves)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "left_erd_mu_pct", "right_erd_mu_pct"])
            n = max(lt.size, rt.size)
            for i in range(n):
                tval = lt[i] if i < lt.size else (rt[i] if i < rt.size else "")
                lval = f"{lv[i]:.4f}" if i < lv.size else ""
                rval = f"{rv[i]:.4f}" if i < rv.size else ""
                w.writerow([f"{tval:.4f}" if tval != "" else "", lval, rval])

    def _erders_export_plots(self, folder, session_id):
        """用 matplotlib 输出汇总图; 不可用时返回提示(不影响数据落盘)。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return "图表: 未安装 matplotlib, 已跳过 PNG(原始数据可离线绘图)。"

        try:
            results = self.erders_results
            trials = [r.trial for r in results]
            # 图1: 每试次 ERD/ERS 柱
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(trials, [r.erd_mu for r in results], "o-", label="ERD Mu", color="#28a05a")
            ax.plot(trials, [r.erd_beta for r in results], "s-", label="ERD Beta", color="#c83c28")
            ax.plot(trials, [r.ers_beta for r in results], "^-", label="ERS Beta", color="#1668c0")
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel("Trial"); ax.set_ylabel("%"); ax.legend(); ax.set_title("Per-trial ERD/ERS")
            fig.tight_layout(); fig.savefig(os.path.join(folder, f"{session_id}_erders_per_trial.png"), dpi=120)
            plt.close(fig)

            # 图2: 平均 ERD(Mu) 曲线 左/右
            lt, lv = erders.average_erd_curve(self.erders_left_mu_curves)
            rt, rv = erders.average_erd_curve(self.erders_right_mu_curves)
            fig, ax = plt.subplots(figsize=(10, 4))
            if lt.size:
                ax.plot(lt, lv, label="Left", color="#1668c0")
            if rt.size:
                ax.plot(rt, rv, label="Right", color="#c05028")
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel("MI time (s)"); ax.set_ylabel("ERD Mu %"); ax.legend()
            ax.set_title("Average ERD (Mu)")
            fig.tight_layout(); fig.savefig(os.path.join(folder, f"{session_id}_avg_erd_mu.png"), dpi=120)
            plt.close(fig)
            return "图表: 已输出 per_trial / avg_erd_mu PNG。"
        except Exception as e:  # noqa: BLE001
            return f"图表: 生成失败({e}), 原始数据可离线绘图。"

    def _build_validate_tab(self):
        """信号校验: 与采集范式相同的 BrainFlow 取数, 固定时长连续录制, 单通道波形回看(降采样显示)。

        用途: 接入已知信号(如正弦波)后录 10s, 切换通道检查波形是否正常。
        """
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        header = QtWidgets.QFrame()
        header.setObjectName("validateHeader")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        self.validate_info_tip = self._add_page_title(
            header_layout,
            "信号校验",
            self._validate_page_intro_text(),
            object_name="validatePageTitle",
            font_size=18,
        )
        left_layout.addWidget(header)

        control = QtWidgets.QFrame()
        control.setObjectName("validateControlBar")
        ctrl = QtWidgets.QHBoxLayout(control)
        ctrl.setContentsMargins(14, 10, 14, 10)
        ctrl.setSpacing(8)
        ctrl.addWidget(QtWidgets.QLabel("录制时长(s)"))
        self.validate_duration_spin = QtWidgets.QDoubleSpinBox()
        self.validate_duration_spin.setRange(1.0, 120.0)
        self.validate_duration_spin.setSingleStep(1.0)
        self.validate_duration_spin.setValue(VALIDATE_DEFAULT_SECONDS)
        self.validate_duration_spin.setMinimumWidth(90)
        ctrl.addWidget(self.validate_duration_spin)

        ctrl.addSpacing(8)
        ctrl.addWidget(QtWidgets.QLabel("显示通道"))
        self.validate_channel_combo = QtWidgets.QComboBox()
        self.validate_channel_combo.addItems([f"CH{c + 1}" for c in range(16)])
        self.validate_channel_combo.setMinimumWidth(80)
        self.validate_channel_combo.currentIndexChanged.connect(self._validate_refresh_plot)
        ctrl.addWidget(self.validate_channel_combo)

        ctrl.addSpacing(8)
        ctrl.addWidget(QtWidgets.QLabel("Y 轴范围(µV)"))
        self.validate_yrange_spin = QtWidgets.QDoubleSpinBox()
        self.validate_yrange_spin.setRange(10.0, 200000.0)
        self.validate_yrange_spin.setSingleStep(50.0)
        self.validate_yrange_spin.setValue(VALIDATE_Y_RANGE_UV)
        self.validate_yrange_spin.setMinimumWidth(100)
        self.validate_yrange_spin.valueChanged.connect(self._validate_apply_yrange)
        ctrl.addWidget(self.validate_yrange_spin)
        ctrl.addStretch()
        left_layout.addWidget(control)

        plot_card = QtWidgets.QFrame()
        plot_card.setObjectName("validatePlotCard")
        plot_layout = QtWidgets.QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(12, 12, 12, 10)
        plot_layout.setSpacing(6)

        self.validate_empty_hint = QtWidgets.QLabel(
            "连接板卡后点击「开始校验采集」，或从文件加载波形进行回看"
        )
        self.validate_empty_hint.setObjectName("validatePlotHint")
        self.validate_empty_hint.setAlignment(QtCore.Qt.AlignCenter)
        plot_layout.addWidget(self.validate_empty_hint)

        self.validate_plot = pg.PlotWidget()
        self.validate_plot.setBackground("w")
        self.validate_plot.setMinimumHeight(300)
        self.validate_plot.setMenuEnabled(False)
        self.validate_plot.showGrid(x=True, y=True, alpha=0.18)
        self.validate_plot.setLabel("left", "电压", units="µV")
        self.validate_plot.setLabel("bottom", "时间", units="s")
        self.validate_plot.setXRange(0, VALIDATE_DEFAULT_SECONDS, padding=0.02)
        self.validate_plot.setYRange(-VALIDATE_Y_RANGE_UV, VALIDATE_Y_RANGE_UV)
        self.validate_curve = self.validate_plot.plot(
            pen=pg.mkPen(color=(22, 104, 192), width=1.5)
        )
        plot_layout.addWidget(self.validate_plot, 1)

        self.validate_plot_info_label = QtWidgets.QLabel("波形: 等待录制或加载文件")
        self.validate_plot_info_label.setObjectName("validatePlotInfo")
        plot_layout.addWidget(self.validate_plot_info_label)
        left_layout.addWidget(plot_card, 1)

        layout.addWidget(left, 4)

        right = QtWidgets.QFrame()
        right.setObjectName("validateSidePanel")
        right.setMinimumWidth(300)
        right.setMaximumWidth(360)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        side_title = QtWidgets.QLabel("操作与结果")
        side_title.setObjectName("validateSideTitle")
        right_layout.addWidget(side_title)

        self.validate_btn = QtWidgets.QPushButton("开始 10 秒校验采集")
        self.validate_btn.setObjectName("btnPrimary")
        self.validate_btn.setEnabled(False)
        self.validate_btn.setMinimumHeight(36)
        self.validate_btn.clicked.connect(self.toggle_validate)
        right_layout.addWidget(self.validate_btn)

        self.validate_reload_btn = QtWidgets.QPushButton("从文件加载波形…")
        self.validate_reload_btn.setObjectName("btnSecondary")
        self.validate_reload_btn.setMinimumHeight(34)
        self.validate_reload_btn.clicked.connect(self._validate_load_file)
        right_layout.addWidget(self.validate_reload_btn)

        meta = QtWidgets.QFrame()
        meta.setObjectName("deviceInline")
        meta_layout = QtWidgets.QVBoxLayout(meta)
        meta_layout.setContentsMargins(2, 4, 2, 4)
        meta_layout.setSpacing(6)

        def _meta_row(label_text, value_widget):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            lbl = QtWidgets.QLabel(label_text)
            lbl.setObjectName("validateMetaLabel")
            lbl.setFixedWidth(56)
            row.addWidget(lbl)
            row.addWidget(value_widget, 1)
            meta_layout.addLayout(row)

        self.validate_status_label = QtWidgets.QLabel("未连接")
        self.validate_status_label.setObjectName("validateMetaValue")
        _meta_row("状态", self.validate_status_label)

        self.validate_progress_label = QtWidgets.QLabel("---")
        self.validate_progress_label.setObjectName("validateMetaMono")
        self.validate_progress_label.setWordWrap(True)
        _meta_row("进度", self.validate_progress_label)

        self.validate_file_label = QtWidgets.QLabel("---")
        self.validate_file_label.setObjectName("validateMetaMono")
        self.validate_file_label.setWordWrap(True)
        _meta_row("文件", self.validate_file_label)
        right_layout.addWidget(meta)

        stat_title = QtWidgets.QLabel("各通道统计 (µV)")
        stat_title.setObjectName("validateSideTitle")
        right_layout.addWidget(stat_title)

        self.validate_stats_view = QtWidgets.QPlainTextEdit()
        self.validate_stats_view.setObjectName("validateStatsView")
        self.validate_stats_view.setReadOnly(True)
        self.validate_stats_view.setMaximumBlockCount(200)
        self.validate_stats_view.setPlaceholderText("录制或加载完成后，此处显示各通道均值 / 峰峰值等统计。")
        right_layout.addWidget(self.validate_stats_view, 1)

        layout.addWidget(right, 1)
        return tab

    def _validate_apply_yrange(self):
        y = float(self.validate_yrange_spin.value())
        self.validate_plot.setYRange(-y, y)

    def _validate_cache_plot_data(self, eeg, eeg_names, sample_rate=None):
        """缓存完整数据，供切换通道时快速重绘。"""
        if sample_rate is None:
            sample_rate = self.sampling_rate
        self.validate_plot_eeg = np.asarray(eeg, dtype=np.float64)
        self.validate_plot_eeg_names = list(eeg_names)
        self.validate_plot_sample_rate = int(sample_rate)

    def _validate_refresh_plot(self):
        """从缓存重绘当前所选通道（降采样）。"""
        if self.validate_plot_eeg is None:
            return
        self._plot_validate_waveforms(self.validate_plot_eeg, self.validate_plot_eeg_names)

    def _validate_load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载校验 CSV", "", "CSV 文件 (*.csv);;所有文件 (*.*)"
        )
        if path:
            try:
                data = load_validation_csv(path)
                self._cache_and_plot_validate(
                    data["eeg"], data["eeg_names"], sample_rate=data["sample_rate"]
                )
                self.validate_file_label.setText(path)
                self.validate_status_label.setText("已从文件加载")
                self._validate_show_stats(data["eeg"], data["eeg_names"])
            except Exception as e:  # noqa: BLE001
                self.validate_status_label.setText("加载失败")
                self.validate_stats_view.setPlainText(str(e))

    def toggle_validate(self):
        if not self.validate_running:
            self.start_validate()
        else:
            self.stop_validate(finished=False)

    def start_validate(self):
        if not self.is_connected or self.board is None:
            self.validate_status_label.setText("请先连接板卡")
            return
        if self.is_streaming:
            self.stop_stream()
        if self.acquire_running:
            self.stop_acquire(finished=False)
        if self.test_running:
            self.stop_test(finished=False)
        if self.imp_running:
            self.stop_impedance()
        if self.focus2_running:
            self.stop_focus2()
        if self.wearing_exp_running:
            self.stop_wearing_exp()

        duration = float(self.validate_duration_spin.value())
        self.validate_recorder = RawStreamRecorder(
            num_channels=len(self.eeg_channels), sample_rate=self.sampling_rate
        )
        self.validate_btn.setText(f"停止 ({duration:.0f}s)")
        self.validate_duration_spin.setEnabled(False)

        try:
            self.board.start_stream()
            self.board.get_board_data()
        except (BrainFlowError, SerialBoardError) as e:
            self.validate_status_label.setText(f"启动失败 - {e}")
            self.validate_btn.setText("开始 10 秒校验采集")
            self.validate_duration_spin.setEnabled(True)
            return

        self.validate_running = True
        self.validate_end_ms = time.time() * 1000.0 + duration * 1000.0
        self.validate_status_label.setText(f"录制中 ({duration:.0f}s)…")
        self.validate_stats_view.clear()
        self.validate_file_label.setText("(录制中)")
        hint = getattr(self, "validate_empty_hint", None)
        if hint is not None:
            hint.hide()
        self.connect_btn.setEnabled(False)
        self.acquire_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.validate_timer.start(VALIDATE_TICK_MS)

    def update_validate(self):
        if not self.validate_running or self.board is None:
            return

        data = self.board.get_board_data()
        if data.shape[1] > 0:
            ts_ms = data[self.timestamp_channel] * 1000.0
            eeg = self._eeg_from_board_filtered(data)
            self.validate_recorder.add_samples(ts_ms, eeg)

        now_ms = time.time() * 1000.0
        remain = max(0.0, (self.validate_end_ms - now_ms) / 1000.0)
        self.validate_progress_label.setText(
            f"剩余 {remain:.1f}s · 已录 {self.validate_recorder.n_samples} 点"
        )

        if now_ms >= self.validate_end_ms:
            self.stop_validate(finished=True)

    def stop_validate(self, finished=False):
        if not self.validate_running:
            return
        self.validate_timer.stop()
        try:
            if self.board is not None:
                self.board.stop_stream()
        except (BrainFlowError, SerialBoardError):
            pass
        self.validate_running = False
        duration = float(self.validate_duration_spin.value())
        self.validate_btn.setText("开始 10 秒校验采集")
        self.validate_duration_spin.setEnabled(True)
        self.connect_btn.setEnabled(True)
        if self.is_connected:
            self.acquire_btn.setEnabled(True)
            self.start_btn.setEnabled(True)

        if finished and self.validate_recorder is not None:
            self._save_and_plot_validate(duration)
        else:
            self.validate_status_label.setText("已手动停止")

    def _save_and_plot_validate(self, duration_s):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, VALIDATE_RECORD_DIR)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"Validate_{ts}.csv")

        try:
            stats = self.validate_recorder.finish(
                out_path, duration_s=duration_s, extra_meta=self._recording_meta_filter_note(),
                log_fn=lambda m: None,
            )
        except Exception as e:  # noqa: BLE001
            self.validate_status_label.setText("保存失败")
            self.validate_stats_view.setPlainText(str(e))
            return

        self.validate_file_label.setText(str(stats["output_path"]))
        self.validate_status_label.setText(
            f"完成 · {stats['n_samples']} 点 · {stats['duration_s_actual']:.2f}s"
        )
        self._cache_and_plot_validate(
            stats["eeg"], stats["eeg_names"], sample_rate=self.sampling_rate
        )
        self._validate_show_stats(stats["eeg"], stats["eeg_names"])

    def _cache_and_plot_validate(self, eeg, eeg_names, sample_rate=None):
        """缓存完整数据并绘制当前所选通道。"""
        self._validate_cache_plot_data(eeg, eeg_names, sample_rate=sample_rate)
        self._validate_refresh_plot()

    def _plot_validate_waveforms(self, eeg, eeg_names):
        n = eeg.shape[0]
        sr = self.validate_plot_sample_rate
        t_plot = uniform_time_s(n, sr)
        t_max = float(t_plot[-1]) if n > 0 else VALIDATE_DEFAULT_SECONDS
        y = float(self.validate_yrange_spin.value())
        ch_idx = self.validate_channel_combo.currentIndex()
        if ch_idx < 0 or ch_idx >= eeg.shape[1]:
            ch_idx = 0
        ch_name = eeg_names[ch_idx] if ch_idx < len(eeg_names) else f"CH{ch_idx + 1}"
        t_ds, y_ds, n_orig, n_disp = downsample_for_plot(
            t_plot, eeg[:, ch_idx], max_points=VALIDATE_MAX_PLOT_POINTS
        )
        self.validate_curve.setData(t_ds, y_ds)
        self.validate_plot.setXRange(0, max(0.1, t_max), padding=0.02)
        self.validate_plot.setYRange(-y, y)
        hint = getattr(self, "validate_empty_hint", None)
        if hint is not None:
            hint.hide()
        self.validate_plot_info_label.setText(
            f"{ch_name} · {sr} Hz · 与 CSV 一致 · {n_orig} 点 → 显示 {n_disp} 点"
        )
    def _validate_show_stats(self, eeg, eeg_names):
        lines = []
        for i, name in enumerate(eeg_names):
            ch = eeg[:, i]
            saturated = np.mean(np.abs(ch) > 0.9 * ADS1299_FULLSCALE_UV) * 100
            flat = float(np.std(ch)) < 0.5
            flag = []
            if saturated > 5:
                flag.append(f"饱和{saturated:.0f}%")
            if flat:
                flag.append("近乎平线")
            extra = f"  ⚠ {','.join(flag)}" if flag else ""
            lines.append(
                f"{name:16s}  min={np.min(ch):9.1f}  max={np.max(ch):9.1f}  "
                f"std={np.std(ch):8.2f}{extra}"
            )
        self.validate_stats_view.setPlainText("\n".join(lines))

    # ---------------- 串口 ----------------
    def refresh_ports(self):
        prev = self.port_combo.currentData()
        self.port_combo.clear()
        if is_debug_mode():
            self.port_combo.addItem(DEBUG_VIRTUAL_PORT_LABEL, DEBUG_VIRTUAL_PORT)
        ports = list_ports.comports()
        if not ports:
            self.port_combo.addItem("未检测到串口", None)
        else:
            for p in ports:
                self.port_combo.addItem(f"{p.device} - {p.description}", p.device)
        if prev is not None:
            idx = self.port_combo.findData(prev)
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)

    def get_selected_port(self):
        data = self.port_combo.currentData()
        if data == DEBUG_VIRTUAL_PORT:
            return DEBUG_VIRTUAL_PORT
        if data:
            return data
        # 兼容"未检测到串口"占位项
        text = self.port_combo.currentText()
        return text.split(" - ")[0] if " - " in text else None

    def _connect_debug_virtual(self):
        """连接虚拟调试端口：可预览采集/测试范式动画，不落盘。"""
        n_rows = BoardShim.get_num_rows(BOARD_ID)
        self.board = VirtualDebugBoard(
            n_rows,
            self.sampling_rate,
            self.timestamp_channel,
            self.eeg_channels,
            tick_ms=ACQUIRE_TICK_MS,
        )
        self.board.prepare_session()
        self.is_debug_virtual = True
        self.is_connected = True
        self.connect_btn.setText("断开")
        self.acquire_btn.setEnabled(True)
        if hasattr(self, "test_acquire_btn"):
            self.test_acquire_btn.setEnabled(True)
        self.port_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.hw_checkbox.setEnabled(False)
        self._set_app_status(
            f"状态: {DEBUG_VIRTUAL_PORT_LABEL} 已连接 · 通道 {self.sampling_rate} Hz · "
            "仅预览范式动画，不保存任何数据"
        )

    def toggle_connect(self):
        if self._connect_worker_busy():
            return
        if not self.is_connected:
            self.connect_board()
        else:
            if self._session_busy():
                if not self._confirm_stop_activity(
                    "断开连接",
                    "仍有采集或检测在进行。确定断开并停止全部活动吗？",
                ):
                    return
            self._save_ui_settings()
            self.disconnect_board()

    def connect_board(self):
        if self._connect_worker_busy():
            return
        port = self.get_selected_port()
        if port == DEBUG_VIRTUAL_PORT:
            try:
                self._connect_debug_virtual()
            except Exception as e:  # noqa: BLE001
                self.board = None
                self.is_debug_virtual = False
                self._set_app_status(f"状态: 虚拟端口连接失败 - {e}")
            return
        if not port or self.port_combo.currentData() is None:
            self._set_app_status("状态: 请先选择有效串口")
            return

        self._on_baud_combo_changed()
        self._connect_port = port
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("连接中...")
        self.port_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.hw_checkbox.setEnabled(False)
        self.baud_combo.setEnabled(False)
        self._show_connect_progress("正在打开串口...", indeterminate=True)

        self._connect_board_worker = BoardConnectWorker(
            port, self.hw_checkbox.isChecked(), self.connect_baud_rate, parent=self,
        )
        self._connect_board_worker.progress.connect(self._on_connect_progress)
        self._connect_board_worker.finished_ok.connect(self._on_connect_ok)
        self._connect_board_worker.failed.connect(self._on_connect_failed)
        self._connect_board_worker.start()

    def _safe_config_board(self, cmd):
        """向板卡发送配置命令; 忽略 BrainFlow 解析响应时的 UTF-8 解码错误。"""
        safe_config_board(self.board, cmd)

    def _imp_worker_busy(self):
        return self._imp_board_worker is not None and self._imp_board_worker.isRunning()

    def _on_imp_board_progress(self, msg):
        self._set_app_status(f"状态: {msg}")
        if hasattr(self, "imp_summary_label"):
            self.imp_summary_label.setText(msg)
        pct = self._parse_fraction_progress_msg(msg)
        if pct is not None:
            self._update_imp_progress(pct)
        else:
            self._update_imp_progress(0, indeterminate=True)

    def _imp_teardown_sync(self):
        if self.board is None:
            return
        try:
            self.board.stop_stream()
        except (BrainFlowError, UnicodeDecodeError):
            pass
        for ch_char in CHANNEL_IDS:
            safe_config_board(self.board, build_impedance_command(ch_char, False, False))
            time.sleep(0.04)

    def apply_hardware_settings(self):
        """把和 OpenBCI GUI 一致的通道硬件设置写入板卡(对全部 16 通道)。"""
        try:
            for ch_char in CHANNEL_IDS:
                self._safe_config_board(build_channel_command(ch_char))
                QtCore.QThread.msleep(60)   # 命令之间留出间隔, 确保板卡处理完
            return "  |  硬件设置已写入(SRB2 Off/SRB1 On, x24)"
        except (BrainFlowError, SerialBoardError) as e:
            return f"  |  硬件设置失败: {e}"

    def disconnect_board(self):
        if self._connect_worker_busy():
            self._connect_board_worker.wait(30000)
            self._connect_board_worker = None
            self._hide_connect_progress()
            self.connect_btn.setText("连接")
            self.connect_btn.setEnabled(True)
            self.port_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self.hw_checkbox.setEnabled(True)
            self.baud_combo.setEnabled(True)
            if self.board is not None:
                try:
                    self.board.release_session()
                except (BrainFlowError, SerialBoardError):
                    pass
                self.board = None
            self.is_connected = False
            self.is_debug_virtual = False
            self._set_app_status("状态: 未连接")
            return
        if self.erders_running:
            self.stop_erders(finished=False)
        if self.validate_running:
            self.stop_validate(finished=False)
        if self.acquire_running:
            self.stop_acquire(finished=False)
        if self.test_running:
            self.stop_test(finished=False)
        if self.wearing_exp_running:
            self.stop_wearing_exp()
        if self.focus2_running:
            self.stop_focus2()
        if self.imp_running or self._imp_worker_busy():
            self.stop_impedance(sync=True)
        if self.is_streaming:
            self.toggle_stream()
        self._auto_verify_pending = False
        if self._sample_rate_verify_worker is not None and self._sample_rate_verify_worker.isRunning():
            self._sample_rate_verify_worker.wait(5000)
            self._sample_rate_verify_worker = None
        try:
            if self.board is not None:
                self.board.release_session()
        except (BrainFlowError, SerialBoardError):
            pass
        self.board = None
        self.is_connected = False
        self.is_debug_virtual = False
        self.connect_btn.setText("连接")
        self.start_btn.setText("开始采集")
        self.start_btn.setEnabled(False)
        self.imp_btn.setEnabled(False)
        self.focus2_btn.setEnabled(False)
        self.focus2_calib_btn.setEnabled(False)
        self.wearing_exp_btn.setEnabled(False)
        self.acquire_btn.setEnabled(False)
        if hasattr(self, "test_acquire_btn"):
            self.test_acquire_btn.setEnabled(False)
        if hasattr(self, "erders_btn"):
            self.erders_btn.setEnabled(False)
            self.erders_status_label.setText("未连接")
        self.validate_btn.setEnabled(False)
        self.port_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.hw_checkbox.setEnabled(True)
        self.baud_combo.setEnabled(True)
        self.board_profile = BoardProfile.profile_for_baud_rate(self.connect_baud_rate)
        self.sampling_rate = self.board_profile.expected_sample_rate
        self._rebuild_rate_dependent_state()
        self._set_app_status("状态: 未连接")
        self._sync_hw_setup_btn_state()

    # ---------------- 采集 ----------------
    def toggle_stream(self):
        if not self.is_streaming:
            self.start_stream()
        else:
            self.stop_stream()

    def start_stream(self):
        if self.wearing_exp_running:
            self.stop_wearing_exp()
        if self.focus2_running:
            self.stop_focus2()
        if self.imp_running:        # 采集与阻抗检测互斥
            self.stop_impedance()
        try:
            self._reset_scope_buffers()
            self.board.start_stream()
            self.is_streaming = True
            self.start_btn.setText("停止采集")
            self._set_monitor_empty_hint_visible(False)
            self.connect_btn.setEnabled(False)
            self.imp_btn.setEnabled(False)
            self.focus2_btn.setEnabled(False)
            self.focus2_calib_btn.setEnabled(False)
            self.wearing_exp_btn.setEnabled(False)
            self._sync_hw_setup_btn_state()
            if not self.is_debug_virtual and not self.board_profile.confirmed:
                self._start_auto_sample_rate_verify()
            else:
                QtCore.QTimer.singleShot(400, self._begin_scope_plot_after_stream_head)
                self._set_app_status("状态: 采集中 · 起流预热…")
        except (BrainFlowError, SerialBoardError) as e:
            self._set_app_status(f"状态: 启动采集失败 - {e}")

    def _begin_scope_plot_after_stream_head(self):
        """起流后丢弃首批不稳定样本，再开始示波器刷新。"""
        if not self.is_streaming or self.board is None:
            return
        self._drain_board_buffer()
        self._reset_scope_buffers()
        self.timer.start(UPDATE_INTERVAL_MS)
        self._set_app_status("状态: 采集中 · 缓冲预热中…")

    def stop_stream(self):
        self.timer.stop()
        self._auto_verify_pending = False
        if self._sample_rate_verify_worker is not None and self._sample_rate_verify_worker.isRunning():
            self._sample_rate_verify_worker.wait(5000)
            self._sample_rate_verify_worker = None
        try:
            if self.board is not None:
                self.board.stop_stream()
        except (BrainFlowError, SerialBoardError):
            pass
        self.is_streaming = False
        self.start_btn.setText("开始采集")
        self._set_monitor_empty_hint_visible(True)
        self.connect_btn.setEnabled(True)
        if self.is_connected:
            self.imp_btn.setEnabled(True)
            self.focus2_btn.setEnabled(True)
            self.focus2_calib_btn.setEnabled(True)
            self.wearing_exp_btn.setEnabled(True)
        self._set_app_status("状态: 已停止采集")
        self._sync_hw_setup_btn_state()

    def update_plot(self):
        if self.board is None or not self.is_streaming or self._auto_verify_pending:
            return
        try:
            data = self.board.get_board_data()
        except (BrainFlowError, SerialBoardError):
            return
        if data.shape[1] == 0:
            return

        warmup_pct = None
        for scope in self.scopes:
            self._ensure_scope_buffer(scope)
            ch_row = self.eeg_channels[scope["combo"].currentIndex()]
            ch_data = data[ch_row]    # 单位已是 uV

            # 滚动写入原始缓冲（缓冲未满前不显示，避免「常数填充 + 正弦」阶跃被带通放大）
            if not scope["has_data"]:
                scope["has_data"] = True
                scope["fill_count"] = 0

            n = len(ch_data)
            if n >= self.raw_size:
                scope["raw_buffer"] = ch_data[-self.raw_size:].astype(np.float64)
                scope["fill_count"] = self.raw_size
            else:
                scope["raw_buffer"] = np.roll(scope["raw_buffer"], -n)
                scope["raw_buffer"][-n:] = ch_data
                scope["fill_count"] = min(self.raw_size, scope.get("fill_count", 0) + n)

            if scope.get("fill_count", 0) < self._min_scope_samples_for_display():
                need = self._min_scope_samples_for_display()
                warmup_pct = scope["fill_count"] / float(need)
                continue

            display = self._scope_filtered_display(scope["raw_buffer"])
            if display is None:
                warmup_pct = self._filter_warmup_progress(scope["raw_buffer"])
                continue

            ch_num = scope["combo"].currentIndex() + 1
            scope["time_curve"].setData(self.time_axis, display)
            scope["value_label"].setText(f"CH{ch_num}: {display[-1]:.2f} uV")
            self.update_fft(scope, display)

        if warmup_pct is not None and warmup_pct < 1.0:
            self._set_app_status(
                f"状态: 采集中 · 滤波预热 {warmup_pct * 100:.0f}%"
            )

    def _ensure_scope_buffer(self, scope):
        """采样率变更后示波器缓冲可能与 self.raw_size 不一致，需强制对齐。"""
        if scope["raw_buffer"].size == self.raw_size:
            return
        scope["raw_buffer"] = np.zeros(self.raw_size)
        scope["has_data"] = False
        scope["fill_count"] = 0

    def _filter_display_trim(self) -> int:
        """滤波后从左侧丢弃的样本数（吸收 filtfilt 边界振铃）。"""
        fs = self._filter_sampling_rate()
        return max(int(self.pad_size), min_filter_samples(fs, FILTER_ORDER))

    def _scope_filter_window_size(self) -> int:
        """参与滤波的尾部窗口 = 显示区 + 左右各一段 trim。"""
        trim = self._filter_display_trim()
        return int(self.buffer_size) + 2 * trim

    def _scope_raw_size(self) -> int:
        """滚动原始缓冲长度（须 ≥ 滤波窗口）。"""
        return self._scope_filter_window_size()

    def _min_scope_samples_for_display(self) -> int:
        return self._scope_filter_window_size()

    def _drain_board_buffer(self):
        """清空板卡 pending 队列（丢弃实测/起流阶段的陈旧样本）。"""
        if self.board is None:
            return
        try:
            for _ in range(64):
                data = self.board.get_board_data()
                if data.shape[1] == 0:
                    break
        except (BrainFlowError, SerialBoardError):
            pass

    def _reset_scope_buffers(self):
        if not hasattr(self, "scopes"):
            return
        for scope in self.scopes:
            scope["raw_buffer"] = np.zeros(self.raw_size)
            scope["has_data"] = False
            scope["fill_count"] = 0

    def _filter_sampling_rate(self) -> int:
        """BrainFlow/scipy 滤波器须使用 int 通道采样率。"""
        return int(round(float(self.sampling_rate)))

    def _min_filter_samples(self) -> int:
        fs = self._filter_sampling_rate()
        pad = int(getattr(self, "pad_size", 0) or 0)
        return max(min_filter_samples(fs, FILTER_ORDER), pad)

    def _filter_warmup_progress(self, raw_buffer: np.ndarray) -> float:
        """返回缓冲预热进度 0..1（供状态栏提示）。"""
        need = self._min_scope_samples_for_display()
        if need <= 0:
            return 1.0
        return min(1.0, float(raw_buffer.size) / float(need))

    def _start_auto_sample_rate_verify(self):
        """开始采集后自动实测通道采样率（约 2s）。"""
        if self.board is None or self.is_debug_virtual:
            self.timer.start(UPDATE_INTERVAL_MS)
            return
        if self._sample_rate_verify_worker is not None and self._sample_rate_verify_worker.isRunning():
            return
        profile = BoardProfile.profile_for_baud_rate(self.connect_baud_rate)
        self._auto_verify_pending = True
        self._set_app_status(
            f"状态: 正在实测通道采样率（期望 {profile.expected_sample_rate} Hz）…"
        )
        self._sample_rate_verify_worker = SampleRateVerifyWorker(
            self.board,
            profile,
            self.ch1_index,
            parent=self,
        )
        self._sample_rate_verify_worker.finished_ok.connect(self._on_auto_verify_ok)
        self._sample_rate_verify_worker.failed.connect(self._on_auto_verify_failed)
        self._sample_rate_verify_worker.start()

    def _on_auto_verify_ok(self, measured: float, msg: str):
        self._sample_rate_verify_worker = None
        self._auto_verify_pending = False
        profile = BoardProfile.profile_for_baud_rate(self.connect_baud_rate).with_measurement(measured)
        self._apply_board_profile(profile, reset_scope_buffers=False)
        # 丢弃实测期间积压的起流样本，再清空示波器，避免左端阶跃伪迹
        self._drain_board_buffer()
        self._reset_scope_buffers()
        if self.is_streaming:
            self.timer.start(UPDATE_INTERVAL_MS)
            self._set_app_status(
                f"状态: 采集中 · 通道 {self.sampling_rate} Hz（已确认）· 缓冲预热中…"
            )
        self._sync_hw_setup_btn_state()

    def _on_auto_verify_failed(self, err: str):
        self._sample_rate_verify_worker = None
        self._auto_verify_pending = False
        if self.is_streaming:
            self.stop_stream()
        QtWidgets.QMessageBox.warning(
            self,
            "采样率实测未通过",
            f"{err}\n\n请检查：\n"
            f"- 工具栏波特率（当前 {self.connect_baud_rate}）是否与板卡 SW4/适配器一致\n"
            "- 是否已在外部发送正确的 ~ 采样率命令（~4 或 ~6）",
        )
        self._set_app_status(f"状态: 采样率实测未通过 — {err}")
        self._sync_hw_setup_btn_state()

    def _scope_filtered_display(self, raw_buffer: np.ndarray):
        """对尾部窗口滤波并裁掉左侧振铃区，避免整段缓冲左边界伪迹进入显示。"""
        win = self._scope_filter_window_size()
        if raw_buffer.size < win:
            return None
        trim = self._filter_display_trim()
        segment = raw_buffer[-win:].astype(np.float64, copy=False)
        filtered = self.apply_filters(segment)
        if filtered.size <= trim:
            return None
        display = filtered[trim:]
        if display.size >= self.buffer_size:
            return display[-self.buffer_size:]
        return None

    def _current_fft_max_freq(self) -> float:
        opts = getattr(self, "_fft_maxfreq_options", FFT_MAXFREQ_OPTIONS)
        idx = self.fft_maxfreq_combo.currentIndex() if hasattr(self, "fft_maxfreq_combo") else len(opts) - 1
        if 0 <= idx < len(opts):
            return float(opts[idx])
        return float(opts[-1])

    def _on_global_filter_changed(self, *_args):
        if self.is_streaming:
            self._refresh_scope_plots_from_buffers()

    def _refresh_scope_plots_from_buffers(self):
        """不重读板卡，仅按当前全局滤波设置刷新示波器。"""
        if not hasattr(self, "scopes"):
            return
        for scope in self.scopes:
            self._ensure_scope_buffer(scope)
            if not scope.get("has_data"):
                continue
            display = self._scope_filtered_display(scope["raw_buffer"])
            if display is None:
                continue
            ch_num = scope["combo"].currentIndex() + 1
            scope["time_curve"].setData(self.time_axis, display)
            scope["value_label"].setText(f"CH{ch_num}: {display[-1]:.2f} uV")
            self.update_fft(scope, display)

    def update_fft(self, scope, filtered_data):
        """计算并显示某个示波器的 FFT 频谱（Hamming 窗 + 幅度补偿）。"""
        n_fft = int(getattr(self, "fft_size", FFT_SIZE_LEGACY))
        block = filtered_data[-n_fft:]
        if len(block) < n_fft:
            return
        block = block - np.mean(block)
        windowed = block * self.fft_window
        amp = np.abs(np.fft.rfft(windowed)) / n_fft
        amp[1:-1] *= 2
        cg = float(getattr(self, "fft_cg", 1.0))
        if cg > 0:
            amp /= cg
        amp = np.maximum(amp, FFT_MIN_AMP)
        scope["fft_curve"].setData(self.fft_freqs, amp)
        scope["fft_plot"].setXRange(0, self._current_fft_max_freq(), padding=0)

    def update_fft_xrange(self):
        max_freq = self._current_fft_max_freq()
        for scope in self.scopes:
            scope["fft_plot"].setXRange(0, max_freq, padding=0)

    def update_fft_logmode(self):
        log = self.fft_log_checkbox.isChecked()
        for scope in self.scopes:
            scope["fft_plot"].setLogMode(False, log)

    def apply_filters(self, data):
        """对一段数据应用当前选中的带通和陷波滤波, 返回滤波后的副本。"""
        return self.apply_filter_options(
            data, self.bandpass_combo.currentText(), self.notch_combo.currentText(),
        )

    def apply_filter_options(self, data, bandpass_text, notch_text):
        """按指定带通/陷波选项滤波, 供不同页面共用。"""
        band = BANDPASS_OPTIONS.get(bandpass_text)
        notch = NOTCH_OPTIONS.get(notch_text)
        return apply_bandpass_notch(
            data,
            self._filter_sampling_rate(),
            band,
            notch,
            order=FILTER_ORDER,
            notch_half_width=NOTCH_HALF_WIDTH,
        )

    def _recording_meta_filter_note(self):
        """记录落盘时已应用的顶部全局带通/陷波(写入 meta 供服务器参考)。"""
        return {
            "eeg_storage": "gui_filtered",
            "gui_applied_filter": {
                "bandpass": self.bandpass_combo.currentText(),
                "notch": self.notch_combo.currentText(),
            },
            **self._board_profile_meta(),
        }

    def _eeg_from_board_filtered(self, board_data):
        """从 get_board_data() 矩阵取多通道 µV 并应用顶部全局带通/陷波。"""
        if board_data.shape[1] == 0:
            return None
        eeg = board_data[self.eeg_channels, :].T
        return self._filter_eeg_multichannel(eeg)

    def _filter_eeg_multichannel(self, eeg_uv):
        """对 (n, C) 多通道数据应用顶部全局带通/陷波。"""
        eeg = np.asarray(eeg_uv, dtype=np.float64)
        if eeg.ndim != 2 or eeg.shape[0] == 0:
            return eeg
        bp = self.bandpass_combo.currentText()
        notch = self.notch_combo.currentText()
        out = np.empty_like(eeg)
        for c in range(eeg.shape[1]):
            out[:, c] = self.apply_filter_options(eeg[:, c], bp, notch)
        return out

    # ---------------- 阻抗检测 ----------------
    def toggle_impedance(self):
        if self._imp_worker_busy():
            return
        if not self.imp_running:
            self.start_impedance()
        else:
            self.stop_impedance()

    def start_impedance(self):
        if not self.is_connected:
            self._set_app_status("状态: 请先连接板卡")
            return
        if self._imp_worker_busy():
            return
        if self.wearing_exp_running:
            self.stop_wearing_exp()
        if self.focus2_running:
            self.stop_focus2()
        if self.is_streaming:
            self.stop_stream()

        self._imp_p_on = bool(self.imp_end_combo.currentData())
        self.imp_btn.setEnabled(False)
        self.imp_btn.setText("正在启动...")
        self.imp_summary_label.setText("正在配置硬件...")
        self._update_imp_progress(0, indeterminate=True)
        self._set_app_status("状态: 正在启动阻抗检测...")

        self._imp_board_worker = ImpedanceBoardWorker(
            self.board, "start", self._imp_p_on, self,
        )
        self._imp_board_worker.progress.connect(self._on_imp_board_progress)
        self._imp_board_worker.finished_ok.connect(self._on_imp_start_ok)
        self._imp_board_worker.failed.connect(self._on_imp_start_failed)
        self._imp_board_worker.start()

    def _on_imp_start_ok(self):
        self._imp_board_worker = None
        self.imp_buffer = np.empty((len(self.eeg_channels), 0))
        self.imp_running = True
        self.imp_btn.setText("停止检测")
        self.imp_btn.setEnabled(True)
        self._reset_imp_channels()
        self.start_btn.setEnabled(False)
        self.focus2_btn.setEnabled(False)
        self.focus2_calib_btn.setEnabled(False)
        self.wearing_exp_btn.setEnabled(False)
        self.connect_btn.setEnabled(False)
        self.imp_end_combo.setEnabled(False)
        self.imp_timer.start(IMP_UPDATE_MS)
        QtCore.QTimer.singleShot(0, self.update_impedance)
        p_label = "P" if self._imp_p_on else "N"
        self._set_app_status(f"状态: 16 通道阻抗检测中... ({p_label} 端)")

    def _on_imp_start_failed(self, err):
        self._imp_board_worker = None
        self.imp_running = False
        self.imp_btn.setEnabled(True)
        self.imp_btn.setText("开始阻抗检测")
        self.imp_summary_label.setText("待检测 · 0/16 通道合格")
        self._update_imp_progress(0)
        self._set_app_status(f"状态: 启动阻抗检测失败 - {err}")

    def _restore_impedance_ui(self):
        """阻抗检测结束后的界面恢复(无论板卡命令是否成功)。"""
        self.imp_btn.setText("开始阻抗检测")
        self.imp_end_combo.setEnabled(True)
        if self.is_connected:
            self.start_btn.setEnabled(True)
            self.focus2_btn.setEnabled(True)
            self.focus2_calib_btn.setEnabled(True)
            self.wearing_exp_btn.setEnabled(True)
            self.connect_btn.setEnabled(True)

    def stop_impedance(self, sync=False):
        self.imp_timer.stop()
        self.imp_running = False

        if self._imp_worker_busy():
            if sync:
                self._imp_board_worker.wait(30000)
                self._imp_board_worker = None
            else:
                return

        if self.board is None:
            self._restore_impedance_ui()
            self._set_app_status("状态: 已停止阻抗检测")
            return

        if sync:
            self._imp_teardown_sync()
            self._restore_impedance_ui()
            self._set_app_status("状态: 已停止阻抗检测")
            return

        self.imp_btn.setEnabled(False)
        self.imp_btn.setText("正在停止...")
        self._set_app_status("状态: 正在停止阻抗检测...")
        self._update_imp_progress(0, indeterminate=True)
        self._imp_board_worker = ImpedanceBoardWorker(self.board, "stop", parent=self)
        self._imp_board_worker.progress.connect(self._on_imp_board_progress)
        self._imp_board_worker.finished_ok.connect(self._on_imp_stop_ok)
        self._imp_board_worker.failed.connect(self._on_imp_stop_failed)
        self._imp_board_worker.start()

    def _on_imp_stop_ok(self):
        self._imp_board_worker = None
        self._restore_impedance_ui()
        self.imp_btn.setEnabled(True)
        self._update_imp_progress(0)
        self._set_app_status("状态: 已停止阻抗检测")

    def _on_imp_stop_failed(self, _err):
        self._imp_board_worker = None
        self._restore_impedance_ui()
        self.imp_btn.setEnabled(True)
        self._update_imp_progress(0)
        self._set_app_status("状态: 已停止阻抗检测")

    def update_impedance(self):
        if self.board is None or not self.imp_running:
            return
        try:
            data = self.board.get_board_data()
            if data.shape[1] > 0:
                new = data[self.eeg_channels, :]        # (16, n)
                self.imp_buffer = np.concatenate([self.imp_buffer, new], axis=1)
                if self.imp_buffer.shape[1] > self.imp_window_size:
                    self.imp_buffer = self.imp_buffer[:, -self.imp_window_size:]

            if self.imp_buffer.shape[1] < self.sampling_rate:   # 至少累积约 1 秒再计算
                for ch in range(len(self.eeg_channels)):
                    self._set_imp_channel_ui(ch, "measuring", "···", "测量中")
                self.imp_summary_label.setText("测量中 · 数据累积...")
                buf_pct = min(
                    100.0, self.imp_buffer.shape[1] / max(1, self.sampling_rate) * 100.0,
                )
                self._update_imp_progress(buf_pct)
                return

            ok_count = 0
            for ch in range(len(self.eeg_channels)):
                seg = self.imp_buffer[ch]
                if np.max(np.abs(seg)) > RAIL_THRESHOLD_UV:
                    self._set_imp_channel_ui(ch, "open", "∞", "未接/超量程")
                    continue
                rms_uv = float(np.std(seg - np.mean(seg)))
                z_ohm = (np.sqrt(2.0) * rms_uv * 1.0e-6) / IMP_DRIVE_AMPS - IMP_SERIES_RESISTOR
                z_kohm = max(0.0, z_ohm) / 1000.0
                desc, _color = self._impedance_quality(z_kohm)
                tier = self._imp_tier_from_z(z_kohm)
                if tier in IMP_QUALITY_OK_TIERS:
                    ok_count += 1
                self._set_imp_channel_ui(ch, tier, f"{z_kohm:.1f}", desc)

            n = len(self.eeg_channels)
            self.imp_summary_label.setText(f"检测中 · {ok_count}/{n} 通道合格 (≤{IMP_PASS_KOHM} kΩ)")
            self._update_imp_progress(ok_count / n * 100.0, success=True)
        except Exception as e:  # noqa: BLE001
            self.imp_summary_label.setText(f"读取异常: {e}")
            self._update_imp_progress(0, indeterminate=True)

    @staticmethod
    def _impedance_quality(z_kohm):
        """根据阻抗值给出简单的信号质量提示和颜色(阈值参考常见经验值)。"""
        if z_kohm <= 10:
            return "极好", "#1a7f37"
        elif z_kohm <= 50:
            return "良好", "#2da44e"
        elif z_kohm <= 100:
            return "可接受", "#bf8700"
        elif z_kohm <= 200:
            return "偏高", "#d1242f"
        else:
            return "差/未接好", "#a40e26"

    # ---------------- 专注指标 ----------------
    def toggle_focus2(self):
        if not self.focus2_running:
            self.start_focus2()
        else:
            self.stop_focus2()

    def start_focus2(self):
        if not self.is_connected:
            self._set_app_status("状态: 请先连接板卡")
            return
        if self.wearing_exp_running:
            self.stop_wearing_exp()
        if self.is_streaming:
            self.stop_stream()
        if self.imp_running:
            self.stop_impedance()
        try:
            self.focus2_history.clear()
            for hist in self.focus2_cog_histories.values():
                hist.clear()
            self.focus2_value = 0.0
            self.focus2_run = False
            self.board.start_stream()
            self.focus2_running = True
            self.focus2_btn.setText("停止专注力检测")
            self.start_btn.setEnabled(False)
            self.imp_btn.setEnabled(False)
            self.wearing_exp_btn.setEnabled(False)
            self.connect_btn.setEnabled(False)
            self.focus2_calib_btn.setEnabled(True)
            self.focus2_timer.start(FOCUS_UPDATE_MS)
            self._refresh_focus2_display(measuring=True)
            self._set_app_status("状态: 专注指标检测中 (CH1+CH2)...")
        except (BrainFlowError, SerialBoardError) as e:
            self.focus2_running = False
            self._set_app_status(f"状态: 启动专注力检测失败 - {e}")

    def stop_focus2(self):
        self.focus2_timer.stop()
        self.focus2_calibrating = False
        try:
            if self.board is not None and self.focus2_running:
                self.board.stop_stream()
        except (BrainFlowError, SerialBoardError):
            pass
        self.focus2_running = False
        self.focus2_btn.setText("开始专注力检测")
        self.focus2_calib_btn.setEnabled(self.is_connected)
        if self.is_connected:
            self.start_btn.setEnabled(True)
            self.imp_btn.setEnabled(True)
            self.wearing_exp_btn.setEnabled(True)
            self.connect_btn.setEnabled(True)
        self._apply_focus2_status_badge("stopped", "已停止")
        if not self.is_streaming and not self.imp_running:
            self._set_app_status("状态: 已停止专注力检测")

    def start_focus2_calibration(self):
        if not self.is_connected:
            self._set_app_status("状态: 请先连接板卡")
            return
        if not self.focus2_running:
            self.start_focus2()
        self.focus2_calibrating = True
        self.focus2_calib_until = time.time() + FOCUS2_CALIBRATION_SECONDS
        self.focus2_calib_focus_vals = []
        self.focus2_calib_ch1_sum = 0.0
        self.focus2_calib_ch2_sum = 0.0
        self.focus2_calib_n_samples = 0
        self.focus2_calib_btn.setEnabled(False)
        self.focus2_calib_label.setText(
            f"校准中: 请闭眼放松 {FOCUS2_CALIBRATION_SECONDS:.0f} 秒..."
        )
        self.focus2_calib_label.setStyleSheet("color: #bf8700;")

    def _finish_focus2_calibration(self):
        self.focus2_calibrating = False
        self.focus2_calib_btn.setEnabled(True)
        if not self.focus2_calib_focus_vals:
            self.focus2_calib_label.setText("个体化阈值: 校准失败 (数据不足)")
            self.focus2_calib_label.setStyleSheet("color: #d1242f;")
            return
        arr = np.asarray(self.focus2_calib_focus_vals, dtype=np.float64)
        self.focus2_baseline_mean = float(np.mean(arr))
        self.focus2_baseline_std = float(np.std(arr))
        self.focus2_personal_threshold = (
            self.focus2_baseline_mean
            + FOCUS2_PERSONAL_THRESHOLD_SIGMA * self.focus2_baseline_std
        )
        if self.focus2_calib_n_samples > 0:
            self.focus2_rest_mean_ch1 = (
                self.focus2_calib_ch1_sum / self.focus2_calib_n_samples
            )
            self.focus2_rest_mean_ch2 = (
                self.focus2_calib_ch2_sum / self.focus2_calib_n_samples
            )
        self.focus2_calib_label.setText(
            f"个体化阈值: {self.focus2_personal_threshold:.1f}  "
            f"(静息 μ={self.focus2_baseline_mean:.1f}, σ={self.focus2_baseline_std:.1f})"
        )
        self.focus2_calib_label.setStyleSheet("color: #1a7f37;")
        if self.focus2_personal_cb.isChecked():
            self.focus2_threshold_line.setPos(self.focus2_personal_threshold)

    def _on_focus2_threshold_changed(self, value):
        if not self.focus2_personal_cb.isChecked():
            self.focus2_threshold_line.setPos(value)

    def _on_focus2_personal_threshold_toggle(self):
        if self.focus2_personal_cb.isChecked() and self.focus2_baseline_mean is not None:
            self.focus2_threshold_line.setPos(self.focus2_personal_threshold)
            self.focus2_threshold_spin.setEnabled(False)
        else:
            self.focus2_threshold_line.setPos(self.focus2_threshold_spin.value())
            self.focus2_threshold_spin.setEnabled(True)

    def _get_focus2_threshold(self):
        if self.focus2_personal_cb.isChecked() and self.focus2_baseline_mean is not None:
            return self.focus2_personal_threshold
        return self.focus2_threshold_spin.value()

    def _preprocess_channel_enhanced(self, channel_data, rest_mean=None):
        """专注指数预处理 + 伪迹剔除 + 可选静息基线扣除。"""
        data = remove_eye_blink_artifacts(channel_data, FOCUS2_ARTIFACT_THRESHOLD_UV)
        if rest_mean is not None:
            data = data - rest_mean
        n = len(data)
        data = np.ascontiguousarray(data, dtype=np.float64)
        data -= np.sum(data) / n
        DataFilter.detrend(data, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(
            data, self.sampling_rate, FOCUS2_BANDPASS[0], FOCUS2_BANDPASS[1],
            FOCUS_FILTER_ORDER, FilterTypes.BUTTERWORTH.value, FOCUS_RIPPLE,
        )
        for lo, hi in FOCUS_NOTCH_BANDS:
            DataFilter.perform_bandstop(
                data, self.sampling_rate, lo, hi,
                FOCUS_FILTER_ORDER, FilterTypes.BUTTERWORTH.value, FOCUS_RIPPLE,
            )
        return data

    def _compute_focus2_metrics(self, ch1_raw, ch2_raw):
        """增强版: 伪迹剔除/静息基线 + 专注指数 + 对比指标。"""
        q1 = estimate_channel_quality(ch1_raw)
        q2 = estimate_channel_quality(ch2_raw)
        signal_ok = q1["ok"] and q2["ok"]

        avg_bands = [0.0] * len(FOCUS_BANDS)
        focus_value1, focus_value2 = 0.0, 0.0

        for channel_idx, raw, rest_mean in (
            (0, ch1_raw, self.focus2_rest_mean_ch1),
            (1, ch2_raw, self.focus2_rest_mean_ch2),
        ):
            processed = self._preprocess_channel_enhanced(raw, rest_mean)
            if len(processed) <= self.focus_psd_size:
                return None

            psd_data = DataFilter.get_psd_welch(
                processed, self.focus_psd_size, self.focus_psd_size // 2,
                self.sampling_rate, WindowOperations.BLACKMAN_HARRIS.value,
            )
            for i, (_, lo, hi, _) in enumerate(FOCUS_BANDS):
                avg_bands[i] += DataFilter.get_band_power(psd_data, lo, hi)

            if avg_bands[2] != 0:
                ratio = (avg_bands[3] / (avg_bands[2] * 4.0)) * 100.0
                if channel_idx == 0:
                    focus_value1 = ratio
                elif channel_idx == 1:
                    focus_value2 = ratio

        focus_bt = (focus_value1 + focus_value2) / 2.0
        band_powers = [b / 2.0 for b in avg_bands]
        cognitive = compute_focus1_cognitive_indices(band_powers)
        return {
            "focus_bt": focus_bt,
            "focus_bt1": focus_value1,
            "focus_bt2": focus_value2,
            "cognitive": cognitive,
            "band_powers": band_powers,
            "signal_ok": signal_ok,
            "q1": q1,
            "q2": q2,
        }

    def _focus2_trend_text(self, history):
        if len(history) < 2:
            return "---"
        arr = np.asarray(history, dtype=np.float64)
        return "上升" if arr[-1] > arr[0] else "下降"

    def _set_focus2_curve_data(self, curve, history):
        """把历史数据右对齐绘制到固定 5s 滑窗。"""
        n = len(history)
        if n == 0:
            return
        pts = self.focus_plot_points
        if n < pts:
            y = np.full(pts, np.nan)
            y[-n:] = history
        else:
            y = np.asarray(history, dtype=np.float64)
        curve.setData(self.focus_time_axis, y)

    def update_focus2(self):
        if self.board is None or not self.focus2_running:
            return

        data = self.board.get_current_board_data(self.focus_window_size)
        if data.shape[1] < self.focus_window_size // 2:
            self._refresh_focus2_display(measuring=True)
            return

        ch1_raw = data[self.ch1_index]
        ch2_raw = data[self.ch2_index]
        metrics = self._compute_focus2_metrics(ch1_raw, ch2_raw)
        if metrics is None:
            self._refresh_focus2_display(measuring=True)
            return

        focus_bt = metrics["focus_bt"]
        cog = metrics["cognitive"]

        if self.focus2_calibrating:
            self.focus2_calib_focus_vals.append(focus_bt)
            self.focus2_calib_ch1_sum += float(np.mean(ch1_raw))
            self.focus2_calib_ch2_sum += float(np.mean(ch2_raw))
            self.focus2_calib_n_samples += 1
            remain = self.focus2_calib_until - time.time()
            if remain > 0:
                self.focus2_calib_label.setText(
                    f"校准中: 请闭眼放松... 剩余 {remain:.1f}s"
                )
            else:
                self._finish_focus2_calibration()

        threshold = self._get_focus2_threshold()
        self.focus2_value = focus_bt
        self.focus2_signal_ok = metrics["signal_ok"]
        self.focus2_history.append(focus_bt)
        for key, _, _, _ in FOCUS2_COG_SPECS:
            self.focus2_cog_histories[key].append(cog[key] * 100.0)

        q1, q2 = metrics["q1"], metrics["q2"]
        if not self.focus2_signal_ok:
            q_text = "信号质量: 差 (结果仅供参考)"
            q_color = "#d1242f"
        else:
            q_text = "信号质量: 良好"
            q_color = "#1a7f37"
        self.focus2_quality_label.setText(
            f"{q_text}  |  CH1: q={q1['quality']:.2f}  CH2: q={q2['quality']:.2f}"
        )
        self.focus2_quality_label.setStyleSheet(f"color: {q_color};")

        self.focus2_bt_label.setText(f"{focus_bt:.1f}")
        self.focus2_bt_sub.setText(
            f"CH1: {metrics['focus_bt1']:.1f}  CH2: {metrics['focus_bt2']:.1f}"
        )
        for key, _, _, _ in FOCUS2_COG_SPECS:
            self.focus2_cog_labels[key].setText(f"{cog[key] * 100:.1f}%")

        can_judge = self.focus2_signal_ok and not self.focus2_calibrating
        self.focus2_run = can_judge and focus_bt > threshold
        if not can_judge:
            self._apply_focus2_status_badge("calibrating", "信号差 / 校准中")
        elif self.focus2_run:
            self._apply_focus2_status_badge("ok", "注意力集中")
        else:
            self._apply_focus2_status_badge("bad", "注意力不集中")

        for lbl, (name, _, _, color), power in zip(
            self.focus2_band_labels, FOCUS_BANDS, metrics["band_powers"],
        ):
            lbl.setText(f"{name}: {power:.3f}")
            lbl.setStyleSheet(f"QLabel#focus2BandChip {{ color: rgb{color}; }}")

        hist_bt = list(self.focus2_history)
        if len(hist_bt) >= 2:
            arr = np.asarray(hist_bt, dtype=np.float64)
            focus_std = float(np.std(arr))
            focus_range = float(np.max(arr) - np.min(arr))
        elif len(hist_bt) == 1:
            focus_std = focus_range = 0.0
        else:
            focus_std = focus_range = float("nan")
        self.focus2_var_label.setText(
            f"σ:{focus_std:.2f}  Δ:{focus_range:.2f}  "
            f"趋势:{self._focus2_trend_text(hist_bt)} ({FOCUS_PLOT_SECONDS:.0f}s)"
        )

        self._set_focus2_curve_data(self.focus2_bt_curve, hist_bt)
        for key, _, _, _ in FOCUS2_COG_SPECS:
            self._set_focus2_curve_data(
                self.focus2_cog_curves[key], list(self.focus2_cog_histories[key]),
            )
        self.focus2_threshold_line.setPos(threshold)
        self.focus2_plot.setXRange(-FOCUS_PLOT_SECONDS, 0, padding=0)
        self.focus2_plot.setYRange(0, 100, padding=0)

    def _refresh_focus2_display(self, measuring=False):
        if measuring:
            self.focus2_bt_label.setText("···")
            self.focus2_bt_sub.setText("CH1:---  CH2:---")
            for key, _, _, _ in FOCUS2_COG_SPECS:
                self.focus2_cog_labels[key].setText("---")
            self._apply_focus2_status_badge("measuring", "数据累积中")
            self.focus2_var_label.setText(
                f"σ:---  Δ:---  趋势:--- ({FOCUS_PLOT_SECONDS:.0f}s)"
            )
            for lbl, (name, _, _, color) in zip(self.focus2_band_labels, FOCUS_BANDS):
                lbl.setText(f"{name}: ---")
                lbl.setStyleSheet(f"QLabel#focus2BandChip {{ color: rgb{color}; }}")

    # ---------------- 实时佩戴检测页 ----------------
    def toggle_wearing_exp(self):
        if not self.wearing_exp_running:
            self.start_wearing_exp()
        else:
            self.stop_wearing_exp()

    def start_wearing_exp(self):
        if not self.is_connected:
            self._set_app_status("状态: 请先连接板卡")
            return
        if self.focus2_running:
            self.stop_focus2()
        if self.imp_running:
            self.stop_impedance()
        if self.is_streaming:
            self.stop_stream()
        try:
            self.wearing_exp_buffer = np.empty((len(self.eeg_channels), 0))
            self.wearing_exp_is_worn = False
            self.wearing_exp_score = 0
            self.wearing_exp_last_analysis_at = 0.0
            self.wearing_exp_score_ema = np.full(16, np.nan)
            self.wearing_exp_channel_ok = [False] * 16
            self.board.start_stream()
            self.wearing_exp_running = True
            self.wearing_exp_btn.setText("停止佩戴检测")
            self.start_btn.setEnabled(False)
            self.imp_btn.setEnabled(False)
            self.focus2_btn.setEnabled(False)
            self.focus2_calib_btn.setEnabled(False)
            self.connect_btn.setEnabled(False)
            self.wearing_exp_timer.start(WEARING_UPDATE_MS)
            self._refresh_wearing_exp_display(measuring=True)
            self._set_app_status("状态: 实时佩戴检测中 (16 通道)...")
        except (BrainFlowError, SerialBoardError) as e:
            self.wearing_exp_running = False
            self._set_app_status(f"状态: 启动实时佩戴检测失败 - {e}")

    def stop_wearing_exp(self):
        self.wearing_exp_timer.stop()
        try:
            if self.board is not None and self.wearing_exp_running:
                self.board.stop_stream()
        except (BrainFlowError, SerialBoardError):
            pass
        self.wearing_exp_running = False
        self.wearing_exp_btn.setText("开始佩戴检测")
        if self.is_connected:
            self.start_btn.setEnabled(True)
            self.imp_btn.setEnabled(True)
            self.focus2_btn.setEnabled(True)
            self.focus2_calib_btn.setEnabled(True)
            self.connect_btn.setEnabled(True)
        self.wearing_exp_summary_label.setText("通道质量: 已停止")
        self.wearing_exp_summary_label.setStyleSheet("color: #666; background: transparent;")
        if not self.is_streaming and not self.imp_running and not self.focus2_running:
            self._set_app_status("状态: 已停止实时佩戴检测")

    def update_wearing_exp(self):
        if self.board is None or not self.wearing_exp_running:
            return

        data = self.board.get_board_data()
        if data.shape[1] > 0:
            new = data[self.eeg_channels, :]
            self.wearing_exp_buffer = np.concatenate([self.wearing_exp_buffer, new], axis=1)
            if self.wearing_exp_buffer.shape[1] > self.wearing_window_size:
                self.wearing_exp_buffer = self.wearing_exp_buffer[:, -self.wearing_window_size:]

        if self.wearing_exp_buffer.shape[1] == 0:
            self._refresh_wearing_exp_display(measuring=True)
            return

        self._update_wearing_exp_plot()
        if self.wearing_exp_buffer.shape[1] < self._wearing_analysis_samples():
            self._refresh_wearing_exp_display(measuring=True)
            return

        now = time.time()
        if now - self.wearing_exp_last_analysis_at < WEARING_ANALYSIS_INTERVAL_SECONDS:
            return
        self.wearing_exp_last_analysis_at = now

        is_worn, ok_channels, channels_info, avg_score = check_wearing_sqi(
            self.wearing_exp_buffer, self.sampling_rate, WEARING_SQI_WINDOW_SECONDS,
        )
        stable_ok_channels = 0
        stable_scores = []
        for idx, info in enumerate(channels_info):
            raw_score = float(info["score"])
            old_score = self.wearing_exp_score_ema[idx]
            if np.isnan(old_score):
                smooth_score = raw_score
            else:
                smooth_score = (
                    WEARING_SQI_EMA_ALPHA * raw_score
                    + (1.0 - WEARING_SQI_EMA_ALPHA) * old_score
                )
            self.wearing_exp_score_ema[idx] = smooth_score

            if self.wearing_exp_channel_ok[idx]:
                channel_ok = smooth_score >= WEARING_SQI_OK_OFF_THRESHOLD
            else:
                channel_ok = smooth_score >= WEARING_SQI_OK_ON_THRESHOLD
            self.wearing_exp_channel_ok[idx] = channel_ok
            info["stable_score"] = round(float(smooth_score), 1)
            info["stable_worn"] = channel_ok
            stable_scores.append(smooth_score)
            stable_ok_channels += 1 if channel_ok else 0

        self.wearing_exp_is_worn = is_worn
        self.wearing_exp_score = stable_ok_channels
        stable_avg_score = float(np.mean(stable_scores)) if stable_scores else avg_score

        self.wearing_exp_summary_label.setText("通道质量: 单通道判定中")
        self.wearing_exp_summary_label.setStyleSheet("color: #444;")
        self.wearing_exp_score_label.setText(
            f"OK通道: {stable_ok_channels} / 16    平均SQI: {stable_avg_score:.1f}"
        )

        self._update_wearing_exp_channels_ui(channels_info)

    def _update_wearing_exp_plot(self):
        n = min(self.wearing_exp_buffer.shape[1], self.wearing_window_size)
        if n <= 0:
            return
        display_n = min(n, WEARING_DISPLAY_POINTS)
        src_idx = np.linspace(0, n - 1, display_n).astype(int)
        x_full = self.wearing_time_axis[-n:]
        x = x_full[src_idx]
        for ch, curve in enumerate(self.wearing_exp_curves):
            if ch >= self.wearing_exp_buffer.shape[0]:
                curve.clear()
                continue
            raw = self.wearing_exp_buffer[ch, -n:]
            filtered = self.apply_filters(raw)
            curve.setData(x, filtered[src_idx])
            self.wearing_exp_plots[ch].setXRange(-WEARING_WINDOW_SECONDS, 0, padding=0)

    def _refresh_wearing_exp_display(self, measuring=False):
        if measuring:
            self.wearing_exp_summary_label.setText("通道质量: 数据累积中...")
            self.wearing_exp_summary_label.setStyleSheet("color: #bf8700; background: transparent;")
            self.wearing_exp_score_label.setText("OK通道: --- / 16")
            self._update_wearing_exp_channels_ui(measuring=True)

    # ---------------- 关闭窗口 ----------------
    def closeEvent(self, event):
        if self._session_busy():
            reply = QtWidgets.QMessageBox.question(
                self,
                "确认退出",
                "仍有采集或检测在进行。确定退出并断开板卡吗？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        self._save_ui_settings()
        try:
            if getattr(self, "_subject_cue_window", None) is not None:
                self._subject_cue_window.close()
            self.disconnect_board()
        except Exception:
            logging.getLogger("bci_desk").exception("关闭时断开板卡失败")
        event.accept()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=f"{APP_NAME} — {APP_TAGLINE}")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式（显示虚拟端口等开发入口）",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global _DEBUG_MODE
    args = _parse_args(argv)
    env_debug = os.environ.get("BCI_DESK_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    _DEBUG_MODE = bool(args.debug or env_debug)

    project_root = project_root_from(__file__)
    log_path = setup_app_logging(project_root)
    install_exception_hooks(log_path)
    logging.getLogger("bci_desk").info(
        "%s v%s starting (debug=%s, log=%s)", APP_NAME, __version__, _DEBUG_MODE, log_path
    )

    BoardShim.disable_board_logger()   # 不输出 brainflow 调试日志, 需要可改为 enable
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(APP_ORG)
    app.setWindowIcon(load_app_icon())
    # 统一基础风格 + 全局字体 + 医疗软件主题(仅外观, 不影响功能逻辑)
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(MEDICAL_STYLESHEET + spinbox_arrow_stylesheet())
    win = EEGMonitor()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
