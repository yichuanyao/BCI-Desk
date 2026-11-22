"""采集范式注册表：手指 ME、手形 ME 等多范式统一配置。

GUI 与落盘 meta 通过 ``ParadigmSpec.id`` 引用，避免在界面层写死类别与路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from bci_desk.acquisition.paradigm import MISessionTiming
from bci_desk.config import (
    CONDITION_TO_LABEL,
    CONDITION_TO_MARKER,
    LABEL_NAMES_CN,
    conditions_for_class_mode,
    finger_names_cn,
    labels_for_class_mode,
)

# 手形 ME 独立 marker（与手指 1~4 区分）
HAND_MARKER_FIST = 1
HAND_MARKER_RELAX = 2
HAND_MARKER_PALM = 3

HAND_CONDITION_TO_LABEL = {
    "fist": "FIST",
    "relax": "RELAX",
    "palm": "PALM",
}

HAND_CONDITION_TO_MARKER = {
    "fist": HAND_MARKER_FIST,
    "relax": HAND_MARKER_RELAX,
    "palm": HAND_MARKER_PALM,
}

HAND_LABEL_NAMES_CN = {
    "FIST": "握拳",
    "RELAX": "放松",
    "PALM": "张掌",
    "fist": "握拳",
    "relax": "放松",
    "palm": "张掌",
}

DEFAULT_TRAIN_TIMING = MISessionTiming()
# 手形二分类：握拳 4s、放松 3s，试次间无「+」休息（文献对齐 Wu2020 / 康复 ME）
HAND_2CLASS_TRAIN_TIMING = MISessionTiming(
    label_s=4.0,
    rest_s=0.0,
    condition_label_s={"fist": 4.0, "relax": 3.0},
    trial_order="alternate",
)
DEFAULT_TEST_SEGMENTS = 1
DEFAULT_TEST_TRIALS = 20


@dataclass(frozen=True)
class ParadigmSpec:
    """一种可采集范式的完整描述。"""

    id: str
    name_cn: str
    family: str  # "finger" | "hand"
    conditions: Tuple[str, ...]
    labels: Tuple[str, ...]
    condition_to_label: Dict[str, str]
    condition_to_marker: Dict[str, int]
    label_names_cn: Dict[str, str]
    cue_hint: str
    cue_image_subdir: str
    train_timing: MISessionTiming
    test_segments: int = DEFAULT_TEST_SEGMENTS
    test_trials: int = DEFAULT_TEST_TRIALS
    session_prefix: str = "FINGER2"
    record_subdir: str = ""

    def __post_init__(self):
        if not self.record_subdir:
            object.__setattr__(self, "record_subdir", self.id)

    def conditions_cn(self) -> str:
        parts = []
        for c in self.conditions:
            label = self.condition_to_label.get(c, c)
            parts.append(self.label_names_cn.get(label, self.label_names_cn.get(c, c)))
        return " / ".join(parts)

    def marker_summary(self) -> str:
        markers = sorted(set(self.condition_to_marker[c] for c in self.conditions))
        return " / ".join(str(m) for m in markers)

    def event_types_summary(self) -> str:
        return " / ".join(self.labels)

    def _timing_summary(self, timing: MISessionTiming) -> str:
        trials = timing.n_segments * timing.n_trials_per_segment
        avg_trial = timing.avg_trial_duration_s(list(self.conditions))
        total_min = trials * avg_trial / 60.0
        if timing.condition_label_s and self.conditions:
            parts = []
            for c in self.conditions:
                cn = self.label_names_cn.get(c, c)
                parts.append(f"{cn} {timing.label_duration_for(c):.0f}s")
            rhythm = " / ".join(parts)
            if timing.rest_s > 0:
                rhythm += f" + 休息 {timing.rest_s:.0f}s"
            else:
                rhythm += "，试次间无休息"
            if timing.trial_order == "alternate":
                rhythm += " · 握拳/放松交替"
        else:
            rhythm = timing.rhythm_summary(list(self.conditions))
        return (
            f"- **类别**: {self.conditions_cn()} (marker **{self.marker_summary()}**)\n"
            f"- **节奏**: {rhythm}; "
            f"**{timing.n_segments} Segment × {timing.n_trials_per_segment} 试次** "
            f"= {trials} 次 (约 **{total_min:.1f} 分钟**)\n"
            f"- **段间**: 按空格继续\n"
            f"- **落盘标签**: BREAK / {self.event_types_summary()}"
            + (" / 0" if timing.rest_s > 0 else "")
        )

    def train_info_tip(self, save_path: str = "") -> str:
        text = f"- **范式**: {self.name_cn}\n{self._timing_summary(self.train_timing)}"
        if save_path:
            text += f"\n- **保存路径**: {save_path}"
        return text

    def test_info_tip(self, timing: MISessionTiming, save_dir: str = "") -> str:
        text = (
            "- **说明**: 缩短版测试集落盘（供服务器 evaluate，非本机实时推理）\n"
            f"- **流程**: 与上方所选范式相同 (缩短版测试集)\n"
            f"{self._timing_summary(timing)}\n"
            f"- **评估**: 服务器 **evaluate.py**"
        )
        if save_dir:
            text += f"\n- **测试目录**: {save_dir}"
        return text


def _finger_spec(
    spec_id: str,
    name_cn: str,
    n_classes: int,
    session_prefix: str,
) -> ParadigmSpec:
    conditions = tuple(conditions_for_class_mode(n_classes))
    labels = tuple(labels_for_class_mode(n_classes))
    return ParadigmSpec(
        id=spec_id,
        name_cn=name_cn,
        family="finger",
        conditions=conditions,
        labels=labels,
        condition_to_label=dict(CONDITION_TO_LABEL),
        condition_to_marker=dict(CONDITION_TO_MARKER),
        label_names_cn=dict(LABEL_NAMES_CN),
        cue_hint="{name} · 请做该指屈伸 1 次",
        cue_image_subdir="finger",
        train_timing=DEFAULT_TRAIN_TIMING,
        session_prefix=session_prefix,
    )


def _hand_spec(
    spec_id: str,
    name_cn: str,
    conditions: Tuple[str, ...],
    session_prefix: str,
    *,
    train_timing: Optional[MISessionTiming] = None,
    cue_hint: str = "{name} · 请按提示保持至结束",
) -> ParadigmSpec:
    labels = tuple(HAND_CONDITION_TO_LABEL[c] for c in conditions)
    return ParadigmSpec(
        id=spec_id,
        name_cn=name_cn,
        family="hand",
        conditions=conditions,
        labels=labels,
        condition_to_label=dict(HAND_CONDITION_TO_LABEL),
        condition_to_marker=dict(HAND_CONDITION_TO_MARKER),
        label_names_cn=dict(HAND_LABEL_NAMES_CN),
        cue_hint=cue_hint,
        cue_image_subdir="hand",
        train_timing=train_timing or DEFAULT_TRAIN_TIMING,
        session_prefix=session_prefix,
    )


PARADIGM_REGISTRY: Dict[str, ParadigmSpec] = {
    "finger_2": _finger_spec("finger_2", "两指运动执行 (拇指/小指)", 2, "FINGER2"),
    "finger_3": _finger_spec("finger_3", "三指运动执行 (拇指/食指/小指)", 3, "FINGER3"),
    "finger_4": _finger_spec("finger_4", "四指运动执行 (全四指)", 4, "FINGER4"),
    "hand_2class": _hand_spec(
        "hand_2class",
        "手形二分类 (握拳/放松)",
        ("fist", "relax"),
        "HAND2",
        train_timing=HAND_2CLASS_TRAIN_TIMING,
    ),
    "hand_3class": _hand_spec("hand_3class", "手形三分类 (握拳/放松/张掌)", ("fist", "relax", "palm"), "HAND3"),
}

DEFAULT_PARADIGM_ID = "finger_2"

# 下拉框分组顺序（id 列表）
PARADIGM_COMBO_GROUPS: List[Tuple[str, List[str]]] = [
    ("手指运动执行", ["finger_2", "finger_3", "finger_4"]),
    ("手形", ["hand_2class", "hand_3class"]),
]


def get_paradigm(paradigm_id: str) -> ParadigmSpec:
    if paradigm_id not in PARADIGM_REGISTRY:
        raise KeyError(f"未知范式 id={paradigm_id!r}")
    return PARADIGM_REGISTRY[paradigm_id]


def default_paradigm() -> ParadigmSpec:
    return get_paradigm(DEFAULT_PARADIGM_ID)


def paradigm_intro_html(spec: ParadigmSpec, timing: MISessionTiming, *, palette_text: str) -> str:
    """生成采集流程说明弹窗 HTML。"""
    trials_total = timing.n_segments * timing.n_trials_per_segment
    avg_trial = timing.avg_trial_duration_s(list(spec.conditions))
    total_min = trials_total * avg_trial / 60.0

    if spec.family == "finger":
        task_desc = (
            "双手放松平放。屏幕出现<b>手指编号</b>时，<b>实际执行</b>该指屈伸 1 次。<br>"
            "休息期仅显示「+」，保持静止等待下一试次。"
        )
        classes_line = f"· {finger_names_cn(list(spec.conditions))}（marker {spec.marker_summary()}）"
        trial_desc = f"显示手指编号，做该指屈伸 1 次（marker {spec.marker_summary()}）"
        segment_rhythm = f"标签 {timing.label_s:.0f}s + 休息 {timing.rest_s:.0f}s"
        rest_bullet = "<li><b>休息期</b> — 注视「+」（marker 0）</li>"
    elif timing.rest_s <= 0 and timing.condition_label_s:
        order_note = "试次按<b>握拳 ↔ 放松</b>交替出现（随机决定先开始哪一类）。" if timing.trial_order == "alternate" else ""
        task_desc = (
            "双手自然放置。屏幕出现<b>手形提示</b>时，<b>实际执行</b>对应姿态并保持至提示结束。<br>"
            "试次间<b>无</b>注视休息，提示结束即切换下一试次。"
            + (f"<br>{order_note}" if order_note else "")
        )
        classes_line = f"· {spec.conditions_cn()}（marker {spec.marker_summary()}）"
        fist_s = timing.label_duration_for("fist")
        relax_s = timing.label_duration_for("relax")
        trial_desc = f"握拳保持 {fist_s:.0f}s（marker 1）/ 放松保持 {relax_s:.0f}s（marker 2）"
        segment_rhythm = timing.rhythm_summary(list(spec.conditions))
        rest_bullet = ""
    else:
        task_desc = (
            "双手自然放置。屏幕出现<b>手形提示</b>时，<b>实际执行</b>对应动作并保持至提示结束。<br>"
            "试次间休息期显示「+」，保持静止。"
        )
        classes_line = f"· {spec.conditions_cn()}（marker {spec.marker_summary()}）"
        trial_desc = f"按提示完成动作（{spec.conditions_cn()}）"
        segment_rhythm = f"标签 {timing.label_s:.0f}s + 休息 {timing.rest_s:.0f}s"
        rest_bullet = "<li><b>休息期</b> — 注视「+」（marker 0）</li>"

    return f"""
<div style="font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif; font-size: 17px; line-height: 1.75; color: {palette_text};">
<p><b>任务说明</b><br>
{task_desc}</p>

<p><b>类别</b><br>
{classes_line}</p>

<p><b>Session 结构</b>（定时部分约 <b>{total_min:.0f} 分钟</b>）</p>
<ol style="margin-top: 0;">
<li><b>开始前</b> — 按<b>空格键</b>开始 Segment 1（休息时长自定）</li>
<li><b>Segment 1</b> — 约 {timing.n_trials_per_segment} 试次（{segment_rhythm}）</li>
<li><b>段间休息</b> — 按<b>空格键</b>继续下一段（落盘 marker 91 / BREAK）</li>
<li>共 <b>{timing.n_segments}</b> 个 Segment</li>
</ol>
<p>合计交互试次 <b>{trials_total}</b> 次。</p>

<p><b>单试次</b></p>
<ul>
<li><b>标签期</b> — {trial_desc}</li>
{rest_bullet}
</ul>

<p><b>注意事项</b><br>
尽量<b>不眨眼、不咬牙、不移动头部和身体</b>。</p>

<p style="margin-bottom: 0;">确认理解后，点击下方「开始」。</p>
</div>
"""
