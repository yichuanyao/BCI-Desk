"""运动执行 / 交互式 ME 提示范式的时序逻辑。

当前主流程（多范式 · 手指/手形 ME）：

    Segment 1~3 各 15 min（~300 试次，标签 1s + 休息 2s）
    段前 / 段间休息由操作者按空格继续（自定休息时长，段间落盘 BREAK）

单试次（≈3 s）：
    标签期 — 显示提示 marker，实际执行对应动作
    休息期 — 注视点 marker 0
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..config import (
    CONDITION_TO_LABEL,
    CONDITION_TO_MARKER,
    EVENT_TYPE_BREAK,
    EVENT_TYPE_RELAX,
    EVENT_TYPE_REST,
    LABEL_NAMES_CN,
    MARKER_BREAK,
    MARKER_RELAX,
    MARKER_REST,
    MI_DEFAULT_CONDITIONS,
)

# ---------- 旧版 ME 默认（兼容） ----------
DEFAULT_CONDITIONS = ["thumb", "index", "middle", "pinky"]
DEFAULT_TOTAL_RUNS = 32
DEFAULT_RUNS_PER_BLOCK = 4
DEFAULT_N_RUNS = DEFAULT_RUNS_PER_BLOCK
DEFAULT_TRIALS_PER_COND_PER_RUN = 5
DEFAULT_NUM_BLOCKS = DEFAULT_TOTAL_RUNS // DEFAULT_RUNS_PER_BLOCK

# ---------- 阶段类型 ----------
KIND_PREPARE = "prepare"
KIND_CUE = "cue"
KIND_EXECUTE = "execute"
KIND_ITI = "iti"
KIND_RUNBREAK = "runbreak"

KIND_RELAX = "relax"
KIND_BREAK = "break"
KIND_LABEL = "label"
KIND_REST = "rest"
KIND_MANUAL_WAIT = "manual_wait"


@dataclass
class ParadigmTiming:
    """旧版 ME 各阶段时长（秒）。"""
    prepare: float = 1.0
    cue: float = 2.0
    execute: float = 5.0
    iti: float = 2.0
    runbreak: float = 15.0


@dataclass
class MISessionTiming:
    """ME session 各段时长与试次数。"""
    relax_s: float = 150.0           # 2.5 min
    break_s: float = 120.0            # 2 min
    label_s: float = 1.0              # 标签期默认时长
    rest_s: float = 2.0               # 试次间休息；0 表示无试次间休息
    n_trials_per_segment: int = 300
    n_segments: int = 3
    condition_label_s: Optional[Dict[str, float]] = None  # 按 condition 覆盖 label_s
    trial_order: str = "random"  # "random" | "alternate"（均衡交替，避免连续同类）

    def label_duration_for(self, condition: str) -> float:
        if self.condition_label_s and condition in self.condition_label_s:
            return self.condition_label_s[condition]
        return self.label_s

    def avg_trial_duration_s(self, conditions: Optional[List[str]] = None) -> float:
        if conditions:
            label_part = sum(self.label_duration_for(c) for c in conditions) / len(conditions)
        else:
            label_part = self.label_s
        return label_part + max(0.0, self.rest_s)

    def rhythm_summary(self, conditions: Optional[List[str]] = None) -> str:
        """人类可读的节奏说明（用于界面 ⓘ）。"""
        if self.condition_label_s and conditions:
            parts = []
            for c in conditions:
                parts.append(f"{c} {self.label_duration_for(c):.0f}s")
            label_desc = " / ".join(parts)
        else:
            label_desc = f"标签 {self.label_s:.0f}s"
        if self.rest_s > 0:
            return f"{label_desc} + 休息 {self.rest_s:.0f}s"
        suffix = "，试次间无休息"
        if self.trial_order == "alternate":
            suffix += " · 交替出现"
        return f"{label_desc}{suffix}"


@dataclass
class Phase:
    """时间表中的一个阶段。"""
    kind: str
    condition: str
    label: str
    duration_s: float
    trial_num: int
    total_trials: int
    run_num: int = 0
    total_runs: int = 0
    segment_num: int = 0
    total_segments: int = 0
    marker: int = 0
    event_type: str = ""
    display_name_cn: str = ""

    @property
    def condition_cn(self) -> str:
        if self.display_name_cn:
            return self.display_name_cn
        return LABEL_NAMES_CN.get(self.label, self.condition)

    @property
    def is_execute(self) -> bool:
        """旧 ME：execute 期打标签；新 MI：label 期打手指标签。"""
        return self.kind in (KIND_EXECUTE, KIND_LABEL)

    @property
    def is_tagged(self) -> bool:
        """该阶段是否应写入 event_type 时间窗。"""
        return bool(self.event_type)

    @property
    def is_manual_wait(self) -> bool:
        return self.kind == KIND_MANUAL_WAIT


def generate_balanced_alternate_trials(
    n_trials: int,
    conditions: List[str],
    seed: Optional[int] = None,
) -> List[str]:
    """均衡数量、严格交替（2 类）；奇数试次时多出的 1 次放在末尾且不会连出 3 连。"""
    if len(conditions) != 2:
        raise ValueError("alternate 试次顺序仅支持 2 类条件")
    if n_trials < 1:
        raise ValueError("n_trials 须 >= 1")
    rng = random.Random(seed)
    c0, c1 = conditions[0], conditions[1]
    counts = {c0: n_trials // 2 + (n_trials % 2), c1: n_trials // 2}
    if counts[c0] > counts[c1]:
        start = c0
    elif counts[c1] > counts[c0]:
        start = c1
    else:
        start = rng.choice([c0, c1])
    trials: List[str] = []
    cur = start
    while len(trials) < n_trials:
        trials.append(cur)
        counts[cur] -= 1
        other = c1 if cur == c0 else c0
        if counts[other] > 0:
            cur = other
    return trials


def generate_segment_trials(
    n_trials: int,
    conditions: Optional[List[str]] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    order: str = "random",
) -> List[str]:
    """生成一个 Segment 内试次列表。

    order:
        random — 独立随机抽样后 shuffle（手指范式默认）
        alternate — 两类数量尽量均衡且严格交替（手形二分类）
    """
    if conditions is None:
        conditions = MI_DEFAULT_CONDITIONS
    if n_trials < 1:
        raise ValueError("n_trials 须 >= 1")
    if order == "alternate":
        return generate_balanced_alternate_trials(n_trials, list(conditions), seed=seed)
    rng = random.Random(seed)
    trials = [rng.choice(conditions) for _ in range(n_trials)]
    if shuffle:
        rng.shuffle(trials)
    return trials


def build_mi_session_schedule(
    conditions: Optional[List[str]] = None,
    timing: Optional[MISessionTiming] = None,
    *,
    manual_waits: bool = True,
    seed: Optional[int] = None,
    condition_to_label: Optional[dict] = None,
    condition_to_marker: Optional[dict] = None,
    label_names_cn: Optional[dict] = None,
) -> List[Phase]:
    """构建 ME/MI 交互 session：段前/段间由操作者按空格继续（自定休息时长）。"""
    if conditions is None:
        conditions = MI_DEFAULT_CONDITIONS
    if timing is None:
        timing = MISessionTiming()
    if condition_to_label is None:
        condition_to_label = CONDITION_TO_LABEL
    if condition_to_marker is None:
        condition_to_marker = CONDITION_TO_MARKER
    if label_names_cn is None:
        label_names_cn = LABEL_NAMES_CN

    schedule: List[Phase] = []
    seg_rng_seed = seed

    for seg in range(1, timing.n_segments + 1):
        if manual_waits:
            if seg == 1:
                schedule.append(
                    Phase(
                        kind=KIND_MANUAL_WAIT,
                        condition="",
                        label="",
                        marker=0,
                        event_type="",
                        duration_s=0.0,
                        trial_num=0,
                        total_trials=0,
                        segment_num=0,
                        total_segments=timing.n_segments,
                    )
                )
            else:
                schedule.append(
                    Phase(
                        kind=KIND_MANUAL_WAIT,
                        condition="",
                        label="",
                        marker=MARKER_BREAK,
                        event_type=EVENT_TYPE_BREAK,
                        duration_s=0.0,
                        trial_num=0,
                        total_trials=0,
                        segment_num=seg - 1,
                        total_segments=timing.n_segments,
                    )
                )

        trials = generate_segment_trials(
            timing.n_trials_per_segment,
            conditions=conditions,
            shuffle=True,
            seed=None if seg_rng_seed is None else seg_rng_seed + seg,
            order=timing.trial_order,
        )
        for i, cond in enumerate(trials, start=1):
            label = condition_to_label.get(cond, cond)
            marker = condition_to_marker.get(cond, 0)
            display_name = label_names_cn.get(label, label_names_cn.get(cond, cond))
            schedule.append(
                Phase(
                    kind=KIND_LABEL,
                    condition=cond,
                    label=label,
                    marker=marker,
                    event_type=label,
                    duration_s=timing.label_duration_for(cond),
                    trial_num=i,
                    total_trials=timing.n_trials_per_segment,
                    segment_num=seg,
                    total_segments=timing.n_segments,
                    display_name_cn=display_name,
                )
            )
            if timing.rest_s > 0:
                schedule.append(
                    Phase(
                        kind=KIND_REST,
                        condition="",
                        label="",
                        marker=MARKER_REST,
                        event_type=EVENT_TYPE_REST,
                        duration_s=timing.rest_s,
                        trial_num=i,
                        total_trials=timing.n_trials_per_segment,
                        segment_num=seg,
                        total_segments=timing.n_segments,
                    )
                )

    return schedule


def mi_session_duration_s(timing: Optional[MISessionTiming] = None, **kwargs) -> float:
    """预估 MI session 中定时阶段的时长（不含空格等待的休息）。"""
    kwargs.pop("manual_waits", None)
    return total_duration_s(build_mi_session_schedule(timing=timing, manual_waits=False, **kwargs))


# ---------- 旧版 ME（兼容） ----------

def generate_session(
    conditions: Optional[List[str]] = None,
    n_runs: int = DEFAULT_N_RUNS,
    trials_per_cond_per_run: int = DEFAULT_TRIALS_PER_COND_PER_RUN,
    shuffle: bool = True,
    seed: Optional[int] = None,
    run_start: int = 1,
) -> List[Tuple[int, str]]:
    if conditions is None:
        conditions = DEFAULT_CONDITIONS
    if run_start < 1:
        raise ValueError("run_start 须 >= 1")
    rng = random.Random(seed)

    trials: List[Tuple[int, str]] = []
    for run in range(run_start, run_start + n_runs):
        block: List[str] = []
        for cond in conditions:
            block.extend([cond] * trials_per_cond_per_run)
        if shuffle:
            rng.shuffle(block)
        trials.extend((run, cond) for cond in block)
    return trials


def generate_trial_list(
    conditions: Optional[List[str]] = None,
    n_repeats: int = 5,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[Tuple[int, str]]:
    return generate_session(
        conditions=conditions,
        n_runs=1,
        trials_per_cond_per_run=n_repeats,
        shuffle=shuffle,
        seed=seed,
    )


def build_schedule(
    trials: List[Tuple[int, str]],
    timing: Optional[ParadigmTiming] = None,
    condition_to_label: Optional[dict] = None,
) -> List[Phase]:
    if timing is None:
        timing = ParadigmTiming()
    if condition_to_label is None:
        condition_to_label = CONDITION_TO_LABEL

    total = len(trials)
    total_runs = max((r for r, _ in trials), default=0)
    schedule: List[Phase] = []
    prev_run: Optional[int] = None

    for i, (run_num, cond) in enumerate(trials, start=1):
        if prev_run is not None and run_num != prev_run and timing.runbreak > 0:
            schedule.append(
                Phase(
                    kind=KIND_RUNBREAK,
                    condition="",
                    label="",
                    duration_s=timing.runbreak,
                    trial_num=0,
                    total_trials=total,
                    run_num=run_num,
                    total_runs=total_runs,
                    event_type=EVENT_TYPE_BREAK,
                    marker=MARKER_BREAK,
                )
            )
        prev_run = run_num

        label = condition_to_label.get(cond, cond)
        marker = CONDITION_TO_MARKER.get(cond, 0)
        for kind, dur, et in (
            (KIND_PREPARE, timing.prepare, ""),
            (KIND_CUE, timing.cue, ""),
            (KIND_EXECUTE, timing.execute, label),
            (KIND_ITI, timing.iti, EVENT_TYPE_REST),
        ):
            schedule.append(
                Phase(
                    kind=kind,
                    condition=cond if kind != KIND_ITI else "",
                    label=label if kind == KIND_EXECUTE else "",
                    marker=marker if kind == KIND_EXECUTE else (MARKER_REST if kind == KIND_ITI else 0),
                    event_type=et,
                    duration_s=dur,
                    trial_num=i,
                    total_trials=total,
                    run_num=run_num,
                    total_runs=total_runs,
                )
            )
    return schedule


def total_duration_s(schedule: List[Phase]) -> float:
    return float(sum(p.duration_s for p in schedule))
