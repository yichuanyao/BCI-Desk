"""ME 采集 session 命名、目录与 manifest 管理。

完整离线 session 目标为 32 run，建议分 8 次采集，每次 4 run。
每次采集落盘为一个 block CSV + meta；同一 session 目录下 8 个 block
可直接作为 training/train.py 的 --data 输入（目录内所有 CSV 自动合并）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from bci_desk.paths import TRAIN_RECORD_ROOT, TEST_RECORD_ROOT
DEFAULT_TOTAL_RUNS = 32
DEFAULT_RUNS_PER_BLOCK = 4
DEFAULT_NUM_BLOCKS = DEFAULT_TOTAL_RUNS // DEFAULT_RUNS_PER_BLOCK  # 8


def new_session_id(prefix: str = "ME") -> str:
    """生成 session 标识，如 ME_20260618_143052。"""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def block_run_range(block_index: int, runs_per_block: int = DEFAULT_RUNS_PER_BLOCK) -> Tuple[int, int]:
    """block 序号(1-based) → (run_start, run_end)。"""
    if block_index < 1:
        raise ValueError("block_index 须 >= 1")
    run_start = (block_index - 1) * runs_per_block + 1
    run_end = block_index * runs_per_block
    return run_start, run_end


def make_block_basename(
    session_id: str,
    block_index: int,
    run_start: int,
    run_end: int,
    purpose: str = "train",
) -> str:
    """不含扩展名的 block 文件名 stem。"""
    tag = "test" if purpose == "test" else "train"
    if purpose == "test" and run_start == run_end:
        return f"{session_id}_{tag}_run{run_start:02d}"
    return f"{session_id}_{tag}_block{block_index:02d}_runs{run_start:02d}-{run_end:02d}"


def session_dir(base_dir: str, session_id: str) -> str:
    return os.path.join(base_dir, session_id)


def mi_session_csv_path(
    base_dir: str,
    session_id: str,
    purpose: str = "train",
    paradigm_subdir: str = "",
) -> str:
    """完整 ME session 单文件 CSV 路径（一次采集落盘一个文件）。"""
    root = os.path.join(base_dir, paradigm_subdir) if paradigm_subdir else base_dir
    folder = session_dir(root, session_id)
    os.makedirs(folder, exist_ok=True)
    tag = "test" if purpose == "test" else "train"
    return os.path.join(folder, f"{session_id}_{tag}_me.csv")


def block_csv_path(
    base_dir: str,
    session_id: str,
    block_index: int,
    runs_per_block: int = DEFAULT_RUNS_PER_BLOCK,
    purpose: str = "train",
) -> str:
    """计算 block CSV 的完整路径（目录会自动创建）。"""
    run_start, run_end = block_run_range(block_index, runs_per_block)
    folder = session_dir(base_dir, session_id)
    os.makedirs(folder, exist_ok=True)
    stem = make_block_basename(session_id, block_index, run_start, run_end, purpose=purpose)
    return os.path.join(folder, f"{stem}.csv")


def manifest_path(session_folder: str) -> str:
    return os.path.join(session_folder, "session.json")


def load_manifest(session_folder: str) -> dict:
    path = manifest_path(session_folder)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def update_manifest(
    session_folder: str,
    session_id: str,
    block_index: int,
    run_start: int,
    run_end: int,
    csv_path: str,
    *,
    purpose: str = "train",
    runs_per_block: int = DEFAULT_RUNS_PER_BLOCK,
    total_runs_target: int = DEFAULT_TOTAL_RUNS,
) -> dict:
    """追加或更新 block 记录，写 session.json。"""
    os.makedirs(session_folder, exist_ok=True)
    manifest = load_manifest(session_folder)
    if not manifest:
        manifest = {
            "session_id": session_id,
            "purpose": purpose,
            "total_runs_target": total_runs_target,
            "runs_per_block": runs_per_block,
            "blocks": [],
        }

    entry = {
        "block_index": block_index,
        "run_start": run_start,
        "run_end": run_end,
        "csv": os.path.basename(csv_path),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    blocks: List[dict] = manifest.setdefault("blocks", [])
    blocks = [b for b in blocks if b.get("block_index") != block_index]
    blocks.append(entry)
    blocks.sort(key=lambda b: b.get("block_index", 0))
    manifest["blocks"] = blocks
    manifest["blocks_completed"] = len(blocks)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")

    with open(manifest_path(session_folder), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def next_block_index(session_folder: str) -> int:
    """根据 manifest 推断下一个待采集 block（1-based）。"""
    manifest = load_manifest(session_folder)
    blocks = manifest.get("blocks", [])
    if not blocks:
        return 1
    return max(b.get("block_index", 0) for b in blocks) + 1


def session_summary(manifest: dict) -> str:
    """人类可读的 session 进度摘要。"""
    done = manifest.get("blocks_completed", len(manifest.get("blocks", [])))
    target = DEFAULT_NUM_BLOCKS
    total_runs = manifest.get("total_runs_target", DEFAULT_TOTAL_RUNS)
    runs_done = sum(
        (b.get("run_end", 0) - b.get("run_start", 0) + 1)
        for b in manifest.get("blocks", [])
    )
    return f"已采集 {done}/{target} 块 · {runs_done}/{total_runs} run"
