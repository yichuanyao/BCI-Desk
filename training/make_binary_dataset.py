"""把已采集的四分类（THUMB/INDEX/MIDDLE/PINKY）数据批量转成二分类数据集。

当前训练任务为二分类（仅 拇指 THUMB / 小指 PINKY，对齐参考工程 Finger-BCI-Decoding）。
本脚本读取本地采集定稿格式的带标签 CSV（timestamp, event_type, EEG Channel 1..16），
仅保留 THUMB / PINKY 两类试次，把 INDEX / MIDDLE 段替换成单行哨兵 ``GAP``：

    - 减小数据体积（不上传无关类别的样本）；
    - 关键：保留原始「试次边界」。data_loader 按「连续相同 event_type」切试次，
      若直接删除 INDEX/MIDDLE 行，原本被它们隔开的两个同类试次会被错误合并；
      用 GAP 哨兵分隔可保证切分结果与「在原始四类 CSV 上直接用二分类 CLASSES 加载」完全等价。
      （GAP 不在 CLASSES 内，训练加载时会被自动跳过。）

用法：
    # 整个 finger_4 采集根目录（递归，保持目录结构）
    python make_binary_dataset.py --input ../data/acquisition/finger_4 --output ../data/acquisition_binary/finger_4

    # 单个 session 目录
    python make_binary_dataset.py --input ../data/acquisition/finger_4/FINGER4_20260618_142446 --output ../data/acquisition_binary/finger_4/FINGER4_20260618_142446

    # 单个 CSV 文件（输出写到指定目录下，同名）
    python make_binary_dataset.py --input xxx.csv --output out_dir

转换后把 ``--output`` 目录上传服务器，训练：
    python train.py --data /path/to/data/acquisition_binary --out models/eegnet_thumb_pinky.pth
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List

import pandas as pd

from config import PARADIGM_CLASSES, parse_classes_arg

# 默认保留手指二分类（finger_2）
DEFAULT_KEEP_PARADIGM = "finger_2"
KEEP_LABELS: List[str] = list(PARADIGM_CLASSES[DEFAULT_KEEP_PARADIGM])
# 替换 INDEX/MIDDLE 段的哨兵标签：不在 CLASSES 内，加载时被跳过，仅用于保持试次边界。
GAP_LABEL = "GAP"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉列名两侧空白及可能的 UTF-8 BOM 前缀。"""
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def convert_csv(in_csv: str, out_csv: str, keep: List[str]) -> Dict:
    """转换单个 CSV：保留 keep 的试次块，其余块替换为单行 GAP 哨兵。

    Returns:
        统计 dict：{kept_trials_per_class, n_rows_in, n_rows_out}。
    """
    df = _clean_columns(pd.read_csv(in_csv))
    if "event_type" not in df.columns:
        raise ValueError(f"{in_csv} 缺少 'event_type' 列。")
    eeg_cols = [c for c in df.columns if c.startswith("EEG")]

    # 与 data_loader 一致：连续相同 event_type 归为一个块（试次）。
    ev = df["event_type"]
    block_id = ev.ne(ev.shift()).cumsum()

    out_blocks: List[pd.DataFrame] = []
    kept = {k: 0 for k in keep}
    for _, seg in df.groupby(block_id, sort=False):
        label = seg["event_type"].iloc[0]
        if label in keep:
            out_blocks.append(seg)
            kept[label] += 1
        else:
            # 整段（约 5s×125 行）压缩为 1 行哨兵：保留分隔作用，体积接近 0。
            gap = seg.iloc[[0]].copy()
            gap["event_type"] = GAP_LABEL
            gap.loc[:, eeg_cols] = 0.0
            out_blocks.append(gap)

    out_df = pd.concat(out_blocks, ignore_index=True)

    # 去掉首尾连续的 GAP 哨兵，让文件更干净（中间的 GAP 必须保留以分隔试次）。
    not_gap = out_df["event_type"] != GAP_LABEL
    if not_gap.any():
        first = not_gap.idxmax()
        last = not_gap[::-1].idxmax()
        out_df = out_df.loc[first:last].reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return {
        "kept_trials_per_class": kept,
        "n_rows_in": int(len(df)),
        "n_rows_out": int(len(out_df)),
    }


def write_meta(in_csv: str, out_csv: str, keep: List[str]) -> None:
    """基于源 CSV 的旁车 meta 生成输出 meta（保留 unit/sample_rate，标注二分类筛选）。"""
    src_meta = os.path.splitext(in_csv)[0] + ".meta.json"
    dst_meta = os.path.splitext(out_csv)[0] + ".meta.json"
    meta: Dict = {}
    if os.path.isfile(src_meta):
        try:
            with open(src_meta, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = {}
    else:
        print(f"  [警告] 未找到源 meta：{src_meta}（输出 meta 将缺少 unit/sample_rate，"
              f"data_loader 会按 raw_count 处理，可能导致单位错误！）")

    meta["binary_filter"] = {
        "keep": list(keep),
        "gap_label": GAP_LABEL,
        "source_csv": os.path.basename(in_csv),
    }
    os.makedirs(os.path.dirname(dst_meta) or ".", exist_ok=True)
    with open(dst_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _resolve_base_and_csvs(input_path: str):
    """返回 (base_dir, [csv 绝对路径...])。base_dir 用于计算输出的相对结构。"""
    if os.path.isfile(input_path):
        if not input_path.endswith(".csv"):
            raise ValueError(f"输入文件不是 .csv：{input_path}")
        return os.path.dirname(os.path.abspath(input_path)), [os.path.abspath(input_path)]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    csvs = []
    for root, _, files in os.walk(input_path):
        for fn in sorted(files):
            if fn.endswith(".csv"):
                csvs.append(os.path.join(root, fn))
    if not csvs:
        raise FileNotFoundError(f"在 {input_path} 下未找到任何 .csv 文件。")
    return os.path.abspath(input_path), sorted(csvs)


def _copy_aux_jsons(base: str, out_root: str) -> None:
    """复制 session.json 等非 meta 的 .json（保留目录结构）。"""
    if not os.path.isdir(base):
        return
    for root, _, files in os.walk(base):
        for fn in files:
            if fn.endswith(".json") and not fn.endswith(".meta.json"):
                src = os.path.join(root, fn)
                rel = os.path.relpath(src, base)
                dst = os.path.join(out_root, rel)
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="四分类采集数据 → 二分类(THUMB/PINKY)数据集批处理"
    )
    parser.add_argument("--input", required=True, help="原始 CSV 文件、session 目录或采集根目录")
    parser.add_argument("--output", required=True, help="输出目录（保持与输入相同的子目录结构）")
    parser.add_argument(
        "--paradigm",
        default=DEFAULT_KEEP_PARADIGM,
        choices=["finger_2"],
        help="仅 finger_2：从四指采集中筛出 THUMB/PINKY（默认）",
    )
    parser.add_argument(
        "--keep",
        help="逗号分隔要保留的 event_type，如 THUMB,PINKY（覆盖 --paradigm）",
    )
    args = parser.parse_args()

    keep = parse_classes_arg(args.keep) if args.keep else list(PARADIGM_CLASSES[args.paradigm])

    base, csvs = _resolve_base_and_csvs(args.input)
    out_root = os.path.abspath(args.output)
    os.makedirs(out_root, exist_ok=True)

    print(f"二分类类别：{keep}（其余类别替换为哨兵 {GAP_LABEL!r}）")
    print(f"输入：{args.input}")
    print(f"输出：{out_root}")
    print(f"待处理 CSV：{len(csvs)} 个\n")

    total = {k: 0 for k in keep}
    for in_csv in csvs:
        rel = os.path.relpath(in_csv, base)
        out_csv = os.path.join(out_root, rel)
        stats = convert_csv(in_csv, out_csv, keep)
        write_meta(in_csv, out_csv, keep)
        for k, v in stats["kept_trials_per_class"].items():
            total[k] += v
        print(f"  {rel}: 试次 {stats['kept_trials_per_class']}  "
              f"行 {stats['n_rows_in']} → {stats['n_rows_out']}")

    _copy_aux_jsons(base, out_root)

    print(f"\n完成。二分类试次合计：{total}")
    print(f"上传 {out_root} 到服务器后即可训练：")
    print(f"  python train.py --data <上传后的路径> --out models/eegnet_thumb_pinky.pth")


if __name__ == "__main__":
    main()
