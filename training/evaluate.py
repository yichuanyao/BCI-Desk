"""离线评估：加载训练产出的 .pth + .meta.json，对带标签 CSV 计算试次级准确率。

用法：
    python evaluate.py --model models/eegnet_mi.pth --data ../data/test/ME_test_xxx/ME_test_xxx_test_run01.csv
    python evaluate.py --model models/eegnet_mi.pth --data ../data/test/some_dir/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import EEGNetConfig, PreprocConfig, TrainConfig
from data_loader import load_dataset, list_csv_files
from eegnet import build_eegnet
from preprocessing import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")


def load_model_package(pth_path: str, device: torch.device):
    """加载权重与 meta，重建 EEGNet。"""
    meta_path = os.path.splitext(pth_path)[0] + ".meta.json"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"缺少旁车 meta: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    p = meta["preproc"]
    m = meta["eegnet"]
    preproc = PreprocConfig(
        sample_rate=int(meta.get("sample_rate_train", 500)),
        max_trial_s=float(p["max_trial_s"]),
        window_s=float(p["window_s"]),
        step_samples=int(p["step_samples"]),
        downsample_rate=int(p["downsample_rate"]),
        bandpass=tuple(p["bandpass"]),
        bandpass_order=int(p.get("bandpass_order", 4)),
        bandpass_pad=int(p.get("bandpass_pad", 100)),
        use_car=bool(p.get("use_car", True)),
    )
    model_cfg = EEGNetConfig(
        kern_length=int(m["kern_length"]),
        F1=int(m["F1"]),
        D=int(m["D"]),
        F2=int(m["F2"]),
        dropout_rate=float(m["dropout_rate"]),
        norm_rate=float(m["norm_rate"]),
    )
    classes: List[str] = list(meta["classes"])
    chans = int(meta["num_channels"])
    samples = int(meta["input_samples"])

    model = build_eegnet(
        nb_classes=len(classes),
        chans=chans,
        samples=samples,
        cfg=model_cfg,
    )
    state = torch.load(pth_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, meta, preproc, classes


def predict_trial_class(
    model,
    trial: np.ndarray,
    true_label: int,
    sample_rate: int,
    preproc: PreprocConfig,
    device: torch.device,
) -> int:
    """对单个试次滑窗推理，多数票决定预测类。"""
    X, _ = build_dataset([trial], np.array([true_label], dtype=np.int64), [sample_rate], preproc)
    if X.shape[0] == 0:
        raise RuntimeError("试次过短，无法切片。")
    tx = torch.from_numpy(X).unsqueeze(1).to(device)
    with torch.no_grad():
        logits = model(tx)
        preds = logits.argmax(1).cpu().numpy()
    counts = np.bincount(preds, minlength=int(preds.max()) + 1)
    return int(counts.argmax())


def evaluate_trials(
    model,
    trials: List[np.ndarray],
    labels: np.ndarray,
    sample_rates: List[int],
    preproc: PreprocConfig,
    classes: List[str],
    device: torch.device,
) -> dict:
    """试次级评估，返回准确率与混淆统计。"""
    n = len(trials)
    if n == 0:
        raise RuntimeError("没有试次可评估。")

    correct = 0
    per_class = {c: {"total": 0, "correct": 0} for c in classes}
    details = []

    for i, (trial, y_true, sr) in enumerate(zip(trials, labels, sample_rates)):
        y_pred = predict_trial_class(model, trial, int(y_true), sr, preproc, device)
        ok = y_pred == int(y_true)
        correct += int(ok)
        cname = classes[int(y_true)]
        per_class[cname]["total"] += 1
        per_class[cname]["correct"] += int(ok)
        details.append(
            {
                "trial_index": i,
                "true": cname,
                "pred": classes[y_pred],
                "correct": ok,
            }
        )

    acc = correct / n
    return {
        "accuracy": acc,
        "n_trials": n,
        "n_correct": correct,
        "per_class": per_class,
        "details": details,
        "classes": classes,
    }


def evaluate_file(pth_path: str, csv_path: str, device: Optional[torch.device] = None) -> dict:
    """评估单个 CSV 文件。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, preproc, classes = load_model_package(pth_path, device)
    trials, labels, srs = load_dataset(csv_path, classes)
    result = evaluate_trials(model, trials, labels, srs, preproc, classes, device)
    result["csv_path"] = csv_path
    result["model_path"] = pth_path
    result["meta_val_accuracy"] = meta.get("val_accuracy")
    return result


def evaluate_path(pth_path: str, data_path: str, device: Optional[torch.device] = None) -> dict:
    """评估目录（多个 CSV 合并）或单文件。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, preproc, classes = load_model_package(pth_path, device)
    trials, labels, srs = load_dataset(data_path, classes)
    result = evaluate_trials(model, trials, labels, srs, preproc, classes, device)
    result["data_path"] = data_path
    result["model_path"] = pth_path
    result["meta_val_accuracy"] = meta.get("val_accuracy")
    return result


def main():
    parser = argparse.ArgumentParser(description="EEGNet 离线试次级评估")
    parser.add_argument("--model", required=True, help=".pth 模型权重路径")
    parser.add_argument("--data", required=True, help="CSV 文件或目录")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if os.path.isdir(args.data):
        result = evaluate_path(args.model, args.data, device)
    else:
        result = evaluate_file(args.model, args.data, device)

    log.info("试次 %d, 正确 %d, 准确率 %.2f%%", result["n_trials"], result["n_correct"], result["accuracy"] * 100)
    for c, st in result["per_class"].items():
        if st["total"]:
            log.info("  %s: %d/%d (%.1f%%)", c, st["correct"], st["total"], 100 * st["correct"] / st["total"])


if __name__ == "__main__":
    main()
