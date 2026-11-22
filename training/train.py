"""服务器端训练主脚本：本地定稿 CSV → EEGNet（PyTorch）→ 模型包(.pth + .meta.json)。

用法示例：
    python train.py --data /path/to/csv_or_dir --out models/eegnet_mi.pth
    python train.py --data ../data/acquisition/hand_2class/HAND2_xxx --paradigm hand_2class --out models/eegnet_hand2.pth
    python train.py --data ../data/acquisition/finger_2 --paradigm finger_2 --out models/eegnet_finger2.pth --epochs 300

设计要点：
    - 自包含：本文件夹可整体拷到服务器运行，不依赖本地 BCI-Desk 包。
    - 输入数据走本地定稿格式（timestamp, event_type, EEG Channel 1..16 + meta.json）。
    - 训练/验证按「试次」划分后再切片，避免同一试次的滑窗泄漏到两边。
    - 预处理与模型架构对齐参考工程 Finger-BCI-Decoding（EEGNet-8,2）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import NUM_CHANNELS, TrainConfig, UNIT_MICROVOLT, parse_classes_arg
from data_loader import load_dataset, resolve_classes
from preprocessing import build_dataset
from eegnet import build_eegnet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_trial_indices(
    n_trials: int, train_split: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """在试次层面随机划分训练/验证索引（避免滑窗跨集泄漏）。"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_trials)
    n_train = max(1, int(train_split * n_trials))
    return idx[:n_train], idx[n_train:]


def _subset(trials: List[np.ndarray], labels: np.ndarray, srates: List[int], sel: np.ndarray):
    return (
        [trials[i] for i in sel],
        labels[sel],
        [srates[i] for i in sel],
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int):
    # X: (N, C, T) → (N, 1, C, T)
    tx = torch.from_numpy(X).unsqueeze(1)
    ty = torch.from_numpy(y)
    ds = TensorDataset(tx, ty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def run_epoch(model, loader, criterion, device, optimizer=None) -> Tuple[float, float]:
    """跑一个 epoch；optimizer 为 None 时为评估模式。返回 (loss, acc)。"""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train_mode):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.apply_max_norm()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def save_model_package(
    model, out_path: str, cfg: TrainConfig, chans: int, samples: int, best_val_acc: float
) -> str:
    """保存权重 .pth 和旁车 .meta.json（供本地/推理端用完全一致的预处理重建）。"""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    p = cfg.preproc
    m = cfg.model
    meta = {
        "model_name": "EEGNet",
        "framework": "pytorch",
        "classes": cfg.classes,
        "num_channels": chans,
        "input_samples": samples,
        "sample_rate_train": p.sample_rate,
        "expected_input_unit": UNIT_MICROVOLT,
        "preproc": {
            "max_trial_s": p.max_trial_s,
            "window_s": p.window_s,
            "step_samples": p.step_samples,
            "downsample_rate": p.downsample_rate,
            "bandpass": list(p.bandpass),
            "bandpass_order": p.bandpass_order,
            "use_car": p.use_car,
            "desired_len": p.desired_len,
            "zscore": "per_channel_time",
        },
        "eegnet": {
            "kern_length": m.kern_length,
            "F1": m.F1,
            "D": m.D,
            "F2": m.F2,
            "dropout_rate": m.dropout_rate,
            "norm_rate": m.norm_rate,
        },
        "val_accuracy": round(float(best_val_acc), 4),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="EEGNet 运动想象训练（PyTorch）")
    parser.add_argument("--data", required=True, help="带标签 CSV 文件或目录")
    parser.add_argument("--out", default="models/eegnet_mi.pth", help="输出模型路径(.pth)")
    parser.add_argument(
        "--paradigm",
        choices=["finger_2", "finger_3", "finger_4", "hand_2class", "hand_3class"],
        help="范式 id（与本地采集 meta.paradigm_id 一致）；未指定时尝试从 meta 推断",
    )
    parser.add_argument(
        "--classes",
        help="逗号分隔类别，如 THUMB,PINKY 或 FIST,RELAX（优先级高于 --paradigm）",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed

    explicit_classes = parse_classes_arg(args.classes) if args.classes else None
    cfg.classes = resolve_classes(
        args.data,
        classes=explicit_classes,
        paradigm_id=args.paradigm,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    log.info("设备: %s", device)
    log.info("训练类别 (%d 类): %s", len(cfg.classes), cfg.classes)

    # 1) 加载试次
    trials, labels, srates = load_dataset(args.data, classes=cfg.classes, num_channels=NUM_CHANNELS)
    log.info("加载 %d 个试次，类别分布: %s", len(trials),
             {cfg.classes[i]: int((labels == i).sum()) for i in range(len(cfg.classes))})

    # 2) 试次层面划分（防止滑窗泄漏）
    tr_idx, va_idx = split_trial_indices(len(trials), cfg.train_split, cfg.seed)
    tr_trials, tr_labels, tr_sr = _subset(trials, labels, srates, tr_idx)
    X_train, y_train = build_dataset(tr_trials, tr_labels, tr_sr, cfg.preproc)
    log.info("训练样本(滑窗后): X=%s y=%s", X_train.shape, y_train.shape)

    has_val = len(va_idx) > 0
    if has_val:
        va_trials, va_labels, va_sr = _subset(trials, labels, srates, va_idx)
        X_val, y_val = build_dataset(va_trials, va_labels, va_sr, cfg.preproc)
        log.info("验证样本(滑窗后): X=%s y=%s", X_val.shape, y_val.shape)
    else:
        log.warning("试次过少，无验证集；将仅用训练集并以训练精度保存最优。")

    chans, samples = X_train.shape[1], X_train.shape[2]

    # 3) 构建模型 / 优化器 / 调度
    model = build_eegnet(len(cfg.classes), chans, samples, cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
    )

    train_loader = make_loader(X_train, y_train, cfg.batch_size, True, cfg.num_workers)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, False, cfg.num_workers) if has_val else None

    best_metric = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        if has_val:
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device, None)
            scheduler.step(va_loss)
            monitor_acc = va_acc
            log.info("epoch %3d/%d  train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f",
                     epoch, cfg.epochs, tr_loss, tr_acc, va_loss, va_acc)
        else:
            scheduler.step(tr_loss)
            monitor_acc = tr_acc
            log.info("epoch %3d/%d  train_loss=%.4f acc=%.3f",
                     epoch, cfg.epochs, tr_loss, tr_acc)

        if monitor_acc > best_metric:
            best_metric = monitor_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                log.info("早停于 epoch %d（%d 轮无提升）", epoch, cfg.early_stop_patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    meta_path = save_model_package(model, args.out, cfg, chans, samples, best_metric)
    log.info("训练完成。最优监控精度=%.3f", best_metric)
    log.info("模型已保存 → %s", args.out)
    log.info("元数据已保存 → %s", meta_path)


if __name__ == "__main__":
    main()
