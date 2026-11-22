"""EEGNet（PyTorch 实现，架构对齐参考工程 EEGModels_tf.py 的 Keras EEGNet-8,2）。

输入张量形状 (N, 1, Chans, Samples)，输出 (N, nb_classes) 的 logits。
（训练用 CrossEntropyLoss，内部含 softmax；如需概率在推理处再 softmax。）

与 Keras 版的对应：
    Conv2D(F1,(1,kernLength),same,bias=False) + BN
    DepthwiseConv2D((Chans,1),depth_multiplier=D,bias=False,max_norm=1) + BN + ELU + AvgPool(1,4) + Dropout
    SeparableConv2D(F2,(1,16),same,bias=False) + BN + ELU + AvgPool(1,8) + Dropout
    Flatten + Dense(nb_classes, max_norm=norm_rate)

Keras 的 max_norm 权重约束在 PyTorch 中没有内置层，这里在每个优化步后调用
`apply_max_norm()` 手动裁剪对应权重的范数（见 train.py）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        nb_classes: int,
        chans: int = 16,
        samples: int = 100,
        dropout_rate: float = 0.5,
        kern_length: int = 32,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        norm_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.chans = chans
        self.samples = samples
        self.norm_rate = norm_rate

        # Block 1: 时间卷积 → BN → 深度卷积(空间滤波) → BN → ELU → 池化 → Dropout
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kern_length), padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (chans, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: 可分离卷积(深度(1,16)+逐点(1,1)) → BN → ELU → 池化 → Dropout
        self.sep_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False
        )
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        self.elu = nn.ELU()

        # 两次池化后时间长度：samples // 4 // 8
        pooled_len = (samples // 4) // 8
        if pooled_len < 1:
            raise ValueError(
                f"samples={samples} 经 4×8 池化后长度为 0，请增大 window_s×downsample_rate。"
            )
        self.classify = nn.Linear(F2 * pooled_len, nb_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, Chans, Samples)
        x = self.bn1(self.conv_temporal(x))
        x = self.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))

        x = self.sep_pointwise(self.sep_depthwise(x))
        x = self.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))

        x = torch.flatten(x, start_dim=1)
        return self.classify(x)

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        """复刻 Keras max_norm 约束：深度卷积权重 max_norm=1，分类层 max_norm=norm_rate。"""
        _max_norm_(self.depthwise.weight, 1.0)
        _max_norm_(self.classify.weight, self.norm_rate)


def _max_norm_(weight: torch.Tensor, max_val: float) -> None:
    """按输出单元裁剪权重的 L2 范数到 <= max_val（原地）。"""
    norm = weight.norm(p=2, dim=tuple(range(1, weight.ndim)), keepdim=True)
    desired = torch.clamp(norm, max=max_val)
    weight.mul_(desired / (norm + 1e-8))


def build_eegnet(nb_classes: int, chans: int, samples: int, model_cfg) -> EEGNet:
    """按配置构造 EEGNet。model_cfg 为 config.EEGNetConfig。"""
    return EEGNet(
        nb_classes=nb_classes,
        chans=chans,
        samples=samples,
        dropout_rate=model_cfg.dropout_rate,
        kern_length=model_cfg.kern_length,
        F1=model_cfg.F1,
        D=model_cfg.D,
        F2=model_cfg.F2,
        norm_rate=model_cfg.norm_rate,
    )
