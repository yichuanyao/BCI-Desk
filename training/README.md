# BCI-Desk 训练程序（服务器端）

独立、自包含的 **运动执行（ME）** EEG 训练程序。可整体拷贝到服务器运行，
**不依赖**本地 `BCI-Desk` 包。模型架构与预处理流水线对齐参考工程
`Finger-BCI-Decoding`（EEGNet-8,2），用 **PyTorch** 重写。

## 支持的范式

| 范式 id | 类别 (event_type) | 说明 |
|---------|-------------------|------|
| `finger_2` | THUMB / PINKY | 手指二分类（默认） |
| `finger_3` | THUMB / INDEX / PINKY | 手指三分类 |
| `finger_4` | THUMB / INDEX / MIDDLE / PINKY | 手指四分类 |
| `hand_2class` | FIST / RELAX | 手形二分类（握拳 4s / 放松 3s） |
| `hand_3class` | FIST / RELAX / PALM | 手形三分类 |

**不同范式应分别训练、分别保存模型**，不要混训。

类别顺序即标签索引 `y=0..N-1`，定义在 `config.py` 的 `PARADIGM_CLASSES`。

## 数据契约

吃本地采集定稿格式的带标签 CSV：

- 列：`timestamp, event_type, EEG Channel 1..16`
- 同名旁车 `<csv>.meta.json`（推荐）：

```json
{
  "unit": "microvolt",
  "sample_rate": 500,
  "paradigm_id": "hand_2class",
  "classes": ["FIST", "RELAX"]
}
```

切试次规则：**连续相同 `event_type` 视为一个试次（trial）**。
`BREAK`、`0`、`GAP` 等不在 `classes` 内的标签会被自动跳过。

单位处理：`microvolt` 不缩放；`raw_count` 自动 ×0.0223 换算成 µV。
无 meta 时默认按 `raw_count` 处理。

## 本地落盘目录结构

```
data/acquisition/
  finger_2/
    FINGER2_20260706_143052/
      FINGER2_20260706_143052_train_block01_runs01-04_me.csv
      FINGER2_20260706_143052_train_block01_runs01-04_me.meta.json
      session.json
  hand_2class/
    HAND2_20260706_150000/
      HAND2_20260706_150000_train_block01_runs01-04_me.csv
      ...

data/test/
  hand_2class/
    HAND2_test_xxx/
      ...
```

上传服务器时，指向具体 **session 目录** 或 **范式子目录** 即可。

## 预处理流水线

每个试次：`CAR` → `1s 滑窗切片增强(step 默认 64)` → `resample 降采样到 100Hz`
→ `4–40Hz 带通` → `按时间 zscore`。每个滑窗成为一个训练样本 `(16, 100)`。

> 训练/验证在**试次层面**先划分再切片，避免同一试次的滑窗泄漏到两边。

`max_trial_s=5.0` 对手指 ME（5s）和手形 ME（握拳 4s / 放松 3s）均适用。

## 安装

```bash
pip install -r requirements.txt
```

## 训练

类别解析优先级：`--classes` > `--paradigm` > 数据 meta 中的 `classes` > 默认 `finger_2`。

```bash
# 手指二分类（meta 含 classes 时可省略 --paradigm）
python train.py --data path/to/data/acquisition/finger_2/FINGER2_xxx \
  --paradigm finger_2 --out models/eegnet_finger2.pth --epochs 300

# 手形二分类（推荐显式指定范式）
python train.py --data path/to/data/acquisition/hand_2class/HAND2_xxx \
  --paradigm hand_2class --out models/eegnet_hand2.pth --epochs 300

# 或手动指定类别
python train.py --data path/to/HAND2_xxx --classes FIST,RELAX --out models/eegnet_hand2.pth

# 整个 finger_2 范式目录（递归加载所有 session CSV）
python train.py --data path/to/data/acquisition/finger_2 --paradigm finger_2 \
  --out models/eegnet_finger2.pth
```

常用参数：`--epochs --batch-size --lr --seed --cpu`（其余默认见 `config.py`）。

## 四指 → 手指二分类数据集转换

若采集的是 **finger_4** 四指数据，但只想训 THUMB/PINKY，可用批处理脚本缩小上传体积：

```bash
python make_binary_dataset.py \
  --input ../data/acquisition/finger_4 \
  --output ../data/acquisition_binary/finger_4
```

- 保留 THUMB / PINKY 试次；INDEX / MIDDLE 段压缩为单行哨兵 `GAP`（加载时跳过，保留试次边界）。
- **手形二分类已是二分类，不需要此脚本。**

转换后训练：

```bash
python train.py --data path/to/acquisition_binary/finger_4 --paradigm finger_2 \
  --out models/eegnet_finger2.pth
```

## 离线评估（测试集准确率）

```bash
# 手指模型
python evaluate.py --model models/eegnet_finger2.pth \
  --data ../data/test/finger_2/FINGER2_test_xxx/

# 手形模型
python evaluate.py --model models/eegnet_hand2.pth \
  --data ../data/test/hand_2class/HAND2_test_xxx/
```

评估端从模型旁车 `.meta.json` 读取 `classes`，无需再指定范式。

本地 GUI「采集范式」页右侧测试采集完成后，将测试目录上传服务器用 `evaluate.py` 评估。

## 产出（模型包）

- `models/eegnet_*.pth` —— PyTorch 权重（`state_dict`）
- `models/eegnet_*.meta.json` —— 元数据：类别 / 通道数 / 输入点数 /
  预处理参数 / EEGNet 超参 / 期望输入单位 / 验证精度。

推理端按 `meta.json` 用**完全一致**的预处理与架构重建模型即可加载权重。

## 文件结构

```
training/
  config.py              # 超参 / 范式类别表 / 预处理参数 / 单位增益
  data_loader.py         # CSV(+meta) → 试次 (C,T) + 标签；meta 类别推断
  preprocessing.py       # CAR / 滑窗 / resample / 带通 / zscore
  eegnet.py              # PyTorch EEGNet（含 max_norm 约束）
  train.py               # 训练主脚本（--paradigm / --classes）
  evaluate.py            # 离线试次级评估
  make_binary_dataset.py # 四指 → THUMB/PINKY 批处理（仅 finger_4 场景）
  requirements.txt
  README.md
```
