"""项目路径约定（相对项目根目录）。

- data/    所有采集、验证、ERD 等运行时数据
- assets/  图片与 UI 静态资源
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------- 数据目录 ----------------
DATA_ROOT = "data"

TRAIN_RECORD_ROOT = os.path.join(DATA_ROOT, "acquisition")
TEST_RECORD_ROOT = os.path.join(DATA_ROOT, "test")
VALIDATION_RECORD_ROOT = os.path.join(DATA_ROOT, "validation")
ERDERS_RECORD_ROOT = os.path.join(DATA_ROOT, "erders")
DEBUG_RECORD_ROOT = os.path.join(DATA_ROOT, "debug")
ACQUISITION_BINARY_ROOT = os.path.join(DATA_ROOT, "acquisition_binary")

# ---------------- 资源目录 ----------------
ASSETS_ROOT = "assets"
CUE_IMAGES_DIR = os.path.join(ASSETS_ROOT, "cue_images")
UI_ASSETS_DIR = os.path.join(ASSETS_ROOT, "ui")

# 提示图目录约定（相对 cue_images/）：
#   finger/  — 单指 ME：thumb.jpg, index.jpg, middle.jpg, pinky.jpg
#   hand/    — 手形 ME：fist.jpg, relax.jpg, palm.jpg
#   common/  — 共用：fixation_rest.jpg（试次间注视点）
CUE_IMAGE_SUBDIRS = ("finger", "hand", "common")

# 提示图候选后缀（按优先级）
CUE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def project_root_from(caller_file: str) -> str:
    """从调用方 __file__ 推断项目根目录。"""
    return os.path.dirname(os.path.abspath(caller_file))


def cue_images_root(project_root: str) -> str:
    """采集提示图根目录: ``<project_root>/assets/cue_images``。"""
    return os.path.join(project_root, CUE_IMAGES_DIR)


def resolve_cue_image(project_root: str, subdir: str, condition: str) -> Optional[str]:
    """解析范式提示图路径：``assets/cue_images/<subdir>/<condition>.{jpg,jpeg,png}``。

    Returns:
        存在的文件绝对路径；找不到则 ``None``。
    """
    if not condition or not subdir:
        return None
    base = cue_images_root(project_root)
    stem = os.path.join(base, subdir, condition)
    for ext in CUE_IMAGE_EXTENSIONS:
        path = stem + ext
        if os.path.isfile(path):
            return path
    return None
