"""BCI-Desk 版本与产品元信息。"""

__version__ = "0.1.0"
APP_NAME = "BCI-Desk"
APP_NAME_CN = "BCI-Desk 脑电运动执行工作台"
APP_TAGLINE = "Open-source EEG BCI workbench for hand ME / MI on OpenBCI Cyton + Daisy"
APP_ORG = "yichuanyao"
GITHUB_URL = "https://github.com/yichuanyao/BCI-Desk"
LICENSE_SPDX = "MIT"

# GitHub Topics / 检索标签（README 与 About 共用）
APP_HASHTAGS = (
    "bci",
    "eeg",
    "openbci",
    "brainflow",
    "motor-execution",
    "motor-imagery",
    "hand-me",
    "hand-mi",
    "erd-ers",
    "eegnet",
    "bci-desk",
    "python",
    "pyqt",
)


def hashtags_line(prefix: str = "#") -> str:
    return " ".join(f"{prefix}{t}" for t in APP_HASHTAGS)
