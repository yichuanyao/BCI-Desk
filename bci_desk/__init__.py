"""BCI-Desk — Open-source EEG BCI workbench (hand ME / MI).

本地 GUI 与采集/预处理可复用模块（无 torch）。
"""

from .__about__ import APP_HASHTAGS, APP_NAME, __version__, hashtags_line

__all__ = ["APP_NAME", "APP_HASHTAGS", "__version__", "config", "hashtags_line"]
