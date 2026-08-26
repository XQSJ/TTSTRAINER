"""向后兼容的旧常量出口。 / Backward-compatible legacy constants."""
from .languages import DEFAULT_TRAINING_LANGUAGES


# 保持旧导入路径可用；完整语言集合现由配置驱动的注册表提供。
# Backward-compatible default set. The complete supported set now comes from
# the configuration-driven language registry.
LANGUAGES = DEFAULT_TRAINING_LANGUAGES
LANG_TO_ID = {language: index for index, language in enumerate(LANGUAGES)}
