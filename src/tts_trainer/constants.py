from .languages import DEFAULT_TRAINING_LANGUAGES


# Backward-compatible default set. The complete supported set now comes from
# the configuration-driven language registry.
LANGUAGES = DEFAULT_TRAINING_LANGUAGES
LANG_TO_ID = {language: index for index, language in enumerate(LANGUAGES)}
