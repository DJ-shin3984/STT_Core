"""언어 코드 → 모델별 이름."""
LANG_NAME = {"ko": "Korean", "en": "English", "zh": "Chinese", "ja": "Japanese",
             "yue": "Cantonese", "de": "German", "fr": "French", "es": "Spanish"}


def lang_name(code: str) -> str | None:
    """'ko' → 'Korean'. '' 또는 'auto' → None(자동 감지)."""
    if not code or code == "auto":
        return None
    return LANG_NAME.get(code, code)
