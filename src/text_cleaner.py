import re


def clean_text(text: str) -> str:
    """Basic PDF cleanup."""
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip() if text else ""
