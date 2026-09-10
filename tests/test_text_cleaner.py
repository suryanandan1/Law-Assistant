from src.text_cleaner import clean_text


def test_collapses_internal_whitespace():
    assert clean_text("a\n\n  b\t c") == "a b c"


def test_strips_leading_and_trailing():
    assert clean_text("   hello world   ") == "hello world"


def test_removes_null_bytes():
    assert clean_text("a\x00b") == "a b"


def test_empty_string():
    assert clean_text("") == ""


def test_none():
    assert clean_text(None) == ""


def test_whitespace_only():
    assert clean_text("  \n\t  ") == ""
