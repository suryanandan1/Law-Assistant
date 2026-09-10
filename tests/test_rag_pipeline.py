import pytest

from src.rag_pipeline import RAGPipeline
from src.retriever import _env_flag


@pytest.mark.parametrize(
    "mode,question,history,expected",
    [
        ("auto", "What is Article 21?", [], False),
        ("auto", "It was mentioned earlier", [], True),  # reference-led opener
        ("auto", "क्या है", [], True),  # non-ASCII
        ("auto", "What is Article 21?", [{"question": "x", "answer": "y"}], True),  # has history
        ("always", "What is Article 21?", [], True),
        ("never", "It was mentioned earlier", [], False),
    ],
)
def test_needs_query_rewrite(monkeypatch, mode, question, history, expected):
    monkeypatch.setenv("QUERY_REWRITE", mode)
    assert RAGPipeline._needs_query_rewrite(question, history) is expected


def test_env_flag_truthy_values(monkeypatch):
    for value in ["1", "true", "TRUE", "yes", "on", " On "]:
        monkeypatch.setenv("FLAG", value)
        assert _env_flag("FLAG") is True


def test_env_flag_falsy_values(monkeypatch):
    for value in ["0", "false", "no", "off", ""]:
        monkeypatch.setenv("FLAG", value)
        assert _env_flag("FLAG") is False


def test_env_flag_default(monkeypatch):
    monkeypatch.delenv("FLAG", raising=False)
    assert _env_flag("FLAG") is False
    assert _env_flag("FLAG", default=True) is True
