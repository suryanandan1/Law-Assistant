import pytest


@pytest.fixture
def chunk():
    """Factory for retriever-shaped chunk dicts."""

    def _make(page, text, score=0.6, **extra):
        return {"page": page, "text": text, "score": score, "source": "doc.pdf", **extra}

    return _make
