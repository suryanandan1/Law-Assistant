"""Wire-up test for RAGPipeline with fake collaborators (no models, index or Vertex)."""

import pytest

from src.rag_pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, top_k=None):
        return {"results": self._results, "top_k": top_k, "retrieval_time": 0.0}


class FakeLLM:
    model = "fake-model"

    def __init__(self, answer="Equality before the law (page 5)."):
        self.answer = answer
        self.calls = []

    def build_search_query(self, question, history):
        self.calls.append(("rewrite", question))
        return question

    def generate_answer(self, prompt):
        self.calls.append(("generate", prompt))
        return self.answer


def make_pipeline(results, answer="ok (page 5)"):
    return RAGPipeline(top_k=5, retriever=FakeRetriever(results), llm=FakeLLM(answer))


@pytest.fixture(autouse=True)
def _stable_rewrite_mode(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE", "auto")


def test_ask_returns_a_complete_result_dict():
    results = [
        {"page": 5, "text": "Equality before law.", "score": 0.7},
        {"page": 6, "text": "Equal protection of laws.", "score": 0.65},
    ]
    pipe = make_pipeline(results, answer="Equality is guaranteed (page 5).")

    out = pipe.ask("What does Article 14 say?")

    assert "error" not in out
    assert out["answer"] == "Equality is guaranteed (page 5)."
    assert out["pages"] == [5, 6]
    assert out["sources"] == results
    assert out["retrieval_time"] >= 0
    assert out["generation_time"] >= 0
    assert out["total_time"] >= 0


def test_ask_short_circuits_when_nothing_retrieved():
    pipe = make_pipeline([], answer="must not be used")

    out = pipe.ask("What is the capital of France?")

    assert out["answer"] == RAGPipeline.NO_INFO_ANSWER
    assert out["pages"] == []
    assert pipe.llm.calls == []  # generation never attempted


def test_retrieve_and_build_returns_a_prompt():
    results = [{"page": 21, "text": "Protection of life and personal liberty.", "score": 0.8}]
    pipe = make_pipeline(results)

    prep = pipe.retrieve_and_build("What is Article 21?", language="English")

    assert prep["answer"] is None
    assert "<document_context>" in prep["prompt"]
    assert "Protection of life" in prep["prompt"]
    assert prep["pages"] == [21]


def test_ask_reports_generation_failure_as_error():
    class BoomLLM(FakeLLM):
        def generate_answer(self, prompt):
            raise RuntimeError("vertex exploded")

    pipe = RAGPipeline(
        top_k=5,
        retriever=FakeRetriever([{"page": 1, "text": "text", "score": 0.9}]),
        llm=BoomLLM(),
    )

    out = pipe.ask("anything")

    assert out["error"] == "vertex exploded"
    assert out["pages"] == []
