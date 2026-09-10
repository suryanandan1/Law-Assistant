from src.context_builder import ContextBuilder


def test_first_chunk_over_budget_yields_nothing(chunk):
    cb = ContextBuilder(max_context_chars=40)
    out = cb.build_context("q", [chunk(1, "x" * 100), chunk(2, "y" * 100)])
    assert out["pages"] == []


def test_includes_chunks_until_budget_hit(chunk):
    cb = ContextBuilder(max_context_chars=200)
    chunks = [chunk(1, "a" * 50), chunk(2, "b" * 50), chunk(3, "c" * 5000)]
    out = cb.build_context("q", chunks)
    assert out["pages"] == [1, 2]


def test_exact_duplicate_is_dropped(chunk):
    out = ContextBuilder().build_context(
        "q", [chunk(1, "identical text"), chunk(2, "identical text")]
    )
    assert out["pages"] == [1]


def test_near_duplicate_is_dropped(chunk):
    cb = ContextBuilder(near_dup_threshold=0.8)
    a = "the quick brown fox jumps over the lazy dog"
    b = "the quick brown fox jumps over the lazy dog again"
    out = cb.build_context("q", [chunk(1, a), chunk(2, b)])
    assert out["pages"] == [1]


def test_caps_chunks_per_page(chunk):
    cb = ContextBuilder(max_per_page=2)
    chunks = [chunk(5, f"a distinct passage numbered {i}") for i in range(5)]
    out = cb.build_context("q", chunks)
    assert out["pages"] == [5]
    assert out["prompt"].count("[PAGE 5]") == 2


def test_rerank_score_wins_over_bi_encoder_score(chunk):
    cb = ContextBuilder(max_context_chars=15)  # room for exactly one chunk
    chunks = [
        chunk(1, "aaaa", score=0.9, rerank_score=1.0),
        chunk(2, "bbbb", score=0.1, rerank_score=9.0),
    ]
    out = cb.build_context("q", chunks)
    assert out["pages"] == [2]


def test_language_and_history_are_plumbed_and_truncated(chunk):
    history = [{"question": "Q" * 900, "answer": "A" * 2000}]
    out = ContextBuilder().build_context(
        "what is it", [chunk(3, "some text")], history=history, language="Hindi"
    )
    prompt = out["prompt"]
    assert "Respond ENTIRELY in Hindi" in prompt
    assert "CONVERSATION HISTORY" in prompt
    assert "Q" * 500 in prompt
    assert "Q" * 501 not in prompt


def test_prompt_has_no_rigid_output_scaffold(chunk):
    prompt = ContextBuilder().build_context("q", [chunk(1, "text")])["prompt"]
    assert "OUTPUT FORMAT" not in prompt
    assert "Supporting Pages:" not in prompt
    assert "<document_context>" in prompt
