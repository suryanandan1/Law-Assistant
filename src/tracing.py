"""Append one JSON line per query to logs/traces.jsonl when TRACE is enabled.

Gives you an after-the-fact record of what was retrieved for a bad answer:
question, rewritten query, chunk ids + scores, cited pages, latencies, answer.

    TRACE=1 streamlit run app.py
    jq . logs/traces.jsonl        # inspect
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

_TRACE_FILE = Path(os.getenv("TRACE_FILE", "logs/traces.jsonl"))


def tracing_enabled() -> bool:
    return os.getenv("TRACE", "").strip().lower() in {"1", "true", "yes", "on"}


def trace_turn(
    *,
    question: str,
    search_query: str,
    language: str,
    sources: list[dict],
    pages: list,
    retrieval_time: float,
    generation_time: float,
    answer: str,
    error: str | None = None,
) -> None:
    if not tracing_enabled():
        return

    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": question,
        "search_query": search_query,
        "rewritten": search_query != question,
        "language": language,
        "pages": pages,
        "chunks": [
            {
                "chunk_id": source.get("chunk_id"),
                "page": source.get("page"),
                "score": source.get("score"),
                "rerank_score": source.get("rerank_score"),
            }
            for source in sources
        ],
        "retrieval_time": round(retrieval_time, 4),
        "generation_time": round(generation_time, 4),
        "answer": answer,
        "error": error,
    }
    try:
        _TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _TRACE_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as write_error:  # tracing must never break a request
        logger.warning(f"Trace write failed: {write_error}")
