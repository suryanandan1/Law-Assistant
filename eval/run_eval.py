"""Retrieval + answer evaluation over eval/gold.jsonl.

    python eval/run_eval.py                 # retrieval metrics only (offline, no GCP)
    python eval/run_eval.py --llm           # also generate answers and score keyword recall
    python eval/run_eval.py --k 5 --top-k 8 --limit 5

Metrics
    hit@k        fraction of answerable questions where an expected anchor phrase
                 appears in one of the top-k retrieved chunks
    MRR          mean reciprocal rank of the first chunk containing an anchor
    no-answer    fraction of off-topic questions that retrieved nothing
                 (i.e. the relevance gate fired)
    kw-recall    (--llm only) fraction of expected keywords present in the answer

Results are written to eval/results/<timestamp>_<git-sha>.json.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

from src.retriever import Retriever  # noqa: E402

GOLD = ROOT / "eval" / "gold.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower()).strip()


def first_anchor_rank(anchors: list[str], chunks: list[dict]) -> int | None:
    """1-based rank of the first chunk whose text contains any anchor, else None."""
    norm_anchors = [normalize(a) for a in anchors if a.strip()]
    for rank, chunk in enumerate(chunks, 1):
        text = normalize(chunk["text"])
        if any(a in text for a in norm_anchors):
            return rank
    return None


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def load_gold(limit: int | None) -> list[dict]:
    items = [
        json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return items[:limit] if limit else items


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--k", type=int, default=5, help="k for hit@k / MRR cutoff (default 5)")
    parser.add_argument(
        "--top-k", type=int, default=8, help="chunks the retriever returns (default 8)"
    )
    parser.add_argument(
        "--llm", action="store_true", help="also generate answers and score keyword recall"
    )
    parser.add_argument("--limit", type=int, help="evaluate only the first N gold items")
    args = parser.parse_args()

    gold = load_gold(args.limit)
    retriever = Retriever(top_k=args.top_k)

    generator = None
    if args.llm:
        from src.context_builder import ContextBuilder
        from src.llm_service import GeminiService

        context_builder = ContextBuilder()
        llm = GeminiService()

        def generator(question, chunks):  # noqa: F811
            prompt = context_builder.build_context(question, chunks)["prompt"]
            return llm.generate_answer(prompt)

    rows = []
    print(f"\n{'id':<16} {'hit@k':>6} {'rank':>5} {'kw-recall':>9}  question")
    print("-" * 92)

    for item in gold:
        chunks = retriever.retrieve(item["question"], top_k=args.top_k)["results"]
        top = chunks[: args.k]
        off_topic = item.get("expect_no_answer", False)

        row = {"id": item["id"], "off_topic": off_topic, "retrieved": len(chunks)}

        if off_topic:
            row["no_answer_ok"] = len(chunks) == 0
            hit_display, rank_display = ("-", "-")
        else:
            rank = first_anchor_rank(item["anchors"], top)
            row["hit"] = rank is not None
            row["rr"] = 1.0 / rank if rank else 0.0
            hit_display = "yes" if row["hit"] else "NO"
            rank_display = str(rank) if rank else "-"

        kw_display = "-"
        if generator and not off_topic:
            answer = generator(item["question"], chunks)
            found = [k for k in item["keywords"] if normalize(k) in normalize(answer)]
            row["kw_recall"] = len(found) / len(item["keywords"]) if item["keywords"] else None
            row["answer"] = answer
            kw_display = f"{row['kw_recall']:.2f}" if row["kw_recall"] is not None else "-"

        rows.append(row)
        print(
            f"{item['id']:<16} {hit_display:>6} {rank_display:>5} {kw_display:>9}  {item['question'][:44]}"
        )

    answerable = [r for r in rows if not r["off_topic"]]
    offtopic = [r for r in rows if r["off_topic"]]

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "k": args.k,
        "top_k": args.top_k,
        "n_answerable": len(answerable),
        "n_offtopic": len(offtopic),
        f"hit@{args.k}": round(sum(r["hit"] for r in answerable) / len(answerable), 3)
        if answerable
        else None,
        "mrr": round(sum(r["rr"] for r in answerable) / len(answerable), 3) if answerable else None,
        "no_answer_accuracy": (
            round(sum(r["no_answer_ok"] for r in offtopic) / len(offtopic), 3) if offtopic else None
        ),
    }
    kw_scored = [r["kw_recall"] for r in answerable if r.get("kw_recall") is not None]
    if kw_scored:
        summary["kw_recall"] = round(sum(kw_scored) / len(kw_scored), 3)

    print("-" * 92)
    for key, value in summary.items():
        if key not in {"generated_at", "git_sha"}:
            print(f"{key:>20}: {value}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{stamp}_{git_sha() or 'nogit'}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
