"""
One-command ingestion pipeline.

    PDF  ->  extracted_pages.json  ->  chunks.json  ->  embeddings.npy  ->  faiss.index

Usage:
    python ingest.py                 # run every stage (skips a stage only with --skip-existing)
    python ingest.py --skip-existing # skip any stage whose output already exists
    python ingest.py --force         # delete all derived artifacts, then rebuild from the PDF
    python ingest.py --only chunk    # run a single stage (pdf | chunk | embed | index)

The individual stages still work standalone (`python -m src.chunker`, etc.); this
script just runs them in the right order, from the right directory, and fails
loudly if a stage does not produce its expected output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from src.config import BASE_DIR, EMBEDDING_MODEL, OUTPUT_JSON, PDF_PATH
from src.logging_config import configure_logging

# Several stage modules use paths relative to the project root ("processed/...").
# Anchor the process there so `python ingest.py` works from any directory.
# TODO(Step 10): remove once all paths come from src.config as absolute paths.
os.chdir(BASE_DIR)

PROCESSED = BASE_DIR / "processed"
EXTRACTED_PAGES = Path(OUTPUT_JSON)
CHUNKS = PROCESSED / "chunks.json"
CHUNK_STATS = PROCESSED / "chunk_stats.json"
EMBEDDINGS = PROCESSED / "embeddings.npy"
METADATA = PROCESSED / "metadata.json"
EMBED_CACHE = PROCESSED / "embedding_cache.json"
FAISS_INDEX = PROCESSED / "faiss.index"
INDEX_STATS = PROCESSED / "index_stats.json"
MANIFEST = PROCESSED / "ingest_manifest.json"

DOCUMENT_NAME = Path(PDF_PATH).name

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE, CHUNK_OVERLAP = 1200, 200

STAGE_ORDER = ["pdf", "chunk", "embed", "index"]


def require(path: Path, stage: str) -> None:
    """Abort if a stage did not produce the file the next stage needs."""
    if not path.exists():
        logger.error(f"Stage '{stage}' finished but {path} was not created. Aborting.")
        sys.exit(1)


# ----------------------------------
# Stages
# ----------------------------------


def stage_pdf() -> None:
    from src.pdf_loader import PDFLoader

    if not Path(PDF_PATH).exists():
        logger.error(f"Source PDF not found: {PDF_PATH}")
        logger.error("Place the document there (see README) or set PDF_PATH in .env.")
        sys.exit(1)

    loader = PDFLoader(PDF_PATH)
    pages = loader.extract_pages()
    loader.save_json(pages, EXTRACTED_PAGES)
    require(EXTRACTED_PAGES, "pdf")


def stage_chunk() -> None:
    from src.chunker import run_chunking

    require(EXTRACTED_PAGES, "pdf")
    run_chunking(
        input_json=str(EXTRACTED_PAGES),
        output_chunks=str(CHUNKS),
        output_stats=str(CHUNK_STATS),
        document_name=DOCUMENT_NAME,
    )
    require(CHUNKS, "chunk")


def stage_embed() -> None:
    from src.embedder import Embedder

    require(CHUNKS, "chunk")
    Embedder(save_every=500).generate_embeddings()
    require(EMBEDDINGS, "embed")
    require(METADATA, "embed")


def stage_index() -> None:
    from src.vector_store import VectorStore

    require(EMBEDDINGS, "embed")
    store = VectorStore()
    store.build_index()
    store.save_index()
    require(FAISS_INDEX, "index")


STAGES = {
    "pdf": (stage_pdf, EXTRACTED_PAGES),
    "chunk": (stage_chunk, CHUNKS),
    "embed": (stage_embed, EMBEDDINGS),
    "index": (stage_index, FAISS_INDEX),
}

DERIVED_ARTIFACTS = [
    EXTRACTED_PAGES,
    CHUNKS,
    CHUNK_STATS,
    EMBEDDINGS,
    METADATA,
    EMBED_CACHE,
    FAISS_INDEX,
    INDEX_STATS,
    MANIFEST,
]


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True
        ).strip()
    except Exception:
        return None


def write_manifest() -> None:
    """Record what settings produced the current index, next to the index."""
    stats = json.loads(INDEX_STATS.read_text()) if INDEX_STATS.exists() else {}
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "source_pdf": str(PDF_PATH),
        "document_name": DOCUMENT_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_count": stats.get("total_vectors"),
        "embedding_dim": stats.get("dimension"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Wrote {MANIFEST.relative_to(BASE_DIR)}")
    for key, value in manifest.items():
        logger.info(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="delete all derived artifacts, then rebuild"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="skip a stage if its output already exists"
    )
    parser.add_argument("--only", choices=STAGE_ORDER, help="run just one stage")
    args = parser.parse_args()

    configure_logging(BASE_DIR / "logs" / "ingest.log")

    if args.force:
        for artifact in DERIVED_ARTIFACTS:
            if artifact.exists():
                artifact.unlink()
                logger.info(f"Removed {artifact.relative_to(BASE_DIR)}")

    stages_to_run = [args.only] if args.only else STAGE_ORDER

    logger.info(f"Project root: {BASE_DIR}")
    logger.info(f"Running stages: {', '.join(stages_to_run)}")
    started = time.perf_counter()

    for name in stages_to_run:
        run_stage, output = STAGES[name]
        if args.skip_existing and output.exists():
            logger.info(f"[{name}] output exists, skipping ({output.relative_to(BASE_DIR)})")
            continue
        logger.info(f"[{name}] starting")
        stage_start = time.perf_counter()
        run_stage()
        logger.info(f"[{name}] done in {time.perf_counter() - stage_start:.1f}s")

    if not args.only and FAISS_INDEX.exists():
        write_manifest()

    logger.info(f"Ingestion complete in {time.perf_counter() - started:.1f}s")
    logger.info(f"Index ready: {FAISS_INDEX}")


if __name__ == "__main__":
    main()
