import json
import os
import time
from pathlib import Path

from loguru import logger
from sentence_transformers import CrossEncoder

from src.config import EMBEDDING_MODEL
from src.embedding_model import load_embedding_model, pick_device
from src.logging_config import configure_logging
from src.vector_store import VectorStore


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _check_index_model(model_name: str) -> None:
    """Warn loudly if the index was built with a different embedding model."""
    manifest = Path("processed/ingest_manifest.json")
    if not manifest.exists():
        return
    try:
        built_with = json.loads(manifest.read_text()).get("embedding_model")
    except Exception:
        return
    if built_with and built_with != model_name:
        logger.warning(
            f"Index was built with '{built_with}' but the retriever is configured "
            f"for '{model_name}'. Retrieval will be wrong until you run "
            f"`python ingest.py --force`."
        )


class Retriever:
    def __init__(
        self,
        model_name=None,
        top_k=30,
        chunks_file="processed/chunks.json",
        min_score=None,
        rerank=None,
        rerank_model=None,
        rerank_candidates=None,
    ):

        self.top_k = top_k
        self.model_name = model_name or EMBEDDING_MODEL
        self.chunks_file = Path(chunks_file)
        _check_index_model(self.model_name)

        # Drop results below this cosine similarity so off-topic questions return
        # nothing (the pipeline then answers "not in the document") instead of
        # forcing the LLM to work from irrelevant chunks. Observed with
        # bge-large-en-v1.5: on-topic chunks score ~0.58-0.79, off-topic ~0.36-0.49.
        # Override per-deployment with RETRIEVAL_MIN_SCORE.
        self.min_score = (
            min_score if min_score is not None else float(os.getenv("RETRIEVAL_MIN_SCORE", "0.5"))
        )

        self.device = pick_device()
        self.embedder = load_embedding_model(self.model_name)

        self.vector_store = VectorStore()

        self.vector_store.load_index()

        logger.info("Loading chunks...")

        with open(self.chunks_file, encoding="utf-8") as f:
            chunks = json.load(f)

        self.chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

        logger.info(f"Loaded {len(chunks)} chunks")

        # ------------------------------
        # Optional cross-encoder reranker
        # ------------------------------
        # A bi-encoder (FAISS) is fast but coarse. Re-scoring the top candidates
        # with a cross-encoder markedly improves precision for legal QA.
        # Default model reranks ~30 candidates in ~1s on CPU; bge-reranker-large
        # is far more accurate but ~30x slower without a GPU.
        self.rerank_enabled = rerank if rerank is not None else _env_flag("RERANK", default=True)
        self.rerank_model_name = rerank_model or os.getenv(
            "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.rerank_candidates = int(
            rerank_candidates
            if rerank_candidates is not None
            else os.getenv("RERANK_CANDIDATES", "30")
        )
        self.reranker = None

        if self.rerank_enabled:
            try:
                self.reranker = CrossEncoder(
                    self.rerank_model_name, device=self.device, max_length=512
                )
                logger.info(f"Reranker loaded: {self.rerank_model_name}")
            except Exception as error:
                logger.warning(
                    f"Reranker '{self.rerank_model_name}' unavailable "
                    f"({error}); continuing without reranking."
                )
                self.rerank_enabled = False

    # ----------------------------------
    # Query Embedding
    # ----------------------------------

    def embed_query(self, question: str):

        return self.embedder.encode(question, normalize_embeddings=True, convert_to_numpy=True)

    # ----------------------------------
    # Retrieval
    # ----------------------------------

    def retrieve(self, question: str, top_k: int = None):

        start_time = time.perf_counter()

        if top_k is None:
            top_k = self.top_k

        # When reranking, pull a wider candidate set for the cross-encoder to sort.
        fetch_k = max(top_k, self.rerank_candidates) if self.rerank_enabled else top_k

        logger.info(f"Question: {question}")

        query_embedding = self.embed_query(question)

        vector_results = self.vector_store.search(query_embedding, top_k=fetch_k)

        results = []

        for result in vector_results:
            chunk_id = result["chunk_id"]

            chunk = self.chunk_lookup.get(chunk_id)

            if not chunk:
                continue

            results.append(
                {
                    "chunk_id": chunk_id,
                    "page": chunk["page"],
                    "score": round(result["score"], 4),
                    "source": chunk["source"],
                    "text": chunk["text"],
                }
            )

        if self.min_score > 0:
            kept = [item for item in results if item["score"] >= self.min_score]

            dropped = len(results) - len(kept)

            if dropped:
                logger.info(f"Dropped {dropped} chunk(s) below min_score={self.min_score}")

            results = kept

        # ------------------------------
        # Rerank the survivors, then keep the best top_k
        # ------------------------------
        if self.rerank_enabled and self.reranker is not None and results:
            pairs = [(question, item["text"]) for item in results]

            rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

            for item, rerank_score in zip(results, rerank_scores):
                item["rerank_score"] = round(float(rerank_score), 4)

            results.sort(key=lambda item: item["rerank_score"], reverse=True)

        results = results[:top_k]

        retrieval_time = time.perf_counter() - start_time

        logger.info(f"Retrieved {len(results)} chunks in {retrieval_time:.4f}s")

        return {
            "question": question,
            "top_k": top_k,
            "retrieval_time": round(retrieval_time, 4),
            "results": results,
        }


# ----------------------------------
# Example Test
# ----------------------------------


def test():

    configure_logging("logs/retrieval.log")

    retriever = Retriever(top_k=30)

    response = retriever.retrieve("What is Article 14?")

    print("\n")

    print(f"Retrieval Time: {response['retrieval_time']} sec")

    print(f"Results: {len(response['results'])}")

    for item in response["results"][:3]:
        print("\n")

        print("=" * 80)

        print(f"Page: {item['page']}")

        print(f"Score: {item['score']}")

        print(item["text"][:500])


if __name__ == "__main__":
    test()
