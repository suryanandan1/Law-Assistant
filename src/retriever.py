import json
import time
from pathlib import Path

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        model_name="BAAI/bge-large-en-v1.5",
        top_k=30,
        chunks_file="processed/chunks.json"
    ):

        self.top_k = top_k
        self.chunks_file = Path(chunks_file)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"Retriever device: {self.device}"
        )

        self.embedder = SentenceTransformer(
            model_name,
            device=self.device,
            # The indexed model is cached locally. Avoid a startup-time
            # Hugging Face network check, which also makes the app usable
            # when the machine is offline.
            local_files_only=True
        )

        self.vector_store = VectorStore()

        self.vector_store.load_index()

        logger.info(
            "Loading chunks..."
        )

        with open(
            self.chunks_file,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        self.chunk_lookup = {

            chunk["chunk_id"]: chunk

            for chunk in chunks
        }

        logger.info(
            f"Loaded {len(chunks)} chunks"
        )

    # ----------------------------------
    # Query Embedding
    # ----------------------------------

    def embed_query(
        self,
        question: str
    ):

        return self.embedder.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    # ----------------------------------
    # Retrieval
    # ----------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int = None
    ):

        start_time = time.perf_counter()

        if top_k is None:
            top_k = self.top_k

        logger.info(
            f"Question: {question}"
        )

        query_embedding = (
            self.embed_query(
                question
            )
        )

        vector_results = (
            self.vector_store.search(
                query_embedding,
                top_k=top_k
            )
        )

        results = []

        for result in vector_results:

            chunk_id = result["chunk_id"]

            chunk = self.chunk_lookup.get(
                chunk_id
            )

            if not chunk:
                continue

            results.append(
                {
                    "chunk_id":
                    chunk_id,

                    "page":
                    chunk["page"],

                    "score":
                    round(
                        result["score"],
                        4
                    ),

                    "source":
                    chunk["source"],

                    "text":
                    chunk["text"]
                }
            )

        retrieval_time = (
            time.perf_counter()
            - start_time
        )

        logger.info(
            f"Retrieved "
            f"{len(results)} chunks "
            f"in "
            f"{retrieval_time:.4f}s"
        )

        return {
            "question":
            question,

            "top_k":
            top_k,

            "retrieval_time":
            round(
                retrieval_time,
                4
            ),

            "results":
            results
        }


# ----------------------------------
# Logging
# ----------------------------------

def setup_logging():

    logger.remove()

    logger.add(
        "logs/retrieval.log",
        rotation="10 MB",
        retention=5,
        level="INFO"
    )

    logger.add(
        lambda msg: print(
            msg,
            end=""
        )
    )


# ----------------------------------
# Example Test
# ----------------------------------

def test():

    setup_logging()

    retriever = Retriever(
        top_k=30
    )

    response = retriever.retrieve(
        "What is Article 14?"
    )

    print("\n")

    print(
        f"Retrieval Time: "
        f"{response['retrieval_time']} sec"
    )

    print(
        f"Results: "
        f"{len(response['results'])}"
    )

    for item in response["results"][:3]:

        print("\n")

        print(
            "=" * 80
        )

        print(
            f"Page: "
            f"{item['page']}"
        )

        print(
            f"Score: "
            f"{item['score']}"
        )

        print(
            item["text"][:500]
        )


if __name__ == "__main__":
    test()
