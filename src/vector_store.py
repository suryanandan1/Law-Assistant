import json
from pathlib import Path

import faiss
import numpy as np

from loguru import logger


class VectorStore:

    def __init__(
        self,
        embeddings_file="processed/embeddings.npy",
        metadata_file="processed/metadata.json",
        index_file="processed/faiss.index",
        stats_file="processed/index_stats.json"
    ):

        self.embeddings_file = Path(
            embeddings_file
        )

        self.metadata_file = Path(
            metadata_file
        )

        self.index_file = Path(
            index_file
        )

        self.stats_file = Path(
            stats_file
        )

        self.index = None
        self.metadata = None

    # ----------------------------------
    # Build Index
    # ----------------------------------

    def build_index(self):

        logger.info(
            "Loading embeddings..."
        )

        embeddings = np.load(
            self.embeddings_file
        )

        embeddings = embeddings.astype(
            np.float32
        )

        if len(embeddings.shape) != 2:

            raise ValueError(
                "Embeddings must be 2D"
            )

        total_vectors = embeddings.shape[0]
        dimension = embeddings.shape[1]

        logger.info(
            f"Vectors: {total_vectors}"
        )

        logger.info(
            f"Dimension: {dimension}"
        )

        # normalize for cosine similarity

        faiss.normalize_L2(
            embeddings
        )

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        logger.info(
            f"Indexed vectors: "
            f"{self.index.ntotal}"
        )

        with open(
            self.metadata_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(f)

        if len(self.metadata) != total_vectors:

            raise ValueError(
                "Metadata count mismatch"
            )

        stats = {

            "total_vectors":
            int(total_vectors),

            "dimension":
            int(dimension),

            "index_type":
            "IndexFlatIP",

            "similarity":
            "cosine",

            "metadata_records":
            len(self.metadata)
        }

        with open(
            self.stats_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                stats,
                f,
                indent=2
            )

        logger.info(
            "Index statistics saved"
        )

    # ----------------------------------
    # Save Index
    # ----------------------------------

    def save_index(self):

        if self.index is None:

            raise ValueError(
                "Build index first"
            )

        faiss.write_index(
            self.index,
            str(self.index_file)
        )

        logger.info(
            f"Index saved -> "
            f"{self.index_file}"
        )

    # ----------------------------------
    # Load Existing Index
    # ----------------------------------

    def load_index(self):

        if not self.index_file.exists():

            raise FileNotFoundError(
                self.index_file
            )

        self.index = faiss.read_index(
            str(self.index_file)
        )

        with open(
            self.metadata_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(f)

        logger.info(
            f"Loaded index with "
            f"{self.index.ntotal} vectors"
        )

    # ----------------------------------
    # Search
    # ----------------------------------

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        if self.index is None:

            raise ValueError(
                "Index not loaded"
            )

        query_embedding = np.array(
            query_embedding,
            dtype=np.float32
        )

        query_embedding = (
            query_embedding.reshape(
                1,
                -1
            )
        )

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:
                continue

            results.append(
                {
                    "score":
                    float(score),

                    "chunk_id":
                    self.metadata[idx][
                        "chunk_id"
                    ],

                    "page":
                    self.metadata[idx][
                        "page"
                    ],

                    "source":
                    self.metadata[idx][
                        "source"
                    ]
                }
            )

        return results


# ----------------------------------
# Logging
# ----------------------------------

def setup_logging():

    logger.remove()

    logger.add(
        "logs/faiss.log",
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
# Build Index Runner
# ----------------------------------

def build():

    setup_logging()

    store = VectorStore()

    store.build_index()

    store.save_index()

    logger.info(
        "FAISS build completed"
    )


if __name__ == "__main__":
    build()