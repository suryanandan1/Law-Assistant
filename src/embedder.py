import json
import time
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.config import EMBED_BATCH_SIZE, EMBEDDING_MODEL
from src.embedding_model import load_embedding_model, pick_device
from src.logging_config import configure_logging


class Embedder:
    def __init__(
        self,
        chunks_file="processed/chunks.json",
        embeddings_file="processed/embeddings.npy",
        metadata_file="processed/metadata.json",
        cache_file="processed/embedding_cache.json",
        model_name=None,
        batch_size=None,
        save_every=500,
    ):

        self.chunks_file = Path(chunks_file)

        self.embeddings_file = Path(embeddings_file)

        self.metadata_file = Path(metadata_file)

        self.cache_file = Path(cache_file)

        self.model_name = model_name or EMBEDDING_MODEL
        self.batch_size = batch_size or EMBED_BATCH_SIZE
        self.save_every = save_every

        self.device = pick_device()
        self.model = load_embedding_model(self.model_name)
        logger.info(f"Embedding with {self.model_name} | batch_size={self.batch_size}")

    def load_chunks(self):

        with open(self.chunks_file, encoding="utf-8") as f:
            return json.load(f)

    def load_cache(self):

        if self.cache_file.exists():
            with open(self.cache_file, encoding="utf-8") as f:
                return json.load(f)

        return {}

    def save_cache(self, cache):

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    def save_checkpoint(self, embeddings, metadata, cache):

        np.save(self.embeddings_file, np.array(embeddings, dtype=np.float32))

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.save_cache(cache)

        logger.info(f"Checkpoint saved ({len(metadata)} chunks)")

    def generate_embeddings(self):

        chunks = self.load_chunks()

        logger.info(f"Loaded {len(chunks)} chunks")

        cache = self.load_cache()

        embeddings = []
        metadata = []

        if self.embeddings_file.exists():
            embeddings = np.load(self.embeddings_file).tolist()

        if self.metadata_file.exists():
            with open(self.metadata_file, encoding="utf-8") as f:
                metadata = json.load(f)

        pending = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]

            if chunk_id not in cache:
                pending.append(chunk)

        logger.info(f"Remaining chunks: {len(pending)}")

        processed = 0

        for i in tqdm(range(0, len(pending), self.batch_size), desc="Embedding"):
            batch = pending[i : i + self.batch_size]

            texts = [x["text"] for x in batch]

            try:
                batch_embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

            except Exception as e:
                logger.error(f"Batch failed: {e}")

                time.sleep(2)

                continue

            for chunk, emb in zip(batch, batch_embeddings):
                embeddings.append(emb.tolist())

                metadata.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "page": chunk["page"],
                        "source": chunk["source"],
                    }
                )

                cache[chunk["chunk_id"]] = True

            processed += len(batch)

            if processed >= self.save_every:
                self.save_checkpoint(embeddings, metadata, cache)

                processed = 0

        self.save_checkpoint(embeddings, metadata, cache)

        logger.info("Embedding complete")


def main():

    configure_logging("logs/embedding.log")

    embedder = Embedder(save_every=500)

    embedder.generate_embeddings()


if __name__ == "__main__":
    main()
