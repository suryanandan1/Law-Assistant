import hashlib
from pathlib import Path

import orjson
from tqdm import tqdm
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Enterprise-grade chunking pipeline
    for large PDF documents.

    Input:
        extracted_pages.json

    Output:
        chunks.json
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        document_name: str = "document.pdf"
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_name = document_name

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    @staticmethod
    def load_json(json_path: str):

        with open(json_path, "rb") as f:
            return orjson.loads(f.read())

    @staticmethod
    def save_json(data, output_path: str):

        output_file = Path(output_path)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_file, "wb") as f:
            f.write(
                orjson.dumps(
                    data,
                    option=orjson.OPT_INDENT_2
                )
            )

    @staticmethod
    def create_chunk_id(
        page_number: int,
        chunk_text: str
    ) -> str:
        """
        Stable chunk id.
        """

        hash_value = hashlib.sha256(
            f"{page_number}_{chunk_text}".encode()
        ).hexdigest()

        return hash_value[:24]

    def chunk_document(
        self,
        pages_json_path: str
    ):

        logger.info(
            f"Loading extracted text: "
            f"{pages_json_path}"
        )

        pages = self.load_json(
            pages_json_path
        )

        total_pages = len(pages)

        logger.info(
            f"Total pages loaded: "
            f"{total_pages}"
        )

        chunks = []

        seen_hashes = set()

        stats = {
            "pages_processed": 0,
            "chunks_created": 0,
            "empty_chunks_removed": 0,
            "duplicate_chunks_removed": 0,
            "failed_pages": 0
        }

        for page_data in tqdm(
            pages,
            desc="Chunking Pages"
        ):

            try:

                page_number = page_data.get(
                    "page_number"
                )

                text = page_data.get(
                    "text",
                    ""
                )

                stats["pages_processed"] += 1

                if not text.strip():
                    continue

                split_chunks = (
                    self.splitter.split_text(text)
                )

                for chunk_text in split_chunks:

                    chunk_text = (
                        chunk_text.strip()
                    )

                    if not chunk_text:

                        stats[
                            "empty_chunks_removed"
                        ] += 1

                        continue

                    content_hash = hashlib.md5(
                        chunk_text.encode()
                    ).hexdigest()

                    if content_hash in seen_hashes:

                        stats[
                            "duplicate_chunks_removed"
                        ] += 1

                        continue

                    seen_hashes.add(
                        content_hash
                    )

                    chunk_id = (
                        self.create_chunk_id(
                            page_number,
                            chunk_text
                        )
                    )

                    chunk_record = {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": chunk_text,
                        "source": self.document_name
                    }

                    chunks.append(
                        chunk_record
                    )

                    stats[
                        "chunks_created"
                    ] += 1

            except Exception as e:

                stats["failed_pages"] += 1

                logger.exception(
                    f"Error processing "
                    f"page "
                    f"{page_data.get('page_number')}"
                    f": {e}"
                )

        logger.info(
            "Chunking completed"
        )

        logger.info(
            f"Pages Processed: "
            f"{stats['pages_processed']}"
        )

        logger.info(
            f"Chunks Created: "
            f"{stats['chunks_created']}"
        )

        logger.info(
            f"Empty Removed: "
            f"{stats['empty_chunks_removed']}"
        )

        logger.info(
            f"Duplicates Removed: "
            f"{stats['duplicate_chunks_removed']}"
        )

        logger.info(
            f"Failed Pages: "
            f"{stats['failed_pages']}"
        )

        return chunks, stats


def run_chunking(
    input_json: str,
    output_chunks: str,
    output_stats: str,
    document_name: str
):

    logger.info(
        "===== CHUNKING STARTED ====="
    )

    chunker = DocumentChunker(
        chunk_size=1200,
        chunk_overlap=200,
        document_name=document_name
    )

    chunks, stats = (
        chunker.chunk_document(
            input_json
        )
    )

    chunker.save_json(
        chunks,
        output_chunks
    )

    chunker.save_json(
        stats,
        output_stats
    )

    logger.info(
        f"Chunks saved -> "
        f"{output_chunks}"
    )

    logger.info(
        f"Stats saved -> "
        f"{output_stats}"
    )

    logger.info(
        "===== CHUNKING FINISHED ====="
    )


if __name__ == "__main__":

    run_chunking(
        input_json=
        "processed/extracted_pages.json",

        output_chunks=
        "processed/chunks.json",

        output_stats=
        "processed/chunk_stats.json",

        document_name=
        "document.pdf"
    )
    
