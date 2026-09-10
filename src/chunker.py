import hashlib
import re
from pathlib import Path

import orjson
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

# Heading detection for Indian legal text. The PDF extractor collapses whitespace,
# so these run against a single-line page string rather than line-anchored text.
# A fuller version would preserve layout during extraction (see improvement.md
# Step 19) — this still gives each chunk a usable section number and title.
# Title must start Title-case ("Right to...", "Functions of...") so we don't pick
# up run-on ALL-CAPS chapter banners in the collapsed text.
_SECTION_RE = re.compile(
    r"(?:^|\s)(\d{1,4}[A-Z]{0,2})\.\s+([A-Z][a-z][A-Za-z0-9 ,'’()\-/&]{2,88}?)\.(?=\s|�|$)"
)
_ARTICLE_RE = re.compile(r"\bArticle\s+(\d{1,4}[A-Z]?)\b")
_PART_RE = re.compile(r"\b(PART|CHAPTER)\s+([IVXLC]{1,7})\b")


def find_headings(text):
    """Return [(position, kind, identifier, title)] for headings found in a page."""
    headings = []
    for match in _SECTION_RE.finditer(text):
        headings.append((match.start(), "section", match.group(1), match.group(2).strip()))
    for match in _ARTICLE_RE.finditer(text):
        headings.append((match.start(), "article", match.group(1), ""))
    for match in _PART_RE.finditer(text):
        headings.append((match.start(), "part", match.group(2), match.group(1).title()))
    headings.sort(key=lambda item: item[0])
    return headings


def heading_before(offset, headings):
    """The last heading at or before `offset`, or None."""
    current = None
    for position, kind, identifier, title in headings:
        if position > offset:
            break
        current = (kind, identifier, title)
    return current


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
        self, chunk_size: int = 1200, chunk_overlap: int = 200, document_name: str = "document.pdf"
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_name = document_name

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @staticmethod
    def load_json(json_path: str):

        with open(json_path, "rb") as f:
            return orjson.loads(f.read())

    @staticmethod
    def save_json(data, output_path: str):

        output_file = Path(output_path)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    @staticmethod
    def create_chunk_id(page_number: int, chunk_text: str) -> str:
        """
        Stable chunk id.
        """

        hash_value = hashlib.sha256(f"{page_number}_{chunk_text}".encode()).hexdigest()

        return hash_value[:24]

    def chunk_document(self, pages_json_path: str):

        logger.info(f"Loading extracted text: {pages_json_path}")

        pages = self.load_json(pages_json_path)

        total_pages = len(pages)

        logger.info(f"Total pages loaded: {total_pages}")

        chunks = []

        seen_hashes = set()

        stats = {
            "pages_processed": 0,
            "chunks_created": 0,
            "empty_chunks_removed": 0,
            "duplicate_chunks_removed": 0,
            "failed_pages": 0,
        }

        for page_data in tqdm(pages, desc="Chunking Pages"):
            try:
                page_number = page_data.get("page_number")

                text = page_data.get("text", "")

                stats["pages_processed"] += 1

                if not text.strip():
                    continue

                split_chunks = self.splitter.split_text(text)

                page_headings = find_headings(text)
                scan_cursor = 0

                for chunk_text in split_chunks:
                    chunk_text = chunk_text.strip()

                    if not chunk_text:
                        stats["empty_chunks_removed"] += 1

                        continue

                    # Approximate this chunk's offset in the page to find the
                    # nearest preceding heading.
                    found_at = text.find(chunk_text[:40], scan_cursor)
                    if found_at != -1:
                        scan_cursor = found_at
                    heading = heading_before(scan_cursor, page_headings)

                    content_hash = hashlib.md5(chunk_text.encode()).hexdigest()

                    if content_hash in seen_hashes:
                        stats["duplicate_chunks_removed"] += 1

                        continue

                    seen_hashes.add(content_hash)

                    chunk_id = self.create_chunk_id(page_number, chunk_text)

                    chunk_record = {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": chunk_text,
                        "source": self.document_name,
                    }

                    if heading:
                        kind, identifier, title = heading
                        chunk_record["heading_kind"] = kind
                        chunk_record["heading_id"] = identifier
                        if title:
                            chunk_record["heading"] = title[:120]

                    chunks.append(chunk_record)

                    stats["chunks_created"] += 1

            except Exception as e:
                stats["failed_pages"] += 1

                logger.exception(f"Error processing page {page_data.get('page_number')}: {e}")

        logger.info("Chunking completed")

        logger.info(f"Pages Processed: {stats['pages_processed']}")

        logger.info(f"Chunks Created: {stats['chunks_created']}")

        logger.info(f"Empty Removed: {stats['empty_chunks_removed']}")

        logger.info(f"Duplicates Removed: {stats['duplicate_chunks_removed']}")

        logger.info(f"Failed Pages: {stats['failed_pages']}")

        return chunks, stats


def run_chunking(input_json: str, output_chunks: str, output_stats: str, document_name: str):

    logger.info("===== CHUNKING STARTED =====")

    chunker = DocumentChunker(chunk_size=1200, chunk_overlap=200, document_name=document_name)

    chunks, stats = chunker.chunk_document(input_json)

    chunker.save_json(chunks, output_chunks)

    chunker.save_json(stats, output_stats)

    logger.info(f"Chunks saved -> {output_chunks}")

    logger.info(f"Stats saved -> {output_stats}")

    logger.info("===== CHUNKING FINISHED =====")


if __name__ == "__main__":
    run_chunking(
        input_json="processed/extracted_pages.json",
        output_chunks="processed/chunks.json",
        output_stats="processed/chunk_stats.json",
        document_name="document.pdf",
    )
