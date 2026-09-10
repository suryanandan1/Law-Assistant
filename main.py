"""Stage 1 only: extract text from the source PDF.

For the full pipeline (extract -> chunk -> embed -> index) use `python ingest.py`.
"""

from loguru import logger

from src.config import LOG_FILE, OUTPUT_JSON, PDF_PATH
from src.logging_config import configure_logging
from src.pdf_loader import PDFLoader


def main():
    configure_logging(LOG_FILE)
    logger.info("===== PDF Extraction Started =====")

    loader = PDFLoader(PDF_PATH)
    pages = loader.extract_pages()
    loader.save_json(pages, OUTPUT_JSON)

    logger.info("===== PDF Extraction Finished =====")


if __name__ == "__main__":
    main()
