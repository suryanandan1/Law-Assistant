from loguru import logger

from src.config import (
    PDF_PATH,
    OUTPUT_JSON,
    LOG_FILE
)

from src.pdf_loader import PDFLoader


def setup_logging():

    logger.remove()

    logger.add(
        LOG_FILE,
        rotation="10 MB",
        retention=5,
        level="INFO"
    )

    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO"
    )


def main():

    setup_logging()

    logger.info(
        "===== PDF Extraction Started ====="
    )

    loader = PDFLoader(PDF_PATH)

    pages = loader.extract_pages()

    loader.save_json(
        pages,
        OUTPUT_JSON
    )

    logger.info(
        "===== PDF Extraction Finished ====="
    )


if __name__ == "__main__":
    main()