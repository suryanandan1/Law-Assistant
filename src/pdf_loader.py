from pathlib import Path

import fitz
import orjson
from loguru import logger
from tqdm import tqdm

from src.text_cleaner import clean_text


class PDFLoader:
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = str(pdf_path)

    def extract_pages(self):
        logger.info(f"Opening PDF: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        logger.info(f"Total pages: {total_pages}")
        extracted = []

        for page_num in tqdm(range(total_pages), desc="Extracting PDF"):
            try:
                page = doc.load_page(page_num)
                text = page.get_text("text")
                text = clean_text(text)
                extracted.append({"page_number": page_num + 1, "text": text})

            except Exception as e:
                logger.error(f"Failed page {page_num + 1}: {e}")

                extracted.append({"page_number": page_num + 1, "text": "", "error": str(e)})

        doc.close()

        logger.info("Extraction completed")
        return extracted

    @staticmethod
    def save_json(data, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

        logger.info(f"Saved JSON -> {output_path}")
