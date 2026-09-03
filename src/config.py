from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / os.getenv(
    "PDF_PATH",
    "data/document.pdf"
)

OUTPUT_JSON = BASE_DIR / os.getenv(
    "OUTPUT_JSON",
    "processed/extracted_pages.json"
)

LOG_FILE = BASE_DIR / os.getenv(
    "LOG_FILE",
    "logs/extraction.log"
)

BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", 100)
)

MAX_WORKERS = int(
    os.getenv("MAX_WORKERS", 4)
)