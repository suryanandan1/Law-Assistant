import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / os.getenv("PDF_PATH", "data/document.pdf")

OUTPUT_JSON = BASE_DIR / os.getenv("OUTPUT_JSON", "processed/extracted_pages.json")

LOG_FILE = BASE_DIR / os.getenv("LOG_FILE", "logs/extraction.log")

# The embedder and the retriever MUST use the same model, so it lives here.
# Smaller = much faster on CPU:
#   BAAI/bge-large-en-v1.5  (1024-d, best quality, slowest)
#   BAAI/bge-base-en-v1.5   (768-d,  ~3x faster, minor quality drop)
#   BAAI/bge-small-en-v1.5  (384-d,  ~10x faster, noticeable drop)
# Changing this requires a full re-embed: `python ingest.py --force`.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
