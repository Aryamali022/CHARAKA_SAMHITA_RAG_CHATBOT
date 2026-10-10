"""Central configuration: paths and environment settings.

Every other module imports paths from here instead of hard-coding them,
so moving a folder means changing one line.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reads .env if it exists; does nothing if it doesn't (e.g. in tests or CI).
# Never overrides variables already set in the real environment.
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STORAGE_DIR = PROJECT_ROOT / "storage"

SOURCE_FILENAME = "charaka-samhita-ocr.txt"
SOURCE_PATH = RAW_DIR / SOURCE_FILENAME

# --- Source provenance (see README and scripts/verify_source.py) ---
SOURCE_ID = "charaka-kaviratna-sourcelibrary"
SOURCE_TITLE = "Charaka-Samhita, English translation by Avinash Chandra Kaviratna (Calcutta, 1890-)"
SOURCE_URL = "https://sourcelibrary.org/book/69ef217fb3e2d3a0927f3054"
SOURCE_LICENSE = "Source Library transcription, CC BY-SA 4.0"
SOURCE_SCAN_ARCHIVE_ID = "BIUSante_47357"  # archive.org identifier of the original scan
SOURCE_PAGE_COUNT = 1988
# SHA-256 of the content from "[Page 1]" onward (header excluded, LF line endings).
SOURCE_CONTENT_SHA256 = "c25eb5c353a774ebf4e02fb88d412e923e3354e2c8627305bbbc6be14cead23a"

# --- Generated files ---
PAGES_PATH = PROCESSED_DIR / "pages.jsonl"
LESSON_PAGES_PATH = PROCESSED_DIR / "lesson_pages.jsonl"
STRUCTURE_PATH = PROCESSED_DIR / "structure.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# --- Search index (Phase 4) ---
MODELS_DIR = STORAGE_DIR / "models"                 # downloaded embedding model
INDEX_DIR = STORAGE_DIR / "index"                   # chunk embeddings + metadata
EVAL_QUESTIONS_PATH = DATA_DIR / "eval" / "retrieval_questions.jsonl"

# --- Answer model (Phase 5) ---
# Any OpenAI-compatible endpoint; the API key is read from NVIDIA_API_KEY in .env.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
UNANSWERABLE_QUESTIONS_PATH = DATA_DIR / "eval" / "unanswerable_questions.jsonl"
ANSWER_EVAL_PATH = PROCESSED_DIR / "answer_eval.jsonl"
