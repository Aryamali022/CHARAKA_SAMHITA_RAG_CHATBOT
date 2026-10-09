"""Central configuration: paths and environment settings.

Every other module imports paths from here instead of hard-coding them,
so moving a folder means changing one line.
"""
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
