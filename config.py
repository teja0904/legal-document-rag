import os
import sys
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class AppSettings(BaseSettings):
    APP_NAME: str = "Legal RAG System"
    VERSION: str = "1.0.0"
    ENV: str = Field("development", env="APP_ENV")

    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    INDICES_DIR: Path = DATA_DIR / "indices"
    LOGS_DIR: Path = BASE_DIR / "logs"
    ASSETS_DIR: Path = BASE_DIR / "assets"

    CHROMA_DB_DIR: Path = INDICES_DIR / "chroma_db"
    BM25_INDEX_FILE: Path = INDICES_DIR / "bm25_sparse_index.pkl"
    
    SYSTEM_LOG_FILE: Path = LOGS_DIR / "system_events.log"
    INGESTION_AUDIT_FILE: Path = LOGS_DIR / "ingestion_audit.csv"
    QUERY_LOG_FILE: Path = LOGS_DIR / "query_log.csv"
    BENCHMARK_REPORT_FILE: Path = LOGS_DIR / "benchmark_results.csv"

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"  # Will be dynamically updated

    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_MAX_LENGTH: int = 512

    DATASET_NAME: str = "billsum" 
    CHUNK_SIZE: int = Field(512, description="Token count per chunk")
    CHUNK_OVERLAP: int = Field(128, description="Overlap to preserve semantic context")
    MIN_DOC_LENGTH: int = 100  # Ignore contracts shorter than this
    
    REGEX_PATTERNS: Dict[str, str] = {
        "page_numbers": r"Page \d+\s+of\s+\d+",
        "signatures": r"_{3,}",
        "broken_hyphens": r"(\w+)-\s*\n\s*(\w+)",
        "redactions": r"\[\*\*\*\]",
        "whitespace": r"\s+"
    }

    TOP_K_RETRIEVAL: int = 60
    TOP_K_RERANK: int = 5
    RRF_K_CONSTANT: int = 60

    @validator("EMBEDDING_DEVICE", pre=True, always=True)
    def detect_hardware(cls, v):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = AppSettings()

CRITICAL_DIRS = [
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.INDICES_DIR,
    settings.LOGS_DIR,
    settings.ASSETS_DIR,
    settings.CHROMA_DB_DIR.parent
]

for directory in CRITICAL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if logger.hasHandlers():
        return logger

    file_handler = logging.FileHandler(settings.SYSTEM_LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

system_logger = setup_logger("System")
system_logger.info(f"Device: {settings.EMBEDDING_DEVICE}")
system_logger.info(f"Base dir: {settings.BASE_DIR}")