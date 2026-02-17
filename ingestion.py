import os
import re
import csv
import time
import json
import pickle
import shutil
import hashlib
import logging
import threading
from typing import List, Dict, Optional, Generator, Any, Tuple
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from config import settings
from utils import retry, timer_decorator, CsvLogger, IngestionError, sanitize_text

logger = logging.getLogger("Ingestion")

class AuditLogger:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.headers = [
            "timestamp", "doc_id", "source_title", "original_char_len", 
            "clean_char_len", "chunk_count", "content_hash", "status", "error_msg"
        ]
        self._init_log()

    def _init_log(self):
        if not self.filepath.exists():
            with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, doc_id: str, title: str, orig_len: int, clean_len: int, 
            chunks: int, hash_val: str, status: str, error: str = ""):
        try:
            with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    doc_id,
                    sanitize_text(title)[:50], 
                    orig_len,
                    clean_len,
                    chunks,
                    hash_val,
                    status,
                    error
                ])
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")


class LegalTextNormalizer:
    def __init__(self):
        self.patterns = settings.REGEX_PATTERNS

    def normalize(self, text: str) -> str:
        if not text: return ""
        text = re.sub(self.patterns["page_numbers"], "", text, flags=re.IGNORECASE)
        text = re.sub(self.patterns["redactions"], "[REDACTED]", text)
        text = re.sub(self.patterns["broken_hyphens"], r"\1\2", text)
        text = re.sub(self.patterns["signatures"], "___", text)
        text = re.sub(self.patterns["whitespace"], " ", text)
        return text.strip()

class MetadataEnricher:
    @staticmethod
    def enrich(doc_id: str, title: str, content: str) -> Dict[str, Any]:
        return {
            "doc_id": doc_id,
            "title": sanitize_text(title),
            "ingest_timestamp": datetime.now().timestamp(),
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
            "char_length": len(content),
            "source_system": "HF_BillSum"
        }


class QualityAssurance:
    @staticmethod
    def validate_document(text: str, min_len: int) -> Tuple[bool, str]:
        if not text:
            return False, "EMPTY_CONTENT"
        if len(text) < min_len:
            return False, "TOO_SHORT"
        return True, "OK"


class VectorIndexBuilder:
    def __init__(self, persist_dir: Path):
        self.persist_dir = str(persist_dir)
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': settings.EMBEDDING_DEVICE}
        )

    @timer_decorator
    def build(self, chunks: List[Document]):
        logger.info(f"Building Vector Index at {self.persist_dir}...")
        
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            time.sleep(0.5) 
            
        vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn
        )
        
        batch_size = 5000
        total = len(chunks)
        
        for i in tqdm(range(0, total, batch_size), desc="Indexing Vectors"):
            batch = chunks[i : i + batch_size]
            vector_db.add_documents(batch)
            
        try:
            vector_db.persist() 
        except AttributeError:
            pass 
            
        logger.info(f"Vector Index Complete. Total Chunks: {total}")

class SparseIndexBuilder:
    def __init__(self, output_path: Path):
        self.output_path = output_path

    @timer_decorator
    def build(self, chunks: List[Document]):
        logger.info("Building Sparse (BM25) Index...")
        tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        payload = {
            "model": bm25,
            "chunks": chunks,
            "version": settings.VERSION,
            "build_time": datetime.now().isoformat()
        }
        
        with open(self.output_path, "wb") as f:
            pickle.dump(payload, f)
            
        logger.info(f"Sparse Index Saved to {self.output_path}")


class DataIngestor:
    def __init__(self, limit: Optional[int] = None):
        self.dataset_name = settings.DATASET_NAME
        self.limit = limit

    @retry(max_retries=3, delay=5)
    def fetch_stream(self) -> Generator[Dict, None, None]:
        logger.info(f"Connecting to HuggingFace: {self.dataset_name}")
        
        # Standard load - no special flags needed for BillSum
        dataset = load_dataset(self.dataset_name, split="train")
        
        logger.info(f"Connection Established. Total Records: {len(dataset)}")
        
        count = 0
        for record in dataset:
            if self.limit and count >= self.limit: break
            yield record
            count += 1

class IngestionPipeline:
    def __init__(self):
        self.audit = AuditLogger(settings.INGESTION_AUDIT_FILE)
        self.normalizer = LegalTextNormalizer()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ";", ".", " ", ""]
        )

    def run(self, limit: Optional[int] = None):
        logger.info("Starting ETL pipeline...")
        start_time = time.time()
        
        ingestor = DataIngestor(limit=limit)
        valid_docs = []
        seen_hashes = set()
        
        # --- Phase 1: Ingest & Transform ---
        data_stream = ingestor.fetch_stream()
        
        for i, record in enumerate(tqdm(data_stream, desc="Processing Docs", total=limit)):
            doc_id = f"DOC_{i:05d}"
            
            # BillSum uses 'text' and 'title' keys.
            raw_text = record.get("text") or record.get("context") or ""
            title = record.get("title", f"Legal_Doc_{i}")
            
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)

            # QA Check 1: Length
            is_valid, reason = QualityAssurance.validate_document(raw_text, settings.MIN_DOC_LENGTH)
            if not is_valid:
                self.audit.log(doc_id, title, len(raw_text), 0, 0, "N/A", "SKIPPED", reason)
                continue
                
            try:
                # Normalize
                clean_text = self.normalizer.normalize(raw_text)
                
                # Enrich Metadata
                meta = MetadataEnricher.enrich(doc_id, title, clean_text)
                
                # QA Check 2: Deduplication
                if meta['content_hash'] in seen_hashes:
                    self.audit.log(doc_id, title, len(raw_text), len(clean_text), 0, meta['content_hash'], "SKIPPED", "DUPLICATE")
                    continue
                
                seen_hashes.add(meta['content_hash'])
                
                # Create Object
                doc = Document(page_content=clean_text, metadata=meta)
                valid_docs.append(doc)
                
                self.audit.log(doc_id, title, len(raw_text), len(clean_text), 0, meta['content_hash'], "STAGED", "")
                
            except Exception as e:
                logger.error(f"Processing Error {doc_id}: {e}")
                self.audit.log(doc_id, title, len(raw_text), 0, 0, "N/A", "ERROR", str(e))

        if not valid_docs:
            logger.error("0 Valid Documents found. Please inspect logs.")
            raise IngestionError("Pipeline produced 0 valid documents.")

        logger.info(f"Deduplication Complete. Unique Docs: {len(valid_docs)}")

        # --- Phase 2: Segmentation ---
        logger.info("Segmenting Documents into Semantic Chunks...")
        all_chunks = self.splitter.split_documents(valid_docs)
        logger.info(f"Segmentation Complete. {len(valid_docs)} Docs -> {len(all_chunks)} Chunks.")

        # --- Phase 3: Loading (Indexing) ---
        
        # Dense
        v_builder = VectorIndexBuilder(settings.CHROMA_DB_DIR)
        v_builder.build(all_chunks)
        
        # Sparse
        s_builder = SparseIndexBuilder(settings.BM25_INDEX_FILE)
        s_builder.build(all_chunks)
        
        logger.info(f"ETL complete. Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    IngestionPipeline().run(limit=200)