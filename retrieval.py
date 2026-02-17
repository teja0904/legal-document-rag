import os
import time
import pickle
import logging
import threading
import warnings
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from pydantic import BaseModel, Field

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from config import settings
from utils import timer_decorator, CsvLogger, RetrievalError

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger("Retrieval")


class SearchStrategy(str, Enum):
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    HYBRID_RRF = "hybrid_rrf"

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int
    source_strategy: str
    doc_id: str

class QueryLogRecord(BaseModel):
    query_hash: str
    strategy: str
    total_latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    result_count: int
    timestamp: float = Field(default_factory=time.time)

# --- Logic Components ---

class QueryLogWriter:
    def __init__(self, filepath: str):
        self.logger = CsvLogger(
            filepath, 
            headers=["timestamp", "query_hash", "strategy", "total_ms", "retrieval_ms", "rerank_ms", "count"]
        )

    def log(self, record: QueryLogRecord):
        self.logger.log_row([
            record.timestamp,
            record.query_hash,
            record.strategy,
            f"{record.total_latency_ms:.2f}",
            f"{record.retrieval_latency_ms:.2f}",
            f"{record.rerank_latency_ms:.2f}",
            record.result_count
        ])

class QueryPreprocessor:
    @staticmethod
    def clean(query: str) -> str:
        if not query:
            raise ValueError("Query cannot be empty")
        # Keep only alphanumeric and basic punctuation
        return "".join([c for c in query if c.isalnum() or c in " ?'"]).strip()

class RankFusion:
    @staticmethod
    def reciprocal_rank_fusion(
        results_lists: List[List[Dict]], 
        k: int = 60,
        weights: Optional[List[float]] = None
    ) -> List[Dict]:
        if not weights:
            weights = [1.0] * len(results_lists)
            
        fused_scores = {}
        
        for r_list, weight in zip(results_lists, weights):
            for rank, item in enumerate(r_list):
                content = item["doc"].page_content
                if content not in fused_scores:
                    fused_scores[content] = {
                        "doc": item["doc"],
                        "score": 0.0,
                        "sources": set()
                    }
                fused_scores[content]["score"] += weight * (1.0 / (rank + k))
                fused_scores[content]["sources"].add(item["type"])
        
        # Sort by Fused Score
        final_results = [
            {"doc": v["doc"], "score": v["score"], "source": "+".join(v["sources"])}
            for v in fused_scores.values()
        ]
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results

class HybridRetriever:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(HybridRetriever, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if self._initialized: return
        
        with self._lock:
            if self._initialized: return
            self._load_heavy_resources()
            self._initialized = True

    def _load_heavy_resources(self):
        logger.info("Loading models...")
        
        self.query_logger = QueryLogWriter(settings.QUERY_LOG_FILE)
        
        # 2. Dense Embeddings
        logger.info(f"Loading Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': settings.EMBEDDING_DEVICE}
        )
        
        # 3. Vector DB
        if not settings.CHROMA_DB_DIR.exists():
            raise RetrievalError("Vector DB not found. Please run ingestion first.")
        
        self.vector_db = Chroma(
            persist_directory=str(settings.CHROMA_DB_DIR),
            embedding_function=self.embeddings
        )
        
        # 4. Sparse Index
        if not settings.BM25_INDEX_FILE.exists():
            raise RetrievalError("BM25 Index missing. Please run ingestion first.")
            
        logger.info("Loading BM25 Index into Memory...")
        with open(settings.BM25_INDEX_FILE, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["model"]
            self.bm25_chunks = data["chunks"]
            
        # 5. Re-ranker
        logger.info(f"Loading Cross-Encoder: {settings.RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME, max_length=settings.RERANKER_MAX_LENGTH)
        
        logger.info("Retriever ready.")

    def _dense_search(self, query: str, k: int) -> Tuple[List[Dict], float]:
        start = time.time()
        # similarity_search_with_score returns L2 distance by default in Chroma
        # We wrap it to match our generic dict structure
        docs_scores = self.vector_db.similarity_search_with_score(query, k=k)
        results = [{"doc": d, "score": s, "type": "dense"} for d, s in docs_scores]
        return results, (time.time() - start) * 1000

    def _sparse_search(self, query: str, k: int) -> Tuple[List[Dict], float]:
        start = time.time()
        tokens = query.lower().split()
        top_docs = self.bm25.get_top_n(tokens, self.bm25_chunks, n=k)
        results = [{"doc": d, "score": 0.0, "type": "sparse"} for d in top_docs]
        return results, (time.time() - start) * 1000

    def _rerank_candidates(self, query: str, candidates: List[Dict], top_k: int) -> Tuple[List[SearchResult], float]:
        start = time.time()
        if not candidates: return [], 0.0
        
        # Deduplicate candidates (same doc might appear in dense and sparse)
        unique_map = {c["doc"].page_content: c for c in candidates}
        unique_candidates = list(unique_map.values())
        
        # Prepare pairs for Cross-Encoder
        pairs = [[query, c["doc"].page_content] for c in unique_candidates]
        
        # Inference
        scores = self.reranker.predict(pairs)
        
        # Format Results
        final_results = []
        for i, score in enumerate(scores):
            item = unique_candidates[i]
            meta = item["doc"].metadata
            final_results.append(SearchResult(
                content=item["doc"].page_content,
                metadata=meta,
                score=float(score),
                rank=0, # Assigned later
                source_strategy="hybrid",
                doc_id=meta.get("doc_id", "unknown")
            ))
            
        # Sort & Cut
        final_results.sort(key=lambda x: x.score, reverse=True)
        final_results = final_results[:top_k]
        
        # Assign final rank
        for i, res in enumerate(final_results): res.rank = i + 1
            
        return final_results, (time.time() - start) * 1000

    def search(self, 
               query: str, 
               strategy: SearchStrategy = SearchStrategy.HYBRID_RRF) -> List[SearchResult]:
        overall_start = time.time()
        clean_query = QueryPreprocessor.clean(query)
        
        retrieval_ms = 0.0
        candidates = []

        try:
            # --- Phase 1: Retrieval ---
            if strategy in [SearchStrategy.DENSE_ONLY, SearchStrategy.HYBRID_RRF]:
                dense_res, t_ms = self._dense_search(clean_query, k=settings.TOP_K_RETRIEVAL)
                if strategy == SearchStrategy.DENSE_ONLY: candidates = dense_res
                retrieval_ms += t_ms

            if strategy in [SearchStrategy.SPARSE_ONLY, SearchStrategy.HYBRID_RRF]:
                sparse_res, t_ms = self._sparse_search(clean_query, k=settings.TOP_K_RETRIEVAL)
                if strategy == SearchStrategy.SPARSE_ONLY: candidates = sparse_res
                retrieval_ms += t_ms

            # --- Phase 2: Fusion ---
            if strategy == SearchStrategy.HYBRID_RRF:
                candidates = RankFusion.reciprocal_rank_fusion(
                    [dense_res, sparse_res], 
                    k=settings.RRF_K_CONSTANT
                )

            # --- Phase 3: Re-ranking ---
            # Optimization: Only re-rank the top 50 fused candidates
            rerank_pool = candidates[:50]
            final_results, rerank_ms = self._rerank_candidates(clean_query, rerank_pool, top_k=settings.TOP_K_RERANK)
            
            # Log query metrics
            total_duration = (time.time() - overall_start) * 1000
            
            log_record = QueryLogRecord(
                query_hash=str(hash(clean_query)),
                strategy=strategy.value,
                total_latency_ms=total_duration,
                retrieval_latency_ms=retrieval_ms,
                rerank_latency_ms=rerank_ms,
                result_count=len(final_results)
            )
            self.query_logger.log(log_record)
            
            return final_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RetrievalError(f"Search failed: {e}")

    # --- Convenience Wrappers for A/B Testing ---
    
    def search_baseline(self, query: str) -> List[SearchResult]:
        clean_q = QueryPreprocessor.clean(query)
        res, _ = self._dense_search(clean_q, k=settings.TOP_K_RETRIEVAL)
        # Format as SearchResult
        formatted = []
        for i, r in enumerate(res[:settings.TOP_K_RERANK]):
            formatted.append(SearchResult(
                content=r["doc"].page_content,
                metadata=r["doc"].metadata,
                score=r["score"],
                rank=i+1,
                source_strategy="baseline",
                doc_id=r["doc"].metadata.get("doc_id", "u")
            ))
        return formatted

    def search_hybrid(self, query: str) -> List[SearchResult]:
        return self.search(query, strategy=SearchStrategy.HYBRID_RRF)