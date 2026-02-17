import os
import csv
import time
import json
import logging
import statistics
import warnings
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import settings
from retrieval import HybridRetriever, SearchStrategy, SearchResult
from utils import timer_decorator, CsvLogger

logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger("Evaluation")
GOLDEN_SET = [
    ("termination for convenience notice period", "termination"),
    ("governing law jurisdiction california", "law"),
    ("force majeure events list pandemic", "force"),
    ("confidentiality obligations duration", "confidential"),
    ("indemnification for third party claims", "indemnif"),
    ("limitation of liability cap", "liab"),
    ("payment terms net 30 days", "pay"),
    ("intellectual property ownership rights", "property"),
    ("assignment of agreement consent", "assign"),
    ("entire agreement merger clause", "entire"),
    ("insurance coverage requirements", "insur"),
    ("audit rights records retention", "audit"),
    ("survival of representations", "surviv"),
    ("severability of invalid provisions", "sever"),
    ("waiver of jury trial", "jury"),
    ("non-solicitation of employees", "solicit"),
    ("data privacy gdpr compliance", "data"),
    ("dispute resolution arbitration", "arbitra"),
    ("change of control provisions", "control"),
    ("exclusivity non-compete clause", "exclus")
]

class MetricsCalculator:
    @staticmethod
    def calculate_mrr(ranks: List[int]) -> float:
        if not ranks: return 0.0
        return float(np.mean([1.0 / r if r > 0 else 0.0 for r in ranks]))

    @staticmethod
    def calculate_hit_rate(hits: List[int]) -> float:
        if not hits: return 0.0
        return float(np.mean(hits))

    @staticmethod
    def calculate_ndcg(ranks: List[int], k: int = 5) -> float:
        scores = []
        for r in ranks:
            if r == 0 or r > k:
                scores.append(0.0)
            else:
                dcg = 1.0 / np.log2(r + 1)
                scores.append(dcg)
        return float(np.mean(scores))

class AssetGenerator:
    def __init__(self):
        self.assets_dir = settings.ASSETS_DIR
        plt.style.use('ggplot')
        sns.set_context("notebook", font_scale=1.1)
        self.colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    def _save_plot(self, filename: str):
        path = os.path.join(self.assets_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Generated Asset: {filename}")

    def draw_architecture_diagram(self):
        logger.info("Drawing System Architecture...")
        G = nx.DiGraph()

        nodes = {
            "Raw": "BillSum\nDataset",
            "ETL": "Ingestion\nPipeline",
            "Vector": "ChromaDB\n(Dense)",
            "Keyword": "BM25\n(Sparse)",
            "Logic": "Hybrid\nRetriever",
            "Rerank": "Cross-Encoder\n(Neural)",
            "App": "Streamlit\nUI",
            "User": "End User"
        }
        
        G.add_nodes_from(nodes.values())
        
        edges = [
            ("Raw", "ETL"),
            ("ETL", "Vector"),
            ("ETL", "Keyword"),
            ("Vector", "Logic"),
            ("Keyword", "Logic"),
            ("Logic", "Rerank"),
            ("Rerank", "App"),
            ("App", "User")
        ]
        
        real_edges = [(nodes[u], nodes[v]) for u, v in edges]
        G.add_edges_from(real_edges)

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42, k=0.9)
        
        nx.draw_networkx_nodes(G, pos, node_size=3500, node_color="#2c3e50", alpha=0.9, node_shape='s')
        nx.draw_networkx_labels(G, pos, font_color="white", font_size=9, font_weight="bold")
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="#7f8c8d", arrows=True, arrowsize=20)
        
        plt.title("Hybrid RAG Architecture", fontsize=14, pad=20)
        plt.axis("off")
        self._save_plot("architecture.png")

    def plot_latency_distribution(self, latencies: List[float]):
        if not latencies: return
        
        plt.figure(figsize=(8, 5))
        sns.histplot(latencies, kde=True, color=self.colors[2], bins=15, line_kws={'linewidth': 2})
        
        p95 = np.percentile(latencies, 95)
        plt.axvline(p95, color='red', linestyle='--', label=f'P95: {p95:.2f}s')
        
        plt.title("End-to-End Query Latency Distribution")
        plt.xlabel("Latency (Seconds)")
        plt.ylabel("Frequency")
        plt.legend()
        self._save_plot("latency_dist.png")

    def plot_ab_test_results(self, metrics: pd.DataFrame):
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x="Metric", y="Score", hue="Strategy", data=metrics, palette="viridis")
        
        # Add labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
            
        plt.title("Benchmark: Dense Baseline vs. Hybrid RAG")
        plt.ylim(0, 1.1)
        self._save_plot("benchmark_comparison.png")

class QueryLogAnalyzer:
    def __init__(self):
        self.log_path = settings.QUERY_LOG_FILE

    def analyze(self) -> Dict[str, Any]:
        if not os.path.exists(self.log_path):
            logger.warning("No query logs found.")
            return {}

        try:
            df = pd.read_csv(self.log_path)
            if df.empty: return {}

            stats = {
                "total_queries": len(df),
                "avg_latency": df["total_latency_ms"].mean(),
                "p95_latency": df["total_latency_ms"].quantile(0.95),
                "strategies_used": df["strategy"].value_counts().to_dict()
            }
            logger.info(f"Analyzed {len(df)} query log records.")
            return stats
        except Exception as e:
            logger.error(f"Failed to parse logs: {e}")
            return {}

class BenchmarkEngine:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.viz = AssetGenerator()
        self.test_cases = GOLDEN_SET
        self.results_log = CsvLogger(
            settings.BENCHMARK_REPORT_FILE,
            headers=["timestamp", "strategy", "query", "rank", "reciprocal_rank", "hit", "latency"]
        )

    @timer_decorator
    def evaluate_strategy(self, strategy: SearchStrategy) -> Dict[str, float]:
        logger.info(f"--- Benchmarking Strategy: {strategy.value} ---")
        
        ranks = []
        hits = []
        latencies = []
        
        for case in tqdm(self.test_cases, desc=f"Testing {strategy.name}"):
            query = case[0]
            target = case[1]
            
            start = time.time()
            results = self.retriever.search(query, strategy=strategy)
            duration = time.time() - start
            
            rank = 0
            hit = 0
            
            for i, res in enumerate(results[:5]):
                if target.lower() in res.content.lower():
                    rank = i + 1
                    hit = 1
                    break
            
            rr = 1.0 / rank if rank > 0 else 0.0
            
            ranks.append(rank)
            hits.append(hit)
            latencies.append(duration)
            
            self.results_log.log_row([
                datetime.now().isoformat(),
                strategy.value,
                query,
                rank,
                f"{rr:.4f}",
                hit,
                f"{duration:.4f}"
            ])

        return {
            "MRR": MetricsCalculator.calculate_mrr(ranks),
            "HitRate@5": MetricsCalculator.calculate_hit_rate(hits),
            "NDCG@5": MetricsCalculator.calculate_ndcg(ranks, k=5),
            "AvgLatency": float(np.mean(latencies))
        }

    def run_ab_test(self):
        logger.info("Starting A/B test...")
        
        metrics_base = self.evaluate_strategy(SearchStrategy.DENSE_ONLY)
        metrics_hybrid = self.evaluate_strategy(SearchStrategy.HYBRID_RRF)
        
        print(f"\nBenchmark Results (N={len(self.test_cases)} Queries)")
        print("-" * 60)
        print(f"{'Metric':<15} | {'Baseline (Vector)':<20} | {'Hybrid':<20}")
        print("-" * 60)
        for key in metrics_base.keys():
            val_b = metrics_base[key]
            val_h = metrics_hybrid[key]
            
            diff = ((val_h - val_b) / val_b) * 100 if val_b > 0 else 0
            indicator = "+" if val_h >= val_b else "-"
            if "Latency" in key: indicator = "-" if val_h > val_b else "+"
            
            print(f"{key:<15} | {val_b:.4f}               | {val_h:.4f} ({diff:+.1f}%) {indicator}")
        print("="*60 + "\n")
        
        plot_df = pd.DataFrame([
            {"Strategy": "Baseline", "Metric": "MRR", "Score": metrics_base["MRR"]},
            {"Strategy": "Hybrid", "Metric": "MRR", "Score": metrics_hybrid["MRR"]},
            {"Strategy": "Baseline", "Metric": "HitRate@5", "Score": metrics_base["HitRate@5"]},
            {"Strategy": "Hybrid", "Metric": "HitRate@5", "Score": metrics_hybrid["HitRate@5"]},
        ])
        
        self.viz.plot_ab_test_results(plot_df)
        self.viz.draw_architecture_diagram()
        
        analyzer = QueryLogAnalyzer()
        stats = analyzer.analyze()
        if stats:
            print("Query Log Summary:")
            print(f"Total Queries: {stats['total_queries']}")
            print(f"P95 Latency: {stats['p95_latency']:.4f} ms")

if __name__ == "__main__":
    engine = BenchmarkEngine()
    engine.run_ab_test()