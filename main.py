import sys
import argparse
import logging
from config import settings
from ingestion import IngestionPipeline
from evaluation import BenchmarkEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.SYSTEM_LOG_FILE)
    ]
)
logger = logging.getLogger("CLI")

def run_ingestion(limit: int):
    eff_limit = None if limit == 0 else limit
    label = "FULL DATASET" if eff_limit is None else f"{eff_limit} Docs"
    
    logger.info(f"Running ETL Pipeline ({label})...")
    pipeline = IngestionPipeline()
    pipeline.run(limit=eff_limit)

def run_benchmark():
    logger.info("Running benchmark...")
    engine = BenchmarkEngine()
    engine.run_ab_test()

def run_pipeline():
    logger.info("Starting full pipeline...")
    run_ingestion(limit=0)
    run_benchmark()
    logger.info("Pipeline complete. Run 'python main.py ui' to explore.")

def run_ui():
    import os
    if not os.path.exists(settings.CHROMA_DB_DIR):
        logger.error("Vector database missing. Run 'python main.py pipeline' first.")
        return
    logger.info("Launching Streamlit dashboard...")
    os.system("streamlit run app.py")

def main():
    parser = argparse.ArgumentParser(description="Legal RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("pipeline", help="Run full ETL + Benchmark")

    parser_ingest = subparsers.add_parser("ingest", help="Run ETL pipeline only")
    parser_ingest.add_argument("--limit", type=int, default=200, help="0 = All Data")

    subparsers.add_parser("benchmark", help="Run A/B test and generate assets")
    subparsers.add_parser("ui", help="Launch Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "pipeline":
        run_pipeline()
    elif args.command == "ingest":
        run_ingestion(args.limit)
    elif args.command == "benchmark":
        run_benchmark()
    elif args.command == "ui":
        run_ui()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()