import time
import csv
import functools
import logging
from typing import Callable, Any, List
from pathlib import Path

class AppError(Exception):
    pass

class IngestionError(AppError):
    pass

class RetrievalError(AppError):
    pass

class ModelLoadError(AppError):
    pass

def retry(max_retries: int = 3, delay: int = 2, backoff: float = 2.0):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("Utils")
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    logger.warning(
                        f"Function '{func.__name__}' failed (Attempt {retries}/{max_retries}). "
                        f"Error: {e}. Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # If we reach here, we failed
            logger.error(f"Function '{func.__name__}' failed permanently after {max_retries} retries.")
            raise IngestionError(f"Operation failed: {func.__name__}")
        return wrapper
    return decorator

def timer_decorator(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logging.getLogger("Profiler").debug(
            f"'{func.__name__}' completed in {duration:.4f} seconds"
        )
        return result
    return wrapper

class CsvLogger:
    def __init__(self, filepath: Path, headers: List[str]):
        self.filepath = filepath
        self.headers = headers
        self._initialize_file()

    def _initialize_file(self):
        if not self.filepath.exists():
            with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_row(self, row_data: List[Any]):
        try:
            with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        except Exception as e:
            logging.getLogger("Utils").error(f"Failed to write to CSV log: {e}")

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    return text.replace('\x00', '').replace('\n', '\\n').replace('\r', '')