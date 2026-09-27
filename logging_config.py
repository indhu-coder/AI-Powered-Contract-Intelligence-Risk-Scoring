"""ContractIQ Centralized Logging Configuration.

Provides rotating file logging and structured console logging with request/task ID
correlation across FastAPI endpoints, Celery workers, and background threads.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "contractiq.log"


class RequestIdFilter(logging.Filter):
    """Ensures all log records have a request_id attribute for correlation."""
    def __init__(self, request_id: Optional[str] = None):
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = self.request_id or "-"
        return True


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Configures root and application loggers with console and rotating file output."""
    target_file = log_file or LOG_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return logging.getLogger("contractiq")

    root_logger.setLevel(level)

    log_format = "%(asctime)s | %(levelname)-7s | [%(name)s] [req:%(request_id)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    root_logger.addHandler(console_handler)

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        str(target_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RequestIdFilter())
    root_logger.addHandler(file_handler)

    # Silence overly chatty external loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return logging.getLogger("contractiq")


def get_logger(name: str) -> logging.Logger:
    """Retrieves a logger configured under the contractiq namespace."""
    setup_logging()
    return logging.getLogger(f"contractiq.{name}")
