"""
utils/log.py — Centralized Loguru logger for MediCore NL2SQL Platform.

WHY LOGURU OVER STANDARD LOGGING:
  - Zero boilerplate: no getLogger(__name__), no basicConfig, no Formatter classes
  - Colourised console output out of the box
  - Automatic file rotation and retention (configured in params.yaml)
  - Structured exception tracebacks with full context
  - Thread-safe by default

USAGE (in any module):
    from utils.log import logger

    logger.info("Pipeline started")
    logger.debug(f"Schema loaded: {table_count} tables")
    logger.warning("LangFuse keys not set — tracing to local files only")
    logger.error(f"SQL execution failed: {error}")
    logger.exception("Unexpected error")  # includes full traceback

LOG LEVELS (from params.yaml → logging.level, or LOG_LEVEL env var):
    DEBUG    → verbose, including agent prompts / responses
    INFO     → normal operation (default)
    WARNING  → non-fatal issues (missing optional keys, fallbacks)
    ERROR    → failures that need attention
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from config import params


def _configure_logger() -> None:
    """
    Remove Loguru's default handler and replace with:
      1. Colourised stderr output
      2. Rotating file output (rotation + retention from params.yaml)
    """
    # Remove the default handler (plain stderr, no colour)
    logger.remove()

    fmt: str = params.logging.format
    level: str = params.logging.level

    # ── Console handler ────────────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        colorize=True,
        backtrace=True,      # Show variables in exception tracebacks
        diagnose=True,       # Annotate exception tracebacks with local values
    )

    # ── File handler (rotating) ────────────────────────────────────────────────
    log_path = Path(params.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.add(
            str(log_path),
            level=level,
            format=fmt,
            rotation=params.logging.rotation,    # e.g. "10 MB"
            retention=params.logging.retention,  # e.g. "30 days"
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
            enqueue=True,   # Thread-safe async writes where multiprocessing is allowed
        )
    except PermissionError:
        logger.add(
            str(log_path),
            level=level,
            format=fmt,
            rotation=params.logging.rotation,
            retention=params.logging.retention,
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
            enqueue=False,
        )


_configure_logger()

# Re-export the configured logger — all modules import from here
__all__ = ["logger"]
