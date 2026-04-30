"""
utils — Shared utilities used across all modules.

CONTENTS:
  log.py  — Centralized Loguru logger configured from config/params.yaml.
             Every module imports the logger from here instead of instantiating
             its own, ensuring consistent format, level, and file rotation
             across the entire application.

USAGE:
    from utils import logger
    # or
    from utils.log import logger

    logger.info("Pipeline started")
    logger.debug(f"Schema: {schema}")
    logger.warning("LangFuse keys not set — local tracing only")
    logger.error(f"Execution failed: {error}")
    logger.exception("Unexpected error")   # includes full traceback
"""

from utils.log import logger

__all__ = ["logger"]
