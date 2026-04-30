"""
core/schema_loader.py — Dynamic database schema introspection.

WHY LANGCHAIN SQLDatabase:
  - Wraps SQLAlchemy: works with SQLite, PostgreSQL, MySQL, MSSQL without
    changing any code — just update DB_PATH in .env
  - get_table_info() returns a formatted schema string (table names, column
    types, PKs, FKs, and sample rows) ready to inject into the LLM prompt
  - sample_rows_in_table_info count is configurable via params.yaml
  - Actively maintained as part of LangChain's SQL toolchain

WHY DYNAMIC (NOT HARDCODED):
  If a column is renamed or a table is added, the LLM prompt auto-updates.
  Hardcoding the schema would require a code change for every DB migration.

WHY LRU CACHE:
  Schema doesn't change during a session. Caching avoids re-querying the
  database on every pipeline run (~12 PRAGMA calls per run saved).

USAGE:
    from core.schema_loader import get_schema_string, get_schema_dict

    schema_str = get_schema_string()   # → injected into LLM system prompt
    schema_dict = get_schema_dict()    # → used by ValidatorAgent
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from langchain_community.utilities import SQLDatabase
from sqlalchemy import inspect as sa_inspect

from config import params
from utils.log import logger


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_uri(db_path: str) -> str:
    """
    Convert a path or URI to a full SQLAlchemy connection URI.
    SQLite paths are prefixed with sqlite:///
    Full URIs (postgresql://, etc.) are passed through unchanged.
    """
    if "://" in db_path:
        return db_path
    return f"sqlite:///{db_path}"


def _make_db(db_path: str) -> SQLDatabase:
    """Instantiate a LangChain SQLDatabase from a path or URI."""
    uri = _build_uri(db_path)
    db = SQLDatabase.from_uri(
        uri,
        sample_rows_in_table_info=params.database.schema_sample_rows,
    )
    logger.debug(f"SQLDatabase connected → {uri}")
    return db


# ── Public API ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def get_schema_string(db_path: Optional[str] = None) -> str:
    """
    Return a formatted schema string for injection into the LLM system prompt.

    The string includes:
      - Table names and column types
      - Primary and foreign key annotations
      - Sample rows (count configured via params.yaml: database.schema_sample_rows)

    Result is cached per db_path so repeated pipeline runs don't re-query DB.

    Args:
        db_path: SQLite path or SQLAlchemy URI. Defaults to params.database.path.

    Returns:
        Multi-line schema description string.
    """
    db_path = db_path or params.database.path
    db = _make_db(db_path)
    schema = db.get_table_info()
    table_count = len(db.get_usable_table_names())
    logger.info(f"Schema loaded: {table_count} tables from '{db_path}'")
    return schema


@lru_cache(maxsize=4)
def get_schema_dict(db_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Return a dict mapping table names → list of column names.

    Used by ValidatorAgent to check that generated SQL references
    only tables and columns that actually exist in the database.

    Example:
        {
          "patients": ["patient_id", "first_name", "last_name", ...],
          "doctors":  ["doctor_id", "first_name", "specialty_id", ...],
        }

    Args:
        db_path: SQLite path or SQLAlchemy URI. Defaults to params.database.path.

    Returns:
        Dict of {table_name: [column_names]}.
    """
    db_path = db_path or params.database.path
    db = _make_db(db_path)

    inspector = sa_inspect(db._engine)
    result: Dict[str, List[str]] = {}
    for table_name in inspector.get_table_names():
        cols = [col["name"] for col in inspector.get_columns(table_name)]
        result[table_name] = cols

    logger.debug(f"Schema dict built: {list(result.keys())}")
    return result


def get_table_names(db_path: Optional[str] = None) -> List[str]:
    """Return the list of table names in the database."""
    return list(get_schema_dict(db_path).keys())


def invalidate_cache() -> None:
    """
    Clear the LRU schema caches.
    Call this after db_setup loads new data to force a fresh schema read.
    """
    get_schema_string.cache_clear()
    get_schema_dict.cache_clear()
    logger.debug("Schema cache invalidated")
