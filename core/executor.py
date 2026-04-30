"""
core/executor.py — Safe SQL execution against the MediCore database.

SAFETY DESIGN (defense-in-depth):
  1. ValidatorAgent blocks dangerous SQL before it reaches this module
  2. This executor adds a SECOND independent safety check (keyword scan +
     SELECT-only enforcement) — so even if the validator is bypassed, the
     executor refuses anything other than SELECT queries
  3. sqlite3.Error is caught and returned as a structured failure — it is
     NEVER raised to the caller or shown to the end user

WHY RETURN ExecutionResult INSTEAD OF RAISING:
  The LangGraph graph uses return values, not exceptions, to decide routing.
  Returning a structured dataclass keeps error handling explicit and testable.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import params
from utils.log import logger


@dataclass
class ExecutionResult:
    """Structured result from one SQL execution attempt."""

    success: bool
    rows: List[Dict[str, Any]]       # Each row as a dict keyed by column name
    columns: List[str]               # Column names in result order
    row_count: int
    execution_time_ms: float
    query: str = ""
    error: Optional[str] = None      # Set on failure; never shown directly to users


def execute_query(sql: str, db_path: Optional[str] = None) -> ExecutionResult:
    """
    Execute a validated SELECT query and return a structured result.

    Safety checks performed before touching the database:
      1. Blocked keyword scan (reads params.safety.blocked_keywords)
      2. Ensures the first SQL token is SELECT

    Args:
        sql:     A SQL query that has already passed ValidatorAgent.
        db_path: SQLite path or SQLAlchemy URI. Defaults to params.database.path.

    Returns:
        ExecutionResult — check .success before accessing .rows / .columns.
    """
    db_path = db_path or params.database.path
    t0 = time.perf_counter()

    # ── Final safety gate ─────────────────────────────────────────────────────
    sql_upper = sql.strip().upper()

    blocked: List[str] = [kw.upper() for kw in params.safety.blocked_keywords]
    for keyword in blocked:
        if keyword in sql_upper:
            logger.warning(f"Executor blocked query containing '{keyword}'")
            return ExecutionResult(
                success=False,
                rows=[],
                columns=[],
                row_count=0,
                execution_time_ms=0.0,
                query=sql,
                error=f"Forbidden keyword '{keyword}' detected in query.",
            )

    if not sql_upper.lstrip().startswith("SELECT"):
        logger.warning("Executor rejected non-SELECT query")
        return ExecutionResult(
            success=False,
            rows=[],
            columns=[],
            row_count=0,
            execution_time_ms=0.0,
            query=sql,
            error="Only SELECT queries are permitted.",
        )

    # ── Execute ───────────────────────────────────────────────────────────────
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        raw_rows = cursor.fetchall()

        columns = [desc[0] for desc in (cursor.description or [])]
        rows = [dict(zip(columns, row)) for row in raw_rows]
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(f"Query OK — {len(rows)} rows in {elapsed}ms")
        return ExecutionResult(
            success=True,
            rows=rows,
            columns=columns,
            row_count=len(rows),
            execution_time_ms=elapsed,
            query=sql,
        )

    except sqlite3.Error as exc:
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.error(f"SQL execution error: {exc}")
        return ExecutionResult(
            success=False,
            rows=[],
            columns=[],
            row_count=0,
            execution_time_ms=elapsed,
            query=sql,
            error=str(exc),
        )

    finally:
        if conn:
            conn.close()
