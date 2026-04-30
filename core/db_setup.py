"""
core/db_setup.py — Load the MediCore SQL dump into a local SQLite database.

WHY SQLITE FOR DEVELOPMENT:
  - Zero installation — no server process required
  - The entire database is a single file (medicore.db)
  - LangChain SQLDatabase and SQLAlchemy support SQLite natively
  - For production, swap to PostgreSQL by updating DB_PATH in .env
    (everything in schema_loader.py works with both via SQLAlchemy)

WHY A PREPROCESSOR:
  The provided SQL dump is a PostgreSQL export. SQLite speaks a subset of
  standard SQL and rejects several PostgreSQL-specific constructs. Rather than
  asking for a different dump format, we normalise the SQL in memory before
  executing it. Transformations applied:
    - Remove PostgreSQL SET/SELECT config commands
    - Strip `CASCADE` from DROP TABLE statements (PG syntax, not needed in SQLite)
    - Remove IF NOT EXISTS / CREATE SEQUENCE / COMMENT ON / ALTER TABLE … OWNER TO
    - Replace PG-only data types: SERIAL→INTEGER, BIGSERIAL→INTEGER,
      BOOLEAN→INTEGER, TEXT[]→TEXT, JSONB/JSON→TEXT, BYTEA→BLOB, UUID→TEXT
    - Replace NOW()/CURRENT_TIMESTAMP(n) → datetime('now')
    - Drop CONSTRAINT … DEFERRABLE / INITIALLY DEFERRED clauses
    - Remove trailing commas left before a closing paren by stripped lines

HOW TO USE:
    # Place medicore_data.sql in the project root, then run:
    python -m core.db_setup --sql medicore_data.sql

    # Custom output path:
    python -m core.db_setup --sql medicore_data.sql --db /custom/path/medicore.db

    # Also callable from Python (e.g. in the notebook):
    from core.db_setup import load_sql_file
    load_sql_file("medicore_data.sql")
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Optional

from config import params
from core.schema_loader import invalidate_cache
from utils.log import logger


# ── PostgreSQL → SQLite preprocessor ─────────────────────────────────────────

# Lines that start with these patterns serve no purpose in SQLite and are dropped
_DROP_LINE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*SET\s+", re.IGNORECASE),
    re.compile(r"^\s*SELECT\s+pg_catalog\.", re.IGNORECASE),
    re.compile(r"^\s*SELECT\s+setval\b", re.IGNORECASE),   # sequence resets
    re.compile(r"^\s*\\", ),                          # psql meta-commands
    re.compile(r"^\s*CREATE\s+SEQUENCE\b", re.IGNORECASE),
    re.compile(r"^\s*ALTER\s+SEQUENCE\b", re.IGNORECASE),
    re.compile(r"^\s*COMMENT\s+ON\b", re.IGNORECASE),
    re.compile(r"^\s*ALTER\s+TABLE\s+\S+\s+OWNER\s+TO\b", re.IGNORECASE),
    re.compile(r"^\s*GRANT\b", re.IGNORECASE),
    re.compile(r"^\s*REVOKE\b", re.IGNORECASE),
]

# Inline replacements applied to every remaining line
_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # DROP TABLE … CASCADE  →  DROP TABLE …
    (re.compile(r"\bCASCADE\b", re.IGNORECASE), ""),
    # Data types
    (re.compile(r"\bBIGSERIAL\b", re.IGNORECASE), "INTEGER"),
    (re.compile(r"\bSERIAL\b", re.IGNORECASE), "INTEGER"),
    (re.compile(r"\bBOOLEAN\b", re.IGNORECASE), "INTEGER"),
    (re.compile(r"\bBYTEA\b", re.IGNORECASE), "BLOB"),
    (re.compile(r"\bJSONB\b", re.IGNORECASE), "TEXT"),
    (re.compile(r"\bJSON\b", re.IGNORECASE), "TEXT"),
    (re.compile(r"\bUUID\b", re.IGNORECASE), "TEXT"),
    (re.compile(r"\bTEXT\[\]", re.IGNORECASE), "TEXT"),
    (re.compile(r"\bVARCHAR\b", re.IGNORECASE), "TEXT"),
    (re.compile(r"\bCHARACTER VARYING\b", re.IGNORECASE), "TEXT"),
    (re.compile(r"\bDOUBLE PRECISION\b", re.IGNORECASE), "REAL"),
    # Time column type → TEXT (SQLite has no native TIME type)
    (re.compile(r"\bTIMESTAMPTZ\b", re.IGNORECASE), "TEXT"),
    (re.compile(r"\bTIMESTAMP\b(?!\s+DEFAULT)", re.IGNORECASE), "TEXT"),
    (re.compile(r"\bTIME\b(?!\s+ZONE|\s+WITH)", re.IGNORECASE), "TEXT"),
    # Timestamps in DEFAULT — wrap in parens (SQLite requires expressions in parens)
    (re.compile(r"\bNOW\(\)", re.IGNORECASE), "(datetime('now'))"),
    (re.compile(r"\bCURRENT_TIMESTAMP\b", re.IGNORECASE), "(datetime('now'))"),
    # Deferrable constraints (not supported by SQLite)
    (re.compile(
        r"\s*DEFERRABLE\s+INITIALLY\s+(?:DEFERRED|IMMEDIATE)\b",
        re.IGNORECASE,
    ), ""),
    (re.compile(r"\s*NOT\s+DEFERRABLE\b", re.IGNORECASE), ""),
    # nextval('sequence') default  →  NULL (SQLite uses AUTOINCREMENT instead)
    (re.compile(r"DEFAULT\s+nextval\('[^']+'\)", re.IGNORECASE), ""),
]


def _preprocess(sql: str) -> str:
    """
    Convert a PostgreSQL dump to SQLite-compatible SQL.

    WHY in-memory preprocessing (not a separate tool):
      Keeps the setup as one self-contained command. Users don't need
      pg_dump flags or external converters — just hand us the dump file.

    Returns:
        Cleaned SQL string ready for sqlite3.executescript().
    """
    lines = sql.splitlines()
    cleaned: list[str] = []

    for line in lines:
        # Drop PostgreSQL-only lines entirely
        if any(pat.match(line) for pat in _DROP_LINE_PATTERNS):
            continue

        # Apply inline substitutions
        for pattern, replacement in _REPLACEMENTS:
            line = pattern.sub(replacement, line)

        cleaned.append(line)

    result = "\n".join(cleaned)

    # Fix trailing commas before closing parens left by stripped lines
    # e.g.  "    col TEXT,\n)" → "    col TEXT\n)"
    result = re.sub(r",(\s*\))", r"\1", result)

    return result


# ── Main loader ───────────────────────────────────────────────────────────────

def load_sql_file(
    sql_path: str,
    db_path: Optional[str] = None,
) -> None:
    """
    Execute a SQL dump file against a SQLite database.

    Preprocesses PostgreSQL-specific syntax before execution so the same
    dump file works without modification.

    Args:
        sql_path: Path to the .sql dump file (e.g. "medicore_data.sql").
        db_path:  Output SQLite database path. Defaults to params.database.path.

    Raises:
        FileNotFoundError: If sql_path does not exist.
        sqlite3.Error:     If the SQL execution fails after preprocessing.
    """
    db_path = db_path or params.database.path
    sql_file = Path(sql_path)

    if not sql_file.exists():
        raise FileNotFoundError(
            f"SQL file not found: {sql_path}\n"
            "Place medicore_data.sql in the project root and re-run."
        )

    logger.info(f"Loading '{sql_path}' → '{db_path}'")

    raw_sql = sql_file.read_text(encoding="utf-8")
    sql_content = _preprocess(raw_sql)
    logger.debug(f"Preprocessed SQL: {len(raw_sql)} → {len(sql_content)} chars")

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql_content)
        conn.commit()

        # Verify load succeeded
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        logger.info(f"Database loaded: {len(tables)} tables → {tables}")
        print(f"Database ready: {len(tables)} tables in '{db_path}'")
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {count:,} rows")

    except sqlite3.Error as exc:
        logger.error(f"Failed to load SQL file: {exc}")
        raise

    finally:
        conn.close()

    # Invalidate schema cache so next pipeline run reads the fresh schema
    invalidate_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load MediCore SQL dump into SQLite database."
    )
    parser.add_argument(
        "--sql",
        default="medicore_data.sql",
        help="Path to the .sql dump file (default: medicore_data.sql)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Output SQLite path (default: from params.yaml / .env)",
    )
    args = parser.parse_args()
    load_sql_file(args.sql, args.db)
