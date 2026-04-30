"""
agents/validator_agent.py — SQL Validation & Safety Layer.

THREE LAYERS (run in order; fail fast on first violation):

  Layer 1 — Keyword Safety (regex):
    Blocks any SQL containing destructive keywords from params.yaml.
    This is the first and fastest check.

  Layer 2 — Schema Check (table existence):
    Extracts table names from FROM / JOIN clauses and verifies each
    exists in the database. Catches hallucinated table names.

  Layer 3 — Syntax Check (sqlparse):
    Parses the SQL and confirms it is syntactically valid and starts
    with SELECT. Catches malformed SQL before it hits the database.

WHY NOT AN LLM:
  - Deterministic: same input always gives same output
  - Fast: ~1ms vs ~400ms for an LLM call
  - No token cost
  - 100% unit-testable without API keys

BLOCKED KEYWORDS come from config/params.yaml (safety.blocked_keywords)
— not hardcoded. Add/remove keywords by editing the YAML file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import sqlparse

from config import params
from core.schema_loader import get_schema_dict
from utils.log import logger


@dataclass
class ValidationResult:
    """Result from one SQL validation run."""

    valid: bool
    sql: str
    error_message: str = ""        # User-friendly message (shown in UI)
    technical_error: str = ""      # Internal detail (fed to generator on retry)


class ValidatorAgent:
    """
    Rules-based SQL validator. No LLM, no network calls.

    Reads blocked keywords from params.yaml so safety rules can be
    tightened without touching source code.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._schema = get_schema_dict(db_path)
        self._valid_tables: set[str] = {t.lower() for t in self._schema.keys()}
        # Normalise blocked keywords to uppercase for case-insensitive matching
        self._blocked: list[str] = [
            kw.upper() for kw in params.safety.blocked_keywords
        ]

    # ── Layer 1: Keyword Safety ───────────────────────────────────────────────

    def _check_keywords(self, sql: str) -> Optional[ValidationResult]:
        sql_upper = sql.upper()
        for keyword in self._blocked:
            # Word boundary prevents false positives ("CREATED" ≠ "CREATE")
            # For non-word tokens like "--" skip the boundary check
            if keyword.isidentifier():
                pattern = r"\b" + re.escape(keyword) + r"\b"
            else:
                pattern = re.escape(keyword)

            if re.search(pattern, sql_upper):
                return ValidationResult(
                    valid=False,
                    sql=sql,
                    error_message=(
                        f"⛔ This query contains a restricted operation ('{keyword}'). "
                        "Only read-only SELECT queries are allowed."
                    ),
                    technical_error=f"Blocked keyword: {keyword}",
                )
        return None

    # ── Layer 2: Schema Check ─────────────────────────────────────────────────

    def _check_schema(self, sql: str) -> Optional[ValidationResult]:
        """Verify all table names in FROM / JOIN clauses exist in the database."""
        table_pattern = r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        mentioned = re.findall(table_pattern, sql, re.IGNORECASE)

        unknown = [
            t for t in mentioned
            if t.lower() not in self._valid_tables
        ]

        if unknown:
            valid_list = ", ".join(sorted(self._schema.keys()))
            return ValidationResult(
                valid=False,
                sql=sql,
                error_message=(
                    f"Table(s) '{', '.join(unknown)}' not found in the database. "
                    f"Available tables: {valid_list}."
                ),
                technical_error=f"Unknown tables: {unknown}",
            )
        return None

    # ── Layer 3: Syntax Check ─────────────────────────────────────────────────

    def _check_syntax(self, sql: str) -> Optional[ValidationResult]:
        """Use sqlparse to confirm the SQL is well-formed and is a SELECT."""
        try:
            parsed = sqlparse.parse(sql)
            if not parsed or not parsed[0].tokens:
                return ValidationResult(
                    valid=False,
                    sql=sql,
                    error_message="The generated query appears to be empty or malformed.",
                    technical_error="sqlparse: empty parse result",
                )

            stmt_type = parsed[0].get_type()
            if stmt_type and stmt_type != "SELECT":
                return ValidationResult(
                    valid=False,
                    sql=sql,
                    error_message=f"Only SELECT queries are permitted (got: {stmt_type}).",
                    technical_error=f"Non-SELECT statement: {stmt_type}",
                )

        except Exception as exc:
            return ValidationResult(
                valid=False,
                sql=sql,
                error_message="The query has a syntax problem. Try rephrasing your question.",
                technical_error=str(exc),
            )
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, sql: str) -> ValidationResult:
        """
        Run all three validation layers on a SQL string.

        Returns the first failure found (fail-fast), or a valid result.

        Args:
            sql: SQL string from the SQL Generator agent.

        Returns:
            ValidationResult — check .valid before proceeding.
        """
        if not sql or not sql.strip():
            return ValidationResult(
                valid=False,
                sql=sql,
                error_message="No SQL was generated. Please try rephrasing your question.",
                technical_error="Empty SQL string",
            )

        for layer, check in [
            ("keyword", self._check_keywords),
            ("schema", self._check_schema),
            ("syntax", self._check_syntax),
        ]:
            result = check(sql)
            if result:
                logger.warning(f"Validator [{layer}] FAIL — {result.technical_error}")
                return result

        logger.info("Validator: all 3 layers passed ✅")
        return ValidationResult(valid=True, sql=sql)
