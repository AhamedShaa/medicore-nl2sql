"""
agents/sql_generator.py — Schema-Aware SQL Generator and Execution Repair Agent.

TWO MODES — WHY BOTH ARE NEEDED:
  generate() — standard NL → SQL generation.
    Used when: first attempt, or when validation fails (schema/syntax errors).
    Failure context: previous validation error (wrong table name, syntax error).
    Prompt focus: "translate this question to SQL using the schema below".

  repair() — targeted SQL fix for runtime execution errors.
    Used when: SQL passed validation but SQLite rejected it at execution time.
    Failure context: the failing SQL + the exact SQLite error message.
    Prompt focus: "this specific SQL fails with this specific error — fix only the error".
    WHY DIFFERENT PROMPT:
      Validation errors mean the LLM got the schema wrong (needs the full schema reminder).
      Execution errors mean the SQL was schema-valid but semantically wrong at runtime
      (e.g. column ambiguity, type mismatch in a comparison, aggregate in WHERE).
      The repair prompt is surgical — it shows the broken SQL and the exact error,
      and asks for a minimal fix rather than a full rewrite.

BACKUP MODEL SUPPORT:
  Both generate() and repair() call invoke_with_backup() from BaseAgent.
  If the primary model (GPT-4o) fails, the backup (GPT-4o-mini) handles the call.
  The return value includes a backup_model_used flag so the graph state can record it.

LCEL CHAINS:
  (system prompt with schema) | self.llm | StrOutputParser
  (repair prompt)             | self.llm | StrOutputParser
  Both have backup variants using self.backup_llm.
"""

from __future__ import annotations

import re
import time
from typing import Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import BaseAgent
from core.schema_loader import get_schema_string
from utils.log import logger


# ── Prompts ───────────────────────────────────────────────────────────────────

_GENERATE_SYSTEM = """\
You are a SQL expert for MediCore Hospital's analytics platform (SQLite database).

DATABASE SCHEMA:
{schema}

STRICT RULES:
1. ONLY generate SELECT statements. NEVER write INSERT, UPDATE, DELETE, DROP, ALTER, or DDL.
2. Use the EXACT table and column names from the schema — no guessing.
3. Always include a LIMIT clause (default LIMIT 100) unless the query asks for aggregations.
4. Use SQLite date functions: DATE(), strftime(), julianday(). Never use NOW() or GETDATE().
5. Return ONLY the SQL query — no explanation, no markdown, no backticks, no comments.

EXAMPLE:
  Question: Top 5 revenue-generating departments
  Answer:
  SELECT d.department_name, SUM(bi.total_amount) AS revenue
  FROM billing_invoices bi
  JOIN departments d ON bi.department_id = d.department_id
  GROUP BY d.department_name
  ORDER BY revenue DESC
  LIMIT 5;
"""

_REPAIR_SYSTEM = """\
You are a SQL debugger for a SQLite database. A previously generated SQL query has
failed at execution time or returned an unusable empty result. Your task is to fix
ONLY that issue.

DATABASE SCHEMA (for reference):
{schema}

RULES:
1. Return ONLY the corrected SELECT statement — no explanation, no markdown.
2. Make the minimal change needed to fix the error or empty-result cause.
3. Do not change the query's intent or the tables/columns used unless the error requires it.
4. Use only SQLite-compatible syntax (no PostgreSQL, no MySQL extensions).
5. If the query returned zero rows, preserve the user's requested meaning while checking for
   overly restrictive filters, wrong enum/literal values, incorrect join paths, alias misuse,
   and date/window assumptions. Do not invent rows by ignoring the user's main filter.

Common SQLite execution errors and fixes:
  "ambiguous column name"   → qualify with table alias, e.g. p.patient_id instead of patient_id
  "no such column"          → check exact column name in the schema above
  "datatype mismatch"       → cast with CAST(col AS type) or use correct comparison
  "no such table"           → use exact table name from the schema
  "aggregate functions not allowed in WHERE" → move aggregate to HAVING clause
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class SQLGeneratorAgent(BaseAgent):
    """
    Generates and repairs SQL queries using schema-aware LCEL chains.

    Two operations:
      generate(query, validation_error?)  — NL → SQL
      repair(sql, exec_error)             — fix a runtime-failing SQL
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        super().__init__()

        # Load schema once at init (LRU-cached in schema_loader)
        schema = get_schema_string(db_path)

        # Primary generation chain
        _gen_prompt = ChatPromptTemplate.from_messages([
            ("system", _GENERATE_SYSTEM),
            ("human", "{query}"),
        ]).partial(schema=schema)

        self._primary_gen   = _gen_prompt | self.llm         | StrOutputParser()
        self._backup_gen    = _gen_prompt | self.backup_llm  | StrOutputParser()

        # Execution repair chain
        _repair_prompt = ChatPromptTemplate.from_messages([
            ("system", _REPAIR_SYSTEM),
            ("human",
             "FAILING SQL:\n{sql}\n\n"
             "ORIGINAL USER QUESTION:\n{query}\n\n"
             "EXECUTION ERROR:\n{exec_error}\n\n"
             "Return the corrected SQL only."),
        ]).partial(schema=schema)

        self._primary_repair  = _repair_prompt | self.llm         | StrOutputParser()
        self._backup_repair   = _repair_prompt | self.backup_llm  | StrOutputParser()

        logger.debug("SQLGeneratorAgent chains built (generate + repair)")

    def generate(
        self,
        query: str,
        validation_error: Optional[str] = None,
    ) -> Tuple[str, float, bool]:
        """
        Generate SQL from a natural language question.

        On retry passes, the validation error is appended to the prompt so
        the LLM can self-correct without human intervention.

        Args:
            query:            Original natural language question.
            validation_error: Previous validation error to include as context
                              (pass on retry attempts, None on first attempt).

        Returns:
            (sql, latency_ms, backup_model_used)
        """
        prompt_text = query
        if validation_error:
            prompt_text = (
                f"{query}\n\n"
                f"PREVIOUS ATTEMPT FAILED VALIDATION:\n{validation_error}\n"
                "Fix the SQL and return only the corrected SELECT statement."
            )

        t0 = time.perf_counter()
        raw, used_backup = self.invoke_with_backup(
            self._primary_gen, self._backup_gen, {"query": prompt_text}
        )
        latency = (time.perf_counter() - t0) * 1000

        sql = _clean_sql(raw)
        logger.info(
            f"SQL Generator → {len(sql)} chars ({latency:.0f}ms)"
            + (" [backup]" if used_backup else "")
        )
        logger.debug(f"Generated SQL:\n{sql}")
        return sql, latency, used_backup

    def repair(
        self,
        sql: str,
        exec_error: str,
        query: Optional[str] = None,
    ) -> Tuple[str, float, bool]:
        """
        Repair a SQL query that passed validation but failed at execution time.

        WHY DIFFERENT FROM generate():
          Execution errors are runtime failures — the SQL was structurally correct
          but had a semantic problem (ambiguous column, type mismatch, etc.).
          The repair prompt shows the exact failing SQL and exact error message,
          asking for a surgical fix rather than a full rewrite.

        Args:
            sql:        The SQL that failed execution.
            exec_error: The exact SQLite error message, or an empty-result repair reason.
            query:      Original natural language question, used to preserve intent.

        Returns:
            (repaired_sql, latency_ms, backup_model_used)
        """
        t0 = time.perf_counter()
        raw, used_backup = self.invoke_with_backup(
            self._primary_repair, self._backup_repair,
            {
                "sql": sql,
                "exec_error": exec_error,
                "query": query or "Not provided",
            },
        )
        latency = (time.perf_counter() - t0) * 1000

        repaired = _clean_sql(raw)
        logger.info(
            f"SQL Repair → {len(repaired)} chars ({latency:.0f}ms)"
            + (" [backup]" if used_backup else "")
        )
        logger.debug(f"Repaired SQL:\n{repaired}")
        return repaired, latency, used_backup


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_sql(raw: str) -> str:
    """
    Strip markdown fences and preamble text from raw LLM output.

    LLMs occasionally wrap SQL in ```sql ... ``` blocks or precede it
    with a sentence like "Here is the corrected SQL:". This extracts
    only the SELECT statement.
    """
    clean = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip("`\n ")
    match = re.search(r"\bSELECT\b", clean, re.IGNORECASE)
    if match:
        return clean[match.start():].strip()
    return clean.strip()
