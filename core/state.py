"""
core/state.py — Shared LangGraph state for the NL2SQL pipeline.

WHY A TYPED STATE:
  LangGraph nodes communicate by reading from and writing to a shared state dict.
  Using TypedDict instead of a plain dict gives:
    - Type hints → IDE autocomplete + mypy checking
    - Self-documenting structure (one place to see all pipeline data)
    - LangGraph validates keys against the TypedDict at compile time

STATE FLOW:
  Each node receives the full state and returns a PARTIAL dict with only
  the keys it modifies. LangGraph merges the partial update into the state.

  route_intent_node     → sets: intent, intent_reason, clarification_needed
  generate_sql_node     → sets: sql, sql_attempts
  validate_sql_node     → sets: is_valid, validation_error_message, last_validation_error
  execute_sql_node      → sets: exec_rows, exec_columns, exec_row_count, exec_time_ms
                                 OR exec_error + failure_stage on error
  repair_sql_node       → sets: sql, exec_repair_attempts (loop back to validate_sql)
  interpret_result_node → sets: insight, chart_spec, summary_title, success=True
  handle_fallback_node  → sets: fallback_message, fallback_suggestions, success=False

RETRY COUNTERS:
  sql_attempts         counts validation-retry cycles (generate → validate → generate)
  exec_repair_attempts counts execution-repair cycles (execute → repair → validate → execute)
  Both are bounded by params.pipeline settings and checked in conditional edges.

BACKUP MODEL:
  backup_model_used is set True by any LLM node that falls back to the backup model
  after the primary model raises an API/network exception.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class NL2SQLState(TypedDict):
    """
    Complete shared state flowing through every node in the LangGraph pipeline.
    All fields are optional at init — nodes populate them progressively.
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    query: str                            # Original natural language query
    session_id: str                       # Unique run ID (used for trace file name)

    # ── Routing ────────────────────────────────────────────────────────────────
    intent: str                           # "analytics" | "clarify" | "blocked"
    intent_reason: str                    # Why the router chose this intent
    clarification_needed: Optional[str]   # Question to ask user (clarify intent only)

    # ── SQL Generation ─────────────────────────────────────────────────────────
    sql: str                              # Most recently generated / repaired SQL
    sql_attempts: int                     # Validation-retry counter (max = max_retry_attempts)
    last_validation_error: str            # Technical error fed back on validation retry

    # ── Validation ─────────────────────────────────────────────────────────────
    is_valid: bool                        # Did SQL pass all 3 validation layers?
    validation_error_message: str         # User-friendly validation error
    validation_technical_error: str       # Technical detail (retry context)

    # ── Execution ──────────────────────────────────────────────────────────────
    exec_rows: List[Dict[str, Any]]       # Result rows as list of dicts
    exec_columns: List[str]               # Column names in result order
    exec_row_count: int                   # Total rows returned
    exec_time_ms: float                   # SQL execution time in ms
    exec_error: str                       # Execution error message (repair context)

    # ── Execution Repair ───────────────────────────────────────────────────────
    exec_repair_attempts: int             # How many execution-repair cycles used
    # WHY SEPARATE from sql_attempts:
    #   sql_attempts tracks "generator didn't know the schema well enough"
    #   exec_repair_attempts tracks "SQL was structurally valid but logically wrong at runtime"
    #   Different error types need different prompts, different budgets, different fallback messages.

    # ── Result Interpretation ──────────────────────────────────────────────────
    insight: str                          # 2-3 sentence natural language insight
    chart_spec: Optional[Dict]            # {type, title, x_column, y_column, ...}
    summary_title: str                    # Short panel title (max 8 words)

    # ── Fallback ───────────────────────────────────────────────────────────────
    fallback_message: str                 # User-visible error / clarification
    fallback_suggestions: List[str]       # Schema-aware example queries to try
    failure_stage: str                    # "routing"|"generation"|"validation"|"execution"|"interpretation"
    failure_category: str                 # "validation"|"execution"|"api_error"|"ambiguous"|"unknown"

    # ── Observability metadata ─────────────────────────────────────────────────
    success: bool                         # True = reached interpret_result; False = fallback
    backup_model_used: bool               # True if any node fell back to backup model
    error: Optional[str]                  # Internal error (logged, not shown to user)
    trace_file: str                       # Path to saved JSON trace
    total_latency_ms: float               # End-to-end wall-clock time


def make_initial_state(query: str, session_id: str) -> NL2SQLState:
    """
    Build a clean initial state for a new pipeline run.
    All fields start empty / zero — nodes fill them in progressively.

    Args:
        query:      The user's natural language question.
        session_id: Unique run identifier (used for trace file naming).

    Returns:
        A fully-typed NL2SQLState dict ready for nl2sql_app.invoke().
    """
    return NL2SQLState(
        query=query,
        session_id=session_id,
        # Routing
        intent="",
        intent_reason="",
        clarification_needed=None,
        # SQL generation
        sql="",
        sql_attempts=0,
        last_validation_error="",
        # Validation
        is_valid=False,
        validation_error_message="",
        validation_technical_error="",
        # Execution
        exec_rows=[],
        exec_columns=[],
        exec_row_count=0,
        exec_time_ms=0.0,
        exec_error="",
        # Execution repair
        exec_repair_attempts=0,
        # Interpretation
        insight="",
        chart_spec=None,
        summary_title="",
        # Fallback
        fallback_message="",
        fallback_suggestions=[],
        failure_stage="",
        failure_category="",
        # Metadata
        success=False,
        backup_model_used=False,
        error=None,
        trace_file="",
        total_latency_ms=0.0,
    )
