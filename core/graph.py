"""
core/graph.py — LangGraph StateGraph: targeted-retry NL2SQL pipeline.

TARGETED RETRY DESIGN — WHY NOT RETRY EVERYTHING:
  Retrying every node adds latency for the common case (most queries succeed on
  the first attempt). Instead, retries are applied only where they add clear value:

  1. SQL generation after validation failure
       generate_sql → validate_sql → generate_sql (loop, max_retry_attempts)
       The validation error is passed back so the LLM self-corrects.

  2. Transient API failure → backup model (GPT-4o-mini)
       invoke_with_backup() in BaseAgent handles this transparently.
       Primary: GPT-4o. Backup: GPT-4o-mini (different model — cheaper/faster).
       Applied in: generate_sql_node, repair_sql_node, interpret_result_node.

  3. Execution failure ONLY for repairable errors
       _is_repairable_error() classifies the SQLite error before deciding to repair.
       Repairable: ambiguous column, wrong column name, type mismatch, aggregate in WHERE.
       Non-repairable: missing table, permission error, database locked → fallback directly.
       This avoids wasting an LLM call on errors that need a full rewrite, not a patch.

GRAPH TOPOLOGY:

  START
    │
    ▼
  route_intent ──── clarify / blocked ──────────────────────────► handle_fallback ──► END
    │ analytics
    ▼
  generate_sql ◄──────────────────────────────── [retry: validation failed, attempts < max]
    │
    ▼
  validate_sql ──── invalid + attempts ≥ max ───────────────────► handle_fallback ──► END
    │ valid
    ▼
  execute_sql ──── repairable error + repairs < max ──► repair_sql ──► validate_sql ──► execute_sql
    │ success   non-repairable error / repairs exhausted ──────────────► handle_fallback ──► END
    ▼
  interpret_result ──────────────────────────────────────────────────────────────► END
    (partial success if interpret fails: returns raw data with a basic insight)

WHY repair_sql LOOPS BACK THROUGH validate_sql:
  After repair_sql rewrites the SQL, it must be re-validated before execution.
  The repaired SQL might introduce a new schema error (wrong table alias, etc.)

PUBLIC ENTRY POINT:
  from core.graph import run_pipeline

  state = run_pipeline("Which departments generate the most revenue?")
  if state["success"]:
      print(state["insight"])
  else:
      print(state["fallback_message"])
      print(state["fallback_hint"])
      print(state["fallback_suggestions"])
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from agents.fallback_agent import FallbackAgent
from agents.result_interpreter import ResultInterpreterAgent
from agents.router_agent import RouterAgent
from agents.sql_generator import SQLGeneratorAgent
from agents.validator_agent import ValidatorAgent
from config import params
from core.executor import ExecutionResult, execute_query
from core.schema_loader import get_schema_dict
from core.state import NL2SQLState, make_initial_state
from core.tracer import Tracer, set_active_trace_id
from utils.log import logger


# ── Singleton agents ──────────────────────────────────────────────────────────
# Instantiated once at module load; reused for every pipeline call.
# Schema is loaded once (LRU-cached) and baked into the generator at init.

_router      = RouterAgent()
_generator   = SQLGeneratorAgent()
_validator   = ValidatorAgent()
_interpreter = ResultInterpreterAgent()
_fallback    = FallbackAgent()


# ── Node functions ─────────────────────────────────────────────────────────────
# Each node receives the full NL2SQLState and returns a PARTIAL dict containing
# only the keys it modified. LangGraph merges partials into the shared state.
#
# EXCEPTION POLICY:
#   Every node catches ALL exceptions and converts them into a state update
#   with failure_stage and failure_category set. This ensures no unhandled
#   exception can crash the pipeline — the graph always routes to handle_fallback.

def route_intent_node(state: NL2SQLState) -> Dict[str, Any]:
    """Classify the user query: analytics / clarify / blocked."""
    try:
        result, _ = _router.route(state["query"])
        return {
            "intent": result.intent,
            "intent_reason": result.reason,
            "clarification_needed": result.clarification_needed,
        }
    except Exception as exc:
        logger.exception(f"route_intent_node failed: {exc}")
        return {
            "intent": "clarify",
            "intent_reason": "Router error",
            "clarification_needed": "I had trouble understanding your question. Please rephrase it.",
            "failure_stage": "routing",
            "failure_category": "api_error",
            "error": str(exc),
        }


def generate_sql_node(state: NL2SQLState) -> Dict[str, Any]:
    """
    Generate SQL from the user query.

    On retry attempts (sql_attempts > 0), passes the previous validation error
    as context so the LLM self-corrects rather than repeating the same mistake.
    """
    try:
        validation_error = state.get("last_validation_error") or None
        sql, _, used_backup = _generator.generate(
            state["query"],
            validation_error=validation_error,
        )
        return {
            "sql": sql,
            "sql_attempts": state.get("sql_attempts", 0) + 1,
            "backup_model_used": state.get("backup_model_used", False) or used_backup,
        }
    except Exception as exc:
        logger.exception(f"generate_sql_node failed: {exc}")
        return {
            "sql_attempts": state.get("sql_attempts", 0) + 1,
            "failure_stage": "generation",
            "failure_category": "api_error",
            "error": str(exc),
        }


def validate_sql_node(state: NL2SQLState) -> Dict[str, Any]:
    """Run 3-layer safety validation: keyword block → schema check → syntax check."""
    try:
        result = _validator.validate(state.get("sql", ""))
        return {
            "is_valid": result.valid,
            "validation_error_message": result.error_message,
            "last_validation_error": result.technical_error,
            "validation_technical_error": result.technical_error,
        }
    except Exception as exc:
        logger.exception(f"validate_sql_node failed: {exc}")
        return {
            "is_valid": False,
            "validation_error_message": "Validation error",
            "last_validation_error": str(exc),
            "failure_stage": "validation",
            "failure_category": "unknown",
            "error": str(exc),
        }


def execute_sql_node(state: NL2SQLState) -> Dict[str, Any]:
    """Execute the validated SQL and return structured results."""
    try:
        result = execute_query(state["sql"])

        if result.success:
            return {
                "exec_rows":      result.rows,
                "exec_columns":   result.columns,
                "exec_row_count": result.row_count,
                "exec_time_ms":   result.execution_time_ms,
                "exec_error":     "",
            }

        # Execution failed — store error for repair_sql_node
        return {
            "exec_error":      result.error,
            "failure_stage":   "execution",
            "failure_category": "execution",
        }

    except Exception as exc:
        logger.exception(f"execute_sql_node failed unexpectedly: {exc}")
        return {
            "exec_error":      str(exc),
            "failure_stage":   "execution",
            "failure_category": "execution",
            "error":            str(exc),
        }


def repair_sql_node(state: NL2SQLState) -> Dict[str, Any]:
    """
    Repair SQL that passed validation but failed at execution time.

    WHY THIS IS A SEPARATE NODE FROM generate_sql:
      generate_sql handles schema/syntax errors caught by the validator.
      repair_sql handles runtime errors that only appear when SQLite actually
      executes the query (column ambiguity, type mismatch, aggregate placement).
      The repair prompt is surgical — it shows the exact failing SQL + error,
      asking for a minimal fix rather than a full rewrite from the NL question.

    After repair, the graph loops back to validate_sql to re-check the
    repaired SQL before allowing another execution attempt.
    """
    try:
        repaired_sql, _, used_backup = _generator.repair(
            sql=state["sql"],
            exec_error=state.get("exec_error", "Unknown execution error"),
        )
        return {
            "sql": repaired_sql,
            "exec_repair_attempts": state.get("exec_repair_attempts", 0) + 1,
            "backup_model_used": state.get("backup_model_used", False) or used_backup,
            # Reset validation flags so validate_sql runs fresh on the repaired SQL
            "is_valid": False,
            "last_validation_error": "",
        }
    except Exception as exc:
        logger.exception(f"repair_sql_node failed: {exc}")
        return {
            "exec_repair_attempts": state.get("exec_repair_attempts", 0) + 1,
            "failure_stage": "execution",
            "failure_category": "api_error",
            "error": str(exc),
        }


def interpret_result_node(state: NL2SQLState) -> Dict[str, Any]:
    """
    Convert raw SQL results into a natural language insight + chart spec.

    NON-FATAL: if interpretation fails, a minimal insight is synthesised from
    the raw row count and the pipeline still reports success=True.
    The user gets their data even if the AI couldn't write a nice summary.
    """
    exec_result = ExecutionResult(
        success=True,
        rows=state["exec_rows"],
        columns=state["exec_columns"],
        row_count=state["exec_row_count"],
        execution_time_ms=state["exec_time_ms"],
        query=state["sql"],
    )

    try:
        interpreted, _ = _interpreter.interpret(state["query"], exec_result)
        return {
            "insight":       interpreted.insight,
            "chart_spec":    vars(interpreted.chart_spec) if interpreted.chart_spec else None,
            "summary_title": interpreted.summary_title,
            "success":       True,
        }
    except Exception as exc:
        # Interpretation failure is non-fatal: return raw data with a minimal insight
        logger.warning(f"interpret_result_node failed (non-fatal): {exc}")
        row_count = state.get("exec_row_count", 0)
        cols = state.get("exec_columns", [])
        return {
            "insight": (
                f"Query returned {row_count} row(s) with columns: {', '.join(cols)}. "
                "Automated insight unavailable — see raw data below."
            ),
            "chart_spec": {"type": "table", "title": "Query Results"},
            "summary_title": "Query Results",
            "success": True,   # Still a success — the user got their data
        }


def handle_fallback_node(state: NL2SQLState) -> Dict[str, Any]:
    """
    Final handler for all failure paths.

    Generates:
      - A precise failure message (what happened)
      - A contextual hint (what the user should do)
      - Schema-verified example suggestions
      - A machine-readable recovery_action for the UI

    WHY SCHEMA DICT IS PASSED:
      Dynamic suggestions are filtered against the live database so every
      suggestion shown is guaranteed to be answerable by the current schema.
    """
    intent = state.get("intent", "")

    # ── Clarify / Blocked are routing-level outcomes, not errors ─────────────
    if intent == "blocked":
        message = (
            f"⛔ Request blocked: {state.get('intent_reason', '')}. "
            "This system supports read-only analytics queries only."
        )
        return {
            "fallback_message":     message,
            "fallback_hint":        "Ask for data retrieval — counts, sums, trends, or lists.",
            "fallback_suggestions": _get_suggestions(),
            "fallback_recovery":    "change_query_type",
            "success": False,
            "failure_stage": "routing",
            "failure_category": "validation",
        }

    if intent == "clarify":
        message = (
            state.get("clarification_needed")
            or "Could you be more specific about what data you need?"
        )
        return {
            "fallback_message":     message,
            "fallback_hint":        "Add specifics: which table, time period, or metric.",
            "fallback_suggestions": _get_suggestions(),
            "fallback_recovery":    "add_specifics",
            "success": False,
            "failure_stage": "routing",
            "failure_category": "ambiguous",
        }

    # ── All other failure paths: delegate to FallbackAgent ───────────────────
    category = state.get("failure_category") or state.get("failure_stage") or "unknown"
    error    = state.get("exec_error") or state.get("error") or \
               state.get("validation_error_message", "")

    schema_dict = _get_schema_safe()
    result = _fallback.handle(
        failure_category=category,
        original_query=state.get("query", ""),
        technical_error=error,
        schema_dict=schema_dict,
    )

    return {
        "fallback_message":     result.message,
        "fallback_hint":        result.hint,
        "fallback_suggestions": result.suggestions,
        "fallback_recovery":    result.recovery_action,
        "success": False,
        "failure_category": result.failure_category,
    }


# ── Conditional edge routing functions ────────────────────────────────────────

def _after_route_intent(state: NL2SQLState) -> str:
    if state.get("intent") == "analytics":
        return "generate_sql"
    return "handle_fallback"


def _after_generate_sql(state: NL2SQLState) -> str:
    """If generation itself failed (API error), go to fallback immediately."""
    if state.get("failure_stage") == "generation":
        return "handle_fallback"
    return "validate_sql"


def _after_validate_sql(state: NL2SQLState) -> str:
    if state.get("is_valid"):
        return "execute_sql"

    # Check if this validation is in a repair cycle or a generation retry cycle
    repair_attempts = state.get("exec_repair_attempts", 0)
    if repair_attempts > 0:
        # We're in a repair cycle — limit repair retries
        if repair_attempts >= params.pipeline.max_exec_repair_attempts:
            logger.warning(
                f"Execution repair exhausted after {repair_attempts} repair(s) + re-validation"
            )
            return "handle_fallback"
        # Try to repair again
        return "repair_sql"

    # Normal generation cycle
    sql_attempts = state.get("sql_attempts", 0)
    if sql_attempts >= params.pipeline.max_retry_attempts:
        logger.warning(f"SQL generation exhausted after {sql_attempts} attempt(s)")
        return "handle_fallback"
    return "generate_sql"


def _after_execute_sql(state: NL2SQLState) -> str:
    if not state.get("exec_error"):
        return "interpret_result"

    error = state.get("exec_error", "")
    repair_attempts = state.get("exec_repair_attempts", 0)

    if repair_attempts >= params.pipeline.max_exec_repair_attempts:
        logger.warning(
            f"SQL execution failed — repair budget exhausted "
            f"({repair_attempts}/{params.pipeline.max_exec_repair_attempts})"
        )
        return "handle_fallback"

    if _is_repairable_error(error):
        logger.info(
            f"Repairable execution error — attempting repair "
            f"({repair_attempts + 1}/{params.pipeline.max_exec_repair_attempts}): "
            f"{error[:80]}"
        )
        return "repair_sql"

    logger.warning(f"Non-repairable execution error → fallback directly: {error[:80]}")
    return "handle_fallback"


# ── Graph construction ─────────────────────────────────────────────────────────

def _build_graph() -> Any:
    """Build and compile the full-retry LangGraph StateGraph."""
    graph = StateGraph(NL2SQLState)

    # Register nodes
    graph.add_node("route_intent",      route_intent_node)
    graph.add_node("generate_sql",      generate_sql_node)
    graph.add_node("validate_sql",      validate_sql_node)
    graph.add_node("execute_sql",       execute_sql_node)
    graph.add_node("repair_sql",        repair_sql_node)
    graph.add_node("interpret_result",  interpret_result_node)
    graph.add_node("handle_fallback",   handle_fallback_node)

    # Entry point
    graph.set_entry_point("route_intent")

    # Edges
    graph.add_conditional_edges(
        "route_intent",
        _after_route_intent,
        {"generate_sql": "generate_sql", "handle_fallback": "handle_fallback"},
    )
    graph.add_conditional_edges(
        "generate_sql",
        _after_generate_sql,
        {"validate_sql": "validate_sql", "handle_fallback": "handle_fallback"},
    )
    graph.add_conditional_edges(
        "validate_sql",
        _after_validate_sql,
        {
            "execute_sql":    "execute_sql",
            "generate_sql":   "generate_sql",
            "repair_sql":     "repair_sql",
            "handle_fallback":"handle_fallback",
        },
    )
    graph.add_conditional_edges(
        "execute_sql",
        _after_execute_sql,
        {
            "interpret_result": "interpret_result",
            "repair_sql":       "repair_sql",
            "handle_fallback":  "handle_fallback",
        },
    )
    # repair_sql always feeds back into validate_sql (not directly to execute_sql)
    graph.add_edge("repair_sql",       "validate_sql")
    graph.add_edge("interpret_result", END)
    graph.add_edge("handle_fallback",  END)

    compiled = graph.compile()
    logger.info("LangGraph NL2SQL pipeline compiled ✅ (targeted retry: validation + backup model + repairable exec)")
    return compiled


# Compiled graph singleton — reused across all run_pipeline() calls
nl2sql_app = _build_graph()


# ── Public entry point ─────────────────────────────────────────────────────────

def run_pipeline(query: str) -> NL2SQLState:
    """
    Run the full NL2SQL pipeline for one user query.

    Args:
        query: Natural language question from the user.

    Returns:
        Final NL2SQLState dict. Key fields:

          Successful query:
            state["success"]          → True
            state["sql"]              → Generated/repaired SQL
            state["insight"]          → Natural language insight
            state["chart_spec"]       → {type, title, x_column, y_column}
            state["exec_rows"]        → Raw result rows
            state["exec_row_count"]   → Row count
            state["backup_model_used"]→ True if backup model was activated

          Failed query:
            state["success"]          → False
            state["fallback_message"] → What went wrong (user-facing)
            state["fallback_hint"]    → What to do differently
            state["fallback_suggestions"] → Example queries to try
            state["failure_stage"]    → "routing"|"generation"|"validation"|"execution"

          Both cases:
            state["sql_attempts"]         → How many generation attempts
            state["exec_repair_attempts"] → How many execution repairs
            state["total_latency_ms"]     → End-to-end time
            state["trace_file"]           → Path to saved JSON trace
    """
    session_id = f"q-{uuid.uuid4().hex[:8]}"   # short human-readable ID for logs/JSON
    lf_trace_id = uuid.uuid4().hex             # 32-char hex required by LangFuse v4 OTel
    logger.info(f"Pipeline START — session={session_id} | query='{query[:80]}'")

    # Set active trace ID so all LLM calls in this pipeline run are grouped
    # under one LangFuse trace. Must be 32 lowercase hex chars (OTel format).
    set_active_trace_id(lf_trace_id)

    initial = make_initial_state(query, session_id)
    t0 = time.perf_counter()

    try:
        final_state: NL2SQLState = nl2sql_app.invoke(initial)
    except Exception as exc:
        logger.exception(f"Unhandled pipeline error: {exc}")
        final_state = {
            **initial,
            "success": False,
            "failure_stage": "unknown",
            "failure_category": "unknown",
            "error": str(exc),
            "fallback_message": "An unexpected error occurred. Please try again.",
            "fallback_hint": "If the error persists, check logs/ for details.",
            "fallback_suggestions": _get_suggestions(),
        }

    final_state["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Save trace (local JSON + optional LangFuse)
    tracer = Tracer(session_id=session_id)
    trace_file = tracer.save_from_state(final_state, query)
    tracer.flush()
    final_state["trace_file"] = trace_file

    status = "✅ success" if final_state.get("success") else "❌ fallback"
    logger.info(
        f"Pipeline END [{status}] — "
        f"{final_state['total_latency_ms']:.0f}ms | "
        f"gen_attempts={final_state.get('sql_attempts', 0)} | "
        f"repair_attempts={final_state.get('exec_repair_attempts', 0)} | "
        f"backup={'yes' if final_state.get('backup_model_used') else 'no'}"
    )

    return final_state


# ── Internal helpers ───────────────────────────────────────────────────────────

# SQLite runtime errors that a surgical SQL edit can fix.
# Non-repairable errors (e.g. "no such table") need a full regeneration, not a patch,
# so we send those straight to handle_fallback to avoid wasting an LLM call.
_REPAIRABLE_PATTERNS = [
    "ambiguous column name",                    # qualify with table alias: p.patient_id
    "no such column",                           # column name typo — fix exact name
    "datatype mismatch",                        # add CAST(col AS type)
    "aggregate functions not allowed in where", # move aggregate to HAVING clause
    "misuse of aggregate",                      # aggregate used outside SELECT/HAVING
]


def _is_repairable_error(error: str) -> bool:
    """Return True if the SQLite error is likely fixable with a minimal SQL edit."""
    err_lower = error.lower()
    return any(pattern in err_lower for pattern in _REPAIRABLE_PATTERNS)


def _get_schema_safe() -> dict:
    """Return schema dict without raising — used in handle_fallback_node."""
    try:
        return get_schema_dict()
    except Exception:
        return {}


def _get_suggestions() -> list:
    """Return schema-verified suggestions for use in fallback paths."""
    result = _fallback.handle(schema_dict=_get_schema_safe())
    return result.suggestions
