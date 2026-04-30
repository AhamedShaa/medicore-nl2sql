"""
core — Pipeline orchestration, state machine, execution, schema, and tracing.

WHY THIS PACKAGE EXISTS:
  Separates infrastructure concerns (DB connection, SQL execution, tracing) from
  agent logic (agents/). This makes it possible to swap the database or
  observability backend without touching agent prompts or LLM code.

MODULES:
  graph.py        — LangGraph StateGraph: the single authoritative pipeline.
                    Entry point: run_pipeline(query) → NL2SQLState
  state.py        — NL2SQLState TypedDict shared by all graph nodes
  executor.py     — Safe, sandboxed SQL execution against SQLite/PostgreSQL
  schema_loader.py— Dynamic schema introspection (LRU-cached)
  tracer.py       — Dual tracing: local JSON + optional LangFuse cloud
  db_setup.py     — One-time SQL dump loader (medicore_data.sql → medicore.db)

IMPORT DESIGN — WHY graph.py IS NOT IMPORTED HERE:
  core.graph instantiates all 5 agents as module-level singletons when it is
  first imported. Those agents require langchain_openai, sqlparse, and other
  heavyweight dependencies. Importing graph.py at package level would force
  every tool that touches `core` (including db_setup, schema_loader, executor)
  to pay that cost — even when no LLM is needed.

  Rule: lightweight infrastructure (state, executor, schema, tracer, db_setup)
  is imported eagerly here. The pipeline entry point must be imported explicitly:

      from core.graph import run_pipeline   ← explicit, intentional
      from core import run_pipeline         ← NOT available; prevents accidental load

USAGE:
    # Infrastructure (no agents loaded):
    from core import load_sql_file, execute_query, get_schema_string

    # Pipeline (loads all agents + LangGraph):
    from core.graph import run_pipeline
    state = run_pipeline("Which departments generate the most revenue?")
"""

from core.state import NL2SQLState, make_initial_state
from core.executor import execute_query, ExecutionResult
from core.schema_loader import get_schema_string, get_schema_dict, invalidate_cache
from core.tracer import Tracer, NodeTrace
from core.db_setup import load_sql_file

__all__ = [
    # State
    "NL2SQLState",
    "make_initial_state",
    # SQL execution
    "execute_query",
    "ExecutionResult",
    # Schema introspection
    "get_schema_string",
    "get_schema_dict",
    "invalidate_cache",
    # Observability
    "Tracer",
    "NodeTrace",
    # Database setup
    "load_sql_file",
]
