"""
core/tracer.py — Observability for the MediCore NL2SQL pipeline.

TWO TRACE MODES (both always active):

  1. LOCAL JSON (always works, no account needed):
     After every pipeline run, a JSON file is written to traces/.
     File name: traces/trace_{session_id}.json
     Contents: query, success, total_latency_ms, per-node timing, SQL, insight.
     WHY KEEP LOCAL JSON: Works offline, no account required, useful for debugging
     without a browser. A permanent audit trail of every query run.

  2. LANGFUSE CLOUD (optional — set keys in .env):
     When LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set, every LLM call
     (SQL generator, router, interpreter) is automatically traced via the
     LangChain CallbackHandler in agents/base_agent.py.
     You'll see: prompt sent, model response, token count, latency — per call.
     Also the overall pipeline result is logged as trace metadata.

     Setup: cloud.langfuse.com → create project → Settings → API Keys
     Then add to .env:
       LANGFUSE_PUBLIC_KEY=pk-lf-...
       LANGFUSE_SECRET_KEY=sk-lf-...

LANGFUSE v4 API NOTES:
  langfuse>=4.0 moved to an OpenTelemetry-based model. The old v2/v3 API
  (lf.trace(), trace.span()) no longer exists. The new patterns are:
    - CallbackHandler  — auto-instruments LangChain chains (used in base_agent.py)
    - start_observation() — manual spans for non-LLM steps
    - set_current_trace_io() — set trace-level input/output metadata

GRACEFUL DEGRADATION:
  If LangFuse keys are missing or the SDK throws, tracing continues locally.
  The pipeline never fails due to tracing issues.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import params
from utils.log import logger


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class NodeTrace:
    """Trace record for one LangGraph node execution."""

    node_name: str
    input_summary: str
    output_summary: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Thread-local trace context ────────────────────────────────────────────────
# Stores the current pipeline session_id so agents can group their LLM calls
# under the same LangFuse trace without needing trace_id in every method sig.

import threading as _threading
_trace_ctx = _threading.local()


def set_active_trace_id(trace_id: str) -> None:
    """Set the LangFuse trace ID for the current thread (called by run_pipeline)."""
    _trace_ctx.trace_id = trace_id


def get_active_trace_id() -> Optional[str]:
    """Return the active trace ID for the current thread, or None."""
    return getattr(_trace_ctx, "trace_id", None)


# ── Module-level LangFuse client singleton ────────────────────────────────────
# Initialised once at first use; reused across all Tracer instances.

_lf_client = None
_lf_available = False


def _get_lf_client():
    """Return the module-level LangFuse client, creating it on first call."""
    global _lf_client, _lf_available

    if _lf_client is not None:
        return _lf_client

    pub = params.langfuse_public_key
    sec = params.langfuse_secret_key

    if not (pub and sec):
        logger.debug("LangFuse keys not set — writing traces to local files only.")
        return None

    try:
        from langfuse import Langfuse  # langfuse>=4.0

        # langfuse v4 reads these env vars automatically when Langfuse() is created.
        # We set them explicitly here in case config loaded them via python (not shell).
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", pub)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", sec)
        os.environ.setdefault("LANGFUSE_HOST", params.langfuse.host)

        _lf_client = Langfuse()
        _lf_available = True
        logger.info("LangFuse v4 client ready (LLM calls will be auto-traced via CallbackHandler)")
        return _lf_client

    except Exception as exc:
        logger.warning(f"LangFuse init failed (local tracing only): {exc}")
        return None


def get_langchain_callback_handler(trace_id: Optional[str] = None):
    """
    Return a LangFuse LangChain CallbackHandler for the given trace ID.
    Returns None if LangFuse is not available.

    Usage in agents/base_agent.py:
        handler = get_langchain_callback_handler(session_id)
        chain.invoke(inputs, config={"callbacks": [handler]})

    Each LLM call through the chain is then automatically recorded as a
    generation span in the LangFuse trace — with prompt, response, tokens.
    """
    lf = _get_lf_client()
    if lf is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        if trace_id:
            return CallbackHandler(trace_context={"trace_id": trace_id})
        return CallbackHandler()

    except Exception as exc:
        logger.debug(f"LangFuse CallbackHandler unavailable: {exc}")
        return None


# ── Tracer ────────────────────────────────────────────────────────────────────

class Tracer:
    """
    Collects per-node traces and persists them locally + optionally to LangFuse.

    Usage in core/graph.py:
        tracer = Tracer(session_id="q-abc123")
        trace_file = tracer.save_from_state(final_state, query)
        tracer.flush()

    LLM call tracing is handled automatically by the CallbackHandler
    in agents/base_agent.py — no manual log_node() calls needed for LLM nodes.
    log_node() is kept for non-LLM steps (validator, executor).
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or f"session-{int(time.time())}"
        self.nodes: List[NodeTrace] = []
        self._lf = _get_lf_client()

    # ── Logging API ───────────────────────────────────────────────────────────

    def log_node(
        self,
        node_name: str,
        input_data: Any,
        output_data: Any,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record one node execution locally and as a LangFuse span."""
        node = NodeTrace(
            node_name=node_name,
            input_summary=str(input_data)[:600],
            output_summary=str(output_data)[:600],
            latency_ms=round(latency_ms, 2),
            success=success,
            error=error,
        )
        self.nodes.append(node)

        if self._lf:
            try:
                span = self._lf.start_observation(
                    trace_context={"trace_id": self.session_id},
                    name=node_name,
                    as_type="span",
                    input={"data": node.input_summary},
                    output={"data": node.output_summary},
                    metadata={"latency_ms": latency_ms, "session_id": self.session_id},
                    level="ERROR" if not success else "DEFAULT",
                    status_message=error,
                )
                span.end()
            except Exception as exc:
                logger.debug(f"LangFuse span error: {exc}")

        status = "✅" if success else "❌"
        logger.info(
            f"[{node_name}] {status} {latency_ms:.0f}ms"
            + (f" | {error}" if error else "")
        )

    # ── State-based save ──────────────────────────────────────────────────────

    def save_from_state(self, state: Dict[str, Any], query: str) -> str:
        """
        Save a comprehensive trace from the final LangGraph state.
        Always writes to local JSON. Also updates the LangFuse trace metadata.
        """
        Path("traces").mkdir(exist_ok=True)
        filename = f"traces/trace_{self.session_id}.json"

        payload = {
            "session_id": self.session_id,
            "conversation_id": state.get("conversation_id", ""),
            "turn_id": state.get("turn_id", 0),
            "query": query,
            "resolved_query": state.get("resolved_query", query),
            "success": state.get("success", False),
            "total_latency_ms": state.get("total_latency_ms", 0),
            "intent": state.get("intent", ""),
            "sql": state.get("sql", ""),
            "sql_attempts": state.get("sql_attempts", 0),
            "exec_repair_attempts": state.get("exec_repair_attempts", 0),
            "exec_error": state.get("exec_error", ""),
            "validation_error": state.get("validation_error_message", ""),
            "row_count": state.get("exec_row_count", 0),
            "exec_time_ms": state.get("exec_time_ms", 0),
            "insight": state.get("insight", ""),
            "chart_spec": state.get("chart_spec"),
            "fallback_message": state.get("fallback_message", ""),
            "nodes": [asdict(n) for n in self.nodes],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info(f"Trace saved → {filename}")

        # Update LangFuse trace with pipeline-level metadata
        if self._lf:
            try:
                self._lf.set_current_trace_io(
                    input={"query": query},
                    output={
                        "success": state.get("success"),
                        "sql": state.get("sql", "")[:300],
                        "insight": state.get("insight", "")[:300],
                        "fallback": state.get("fallback_message", "")[:200],
                    },
                )
            except Exception:
                pass

        return filename

    def save_to_file(self, query: str = "") -> str:
        """Legacy API — used by tests. Prefer save_from_state() for new code."""
        Path("traces").mkdir(exist_ok=True)
        filename = f"traces/trace_{self.session_id}.json"

        payload = {
            "session_id": self.session_id,
            "query": query,
            "total_latency_ms": sum(n.latency_ms for n in self.nodes),
            "nodes": [asdict(n) for n in self.nodes],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info(f"Trace saved → {filename}")
        return filename

    def flush(self) -> None:
        """Flush LangFuse buffer. Call before process exits."""
        if self._lf:
            try:
                self._lf.flush()
            except Exception:
                pass
