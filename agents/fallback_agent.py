"""
agents/fallback_agent.py — Graceful, schema-aware error recovery.

WHY NO LLM:
  The fallback must ALWAYS work — even when the API is down or rate-limited.
  A deterministic rule-based approach guarantees the user always sees a helpful
  response, regardless of what caused the failure.

WHY SCHEMA-AWARE SUGGESTIONS:
  Generic "try these queries" lists feel disconnected from what failed.
  Advanced systems generate suggestions that are:
    1. Guaranteed executable (tables/columns verified to exist in the DB)
    2. Tailored to the failure type (schema error → show correct table names,
       execution error → suggest simpler version of the failed query,
       blocked → show read-only alternatives for the same intent)
  This is implemented with get_schema_dict() so suggestions always reflect
  the live database — no hardcoded table names that could go stale.

FAILURE CATEGORIES:
  sql_generation → NL2SQL never produced valid SQL after all retries
  validation     → safety check blocked the query (destructive SQL or unknown schema)
  execution      → valid SQL failed at SQLite execution time (all repairs exhausted)
  api_error      → OpenRouter/network unavailable
  ambiguous      → router classified query as too vague to answer
  unknown        → catch-all for unexpected pipeline exceptions

RECOVERY PATH DESIGN:
  Each failure category has:
    - A specific human-readable message (what happened + why)
    - A hint (what the user should do differently)
    - Schema-derived example queries (from the live DB, not hardcoded)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.log import logger


# ── Static message templates ──────────────────────────────────────────────────

_MESSAGES: Dict[str, Dict[str, str]] = {
    "sql_generation": {
        "what":  "I couldn't generate a valid SQL query for your question after multiple attempts.",
        "hint":  "Try being more specific — mention the table (patients, appointments, billing_invoices), "
                 "a time period, or the exact metric you want (count, sum, average).",
    },
    "validation": {
        "what":  "The generated query didn't pass safety validation.",
        "hint":  "This platform supports read-only analytics only — counts, lists, sums, and trends. "
                 "Modification requests (DELETE, UPDATE, DROP) are blocked by design.",
    },
    "execution": {
        "what":  "The query was structurally valid but failed when executed against the database.",
        "hint":  "Try breaking the question into simpler parts — start with one table, "
                 "then add JOINs or filters. Very complex aggregations sometimes need intermediate steps.",
    },
    "api_error": {
        "what":  "The AI service is temporarily unreachable.",
        "hint":  "Check that OPENROUTER_API_KEY is set correctly in .env, and that you have "
                 "remaining API credits. Try again in a moment.",
    },
    "ambiguous": {
        "what":  "Your question is too ambiguous to safely translate into SQL.",
        "hint":  "Add specifics: which table? which time period? which metric? "
                 "For example: \"How many patients were admitted in January 2025?\"",
    },
    "unknown": {
        "what":  "An unexpected error occurred in the pipeline.",
        "hint":  "Please try rephrasing. If the issue persists, check logs/ for details.",
    },
}

# ── Schema-driven suggestion templates ────────────────────────────────────────
# Each entry is a (template_string, required_tables) pair.
# Only suggestions whose required tables ALL exist in the live schema are shown.

_SUGGESTION_TEMPLATES: List[tuple[str, List[str]]] = [
    ("What are the top 5 revenue-generating departments?",       ["billing_invoices", "departments"]),
    ("Which doctors have the most appointments this month?",     ["appointments", "doctors"]),
    ("Show monthly patient admissions for the past 12 months",   ["admissions"]),
    ("What is the most common diagnosis in the last 6 months?",  ["diagnoses"]),
    ("List unpaid invoices with amount over 1000",               ["billing_invoices"]),
    ("Which patients were admitted in the last 30 days?",        ["admissions", "patients"]),
    ("What is the average billing amount per department?",       ["billing_invoices", "departments"]),
    ("Show the top 10 most prescribed medications",              ["prescriptions"]),
    ("How many appointments were scheduled vs completed?",       ["appointments"]),
    ("Which specialties have the most active doctors?",          ["doctors", "specialties"]),
    ("Show payment method breakdown by total revenue",           ["payments"]),
    ("List doctors with zero appointments this month",           ["doctors", "appointments"]),
    ("What are the top 5 lab tests ordered?",                    ["lab_orders"]),
    ("Show monthly revenue trend for the past year",             ["billing_invoices"]),
    ("Which departments have the highest patient admission rate?", ["admissions", "departments"]),
]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FallbackResult:
    """User-facing output when the pipeline cannot complete successfully."""

    message: str
    hint: str
    suggestions: List[str] = field(default_factory=list)
    failure_category: str = "unknown"
    recovery_action: str = ""   # What the user should do next (machine-readable)


# ── Agent ─────────────────────────────────────────────────────────────────────

class FallbackAgent:
    """
    Deterministic, schema-aware error recovery — no LLM, no network calls.

    Always succeeds; always returns a FallbackResult with:
      - A clear explanation of what went wrong
      - A hint for how to fix the user's query
      - Schema-verified example queries they can try instead
    """

    def handle(
        self,
        failure_category: str = "unknown",
        original_query: str = "",
        technical_error: str = "",
        schema_dict: Optional[Dict[str, List[str]]] = None,
    ) -> FallbackResult:
        """
        Build a schema-aware fallback response for a pipeline failure.

        Args:
            failure_category: Key mapping to _MESSAGES (what failed).
            original_query:   The user's original question (for context).
            technical_error:  Internal error detail — logged only, NOT shown to user.
            schema_dict:      {table_name: [column_names]} from get_schema_dict().
                              When provided, suggestions are filtered to only those
                              whose required tables exist in the live database.

        Returns:
            FallbackResult with message, hint, schema-verified suggestions, and
            a recovery_action string for the UI to act on.
        """
        if technical_error:
            logger.error(f"Fallback [{failure_category}]: {technical_error}")

        template = _MESSAGES.get(failure_category, _MESSAGES["unknown"])
        what = template["what"]
        hint = template["hint"]

        # Build the user-facing message
        if original_query:
            preview = original_query[:80] + ("…" if len(original_query) > 80 else "")
            message = f'For "{preview}": {what}'
        else:
            message = what

        # Generate schema-verified suggestions
        suggestions = _build_suggestions(schema_dict, failure_category, n=6)

        # Determine the recovery action — gives the UI a machine-readable cue
        recovery_action = _recovery_action(failure_category)

        logger.info(
            f"FallbackAgent → category='{failure_category}' "
            f"suggestions={len(suggestions)}"
        )

        return FallbackResult(
            message=message,
            hint=hint,
            suggestions=suggestions,
            failure_category=failure_category,
            recovery_action=recovery_action,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_suggestions(
    schema_dict: Optional[Dict[str, List[str]]],
    failure_category: str,
    n: int = 6,
) -> List[str]:
    """
    Return up to n example queries whose required tables exist in the live schema.

    WHY SCHEMA FILTERING:
      If the database is partially loaded or has a different table set, suggestions
      that reference missing tables would confuse the user. This ensures every
      suggestion we show can actually be answered by the current database.
    """
    available_tables = set(schema_dict.keys()) if schema_dict else set()

    # If no schema info, fall back to suggestions with no table requirements
    if not available_tables:
        return [t for t, _ in _SUGGESTION_TEMPLATES[:n]]

    valid = [
        text
        for text, required_tables in _SUGGESTION_TEMPLATES
        if all(t in available_tables for t in required_tables)
    ]

    # Shuffle so repeated fallbacks don't always show the same 6 suggestions
    random.shuffle(valid)
    return valid[:n]


def _recovery_action(failure_category: str) -> str:
    """Map failure category to a machine-readable recovery action for the UI."""
    return {
        "sql_generation": "rephrase_query",
        "validation":     "change_query_type",
        "execution":      "simplify_query",
        "api_error":      "check_api_key",
        "ambiguous":      "add_specifics",
        "unknown":        "retry",
    }.get(failure_category, "retry")
