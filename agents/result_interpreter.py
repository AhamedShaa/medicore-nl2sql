"""
agents/result_interpreter.py — Result Interpreter Agent.

ROLE:
  Converts raw SQL results (rows + columns) into:
    1. A natural language INSIGHT (2-3 sentences citing specific numbers)
    2. A CHART SPEC (type, title, axes) for the Plotly renderer
    3. A SHORT TITLE for the result card in the UI

WHY INTERPRETATION MATTERS:
  Raw database rows don't tell a story. A hospital manager asking
  "which departments generate the most revenue?" needs an answer like
  "Cardiology leads with $2.4M in Q1, accounting for 34% of total revenue"
  — not a table of numbers.

CHART SPEC DESIGN:
  The ChartSpec dataclass is decoupled from the Plotly renderer
  (dashboard/chart_renderer.py). The interpreter decides WHAT to show;
  the renderer decides HOW to show it. This separation makes both
  independently testable.

LCEL CHAIN:
  ChatPromptTemplate | ChatOpenAI | JsonOutputParser
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import BaseAgent
from core.executor import ExecutionResult
from utils.log import logger


_SYSTEM_PROMPT = """\
You are a hospital data analyst. Given SQL query results, produce:
  1. A natural language INSIGHT (2-3 sentences) citing specific numbers from the data.
  2. A CHART SPEC that tells the frontend how to visualize the data.
  3. A SHORT TITLE (max 8 words) for the result card.

Respond with ONLY a valid JSON object — no markdown, no preamble:
{{
  "insight": "2-3 sentence analysis with SPECIFIC NUMBERS from the data",
  "chart": {{
    "type": "bar" | "line" | "pie" | "table",
    "title": "Descriptive chart title",
    "x_column": "column name for x-axis (null for pie/table)",
    "y_column": "column name for y-axis (null for pie/table)",
    "label_column": "column for pie slice labels (null for non-pie)",
    "value_column": "column for pie slice sizes (null for non-pie)"
  }},
  "summary_title": "Short panel title"
}}

Chart type selection rules:
  bar   → comparisons across categories (departments, doctors, diagnoses)
  line  → time series (monthly trends, date-based data)
  pie   → proportions with ≤ 8 distinct slices
  table → detailed row data or when no clear chart applies

IMPORTANT: The insight MUST cite real numbers from the data. Never write
"the data shows X" without a specific value.
"""


@dataclass
class ChartSpec:
    """Specification for rendering a Plotly chart in the dashboard."""

    type: str                           # bar | line | pie | table
    title: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    label_column: Optional[str] = None  # pie charts: category labels
    value_column: Optional[str] = None  # pie charts: numeric values


@dataclass
class InterpretedResult:
    """Full interpreted output ready for the UI layer."""

    insight: str
    chart_spec: ChartSpec
    summary_title: str
    raw_rows: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    row_count: int = 0


class ResultInterpreterAgent(BaseAgent):
    """
    Interprets SQL results into natural language insight and a chart spec.

    Chain: prompt | GPT-4o via OpenRouter | JsonOutputParser
    """

    def __init__(self) -> None:
        super().__init__()

        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", _SYSTEM_PROMPT),
                    ("human", "{user_message}"),
                ]
            )
            | self.llm
            | JsonOutputParser()
        )

    def interpret(
        self,
        query: str,
        exec_result: ExecutionResult,
    ) -> Tuple[InterpretedResult, float]:
        """
        Generate insight and chart spec for a SQL execution result.

        Args:
            query:       The original natural language question.
            exec_result: Structured result from the SQL Executor.

        Returns:
            (InterpretedResult, latency_ms)
        """
        if not exec_result.success or exec_result.row_count == 0:
            default = _empty_result(exec_result)
            return default, 0.0

        # Limit to 10 sample rows in prompt to keep token usage bounded
        sample = exec_result.rows[:10]
        data_summary = {
            "total_rows": exec_result.row_count,
            "columns": exec_result.columns,
            "sample_data": sample,
        }

        user_message = (
            f"Original question: {query}\n\n"
            f"Query results:\n{json.dumps(data_summary, default=str, indent=2)}"
        )

        t0 = time.perf_counter()
        try:
            parsed: dict = self._chain.invoke({"user_message": user_message})
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.warning(f"Interpreter parse failed: {exc} — using default table view")
            return _default_result(exec_result), latency

        latency = (time.perf_counter() - t0) * 1000

        chart_data = parsed.get("chart", {})
        chart_spec = ChartSpec(
            type=chart_data.get("type", "table"),
            title=chart_data.get("title", "Query Results"),
            x_column=chart_data.get("x_column"),
            y_column=chart_data.get("y_column"),
            label_column=chart_data.get("label_column"),
            value_column=chart_data.get("value_column"),
        )

        logger.info(
            f"Interpreter → chart={chart_spec.type}, "
            f"rows={exec_result.row_count} ({latency:.0f}ms)"
        )

        return InterpretedResult(
            insight=parsed.get("insight", ""),
            chart_spec=chart_spec,
            summary_title=parsed.get("summary_title", "Results"),
            raw_rows=exec_result.rows,
            columns=exec_result.columns,
            row_count=exec_result.row_count,
        ), latency


# ── Fallback helpers ──────────────────────────────────────────────────────────

def _empty_result(exec_result: ExecutionResult) -> InterpretedResult:
    return InterpretedResult(
        insight="The query returned no results. The data may not exist for your filters.",
        chart_spec=ChartSpec(type="table", title="No Results"),
        summary_title="No Results",
        raw_rows=[],
        columns=[],
        row_count=0,
    )


def _default_result(exec_result: ExecutionResult) -> InterpretedResult:
    return InterpretedResult(
        insight=f"Query returned {exec_result.row_count} rows. Review the table below.",
        chart_spec=ChartSpec(type="table", title="Query Results"),
        summary_title="Query Results",
        raw_rows=exec_result.rows,
        columns=exec_result.columns,
        row_count=exec_result.row_count,
    )
