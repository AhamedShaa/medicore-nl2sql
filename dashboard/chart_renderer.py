"""
dashboard/chart_renderer.py — Render Plotly charts from chart specs.

WHY DECOUPLED FROM THE INTERPRETER:
  ResultInterpreterAgent decides WHAT to show (ChartSpec).
  This module decides HOW to show it (Plotly figure).
  Keeping them separate means chart styling can change without
  touching the LLM prompt, and vice versa.

CHART TYPES SUPPORTED:
  bar   → px.bar (horizontal or vertical based on data shape)
  line  → px.line with markers
  pie   → px.pie with percentage labels
  table → go.Table with alternating row colours

FALLBACK:
  If any chart render fails (missing column, wrong type), the function
  automatically falls back to a plain data table. The user always sees data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.log import logger


def render_chart(
    chart_spec: Optional[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    columns: List[str],
) -> Optional[go.Figure]:
    """
    Create a Plotly figure from a chart spec dict and result rows.

    Args:
        chart_spec: Dict from NL2SQLState["chart_spec"] — keys: type, title,
                    x_column, y_column, label_column, value_column.
        rows:       List of row dicts from the SQL executor.
        columns:    Column names in order.

    Returns:
        A Plotly Figure, or None if rows is empty.
    """
    if not rows or not chart_spec:
        return None

    df = pd.DataFrame(rows, columns=columns if columns else None)
    chart_type = (chart_spec.get("type") or "table").lower()

    try:
        if chart_type == "bar":
            return _bar_chart(df, chart_spec)
        elif chart_type == "line":
            return _line_chart(df, chart_spec)
        elif chart_type == "pie":
            return _pie_chart(df, chart_spec)
        else:
            return _data_table(df, chart_spec)

    except Exception as exc:
        logger.error(f"Chart render failed ({chart_type}): {exc} — falling back to table")
        return _data_table(df, chart_spec)


# ── Chart builders ────────────────────────────────────────────────────────────

def _bar_chart(df: pd.DataFrame, spec: Dict[str, Any]) -> go.Figure:
    x_col = spec.get("x_column") or df.columns[0]
    y_col = spec.get("y_column") or (df.columns[1] if len(df.columns) > 1 else df.columns[0])

    # Use horizontal bars when x-axis has long string labels
    orientation = "h" if df[x_col].dtype == object and len(df) > 5 else "v"

    if orientation == "h":
        fig = px.bar(
            df, x=y_col, y=x_col,
            orientation="h",
            title=spec.get("title", ""),
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_layout(
            xaxis_title=_label(y_col),
            yaxis_title="",
        )
    else:
        fig = px.bar(
            df, x=x_col, y=y_col,
            title=spec.get("title", ""),
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_layout(
            xaxis_title=_label(x_col),
            yaxis_title=_label(y_col),
        )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _line_chart(df: pd.DataFrame, spec: Dict[str, Any]) -> go.Figure:
    x_col = spec.get("x_column") or df.columns[0]
    y_col = spec.get("y_column") or (df.columns[1] if len(df.columns) > 1 else df.columns[0])

    fig = px.line(
        df, x=x_col, y=y_col,
        title=spec.get("title", ""),
        markers=True,
        color_discrete_sequence=["#2563eb"],
    )
    fig.update_layout(
        xaxis_title=_label(x_col),
        yaxis_title=_label(y_col),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _pie_chart(df: pd.DataFrame, spec: Dict[str, Any]) -> go.Figure:
    label_col = spec.get("label_column") or df.columns[0]
    value_col = spec.get("value_column") or (
        df.columns[1] if len(df.columns) > 1 else df.columns[0]
    )

    fig = px.pie(
        df,
        names=label_col,
        values=value_col,
        title=spec.get("title", ""),
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def _data_table(df: pd.DataFrame, spec: Dict[str, Any]) -> go.Figure:
    """Fallback: render any data as a Plotly table."""
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[_label(c) for c in df.columns],
                    fill_color="#1e40af",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[df[col].tolist() for col in df.columns],
                    fill_color=[
                        ["#f0f4ff" if i % 2 == 0 else "#ffffff" for i in range(len(df))]
                    ] * len(df.columns),
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )
    fig.update_layout(
        title=spec.get("title", "Results"),
        margin=dict(t=50, b=10),
        height=min(700, max(250, len(df) * 30 + 80)),
    )
    return fig


def _label(column_name: str) -> str:
    """Convert snake_case column name to Title Case for axis labels."""
    return column_name.replace("_", " ").title()
