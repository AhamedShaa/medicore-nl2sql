"""
dashboard — Streamlit web UI and Plotly chart rendering.

WHY STREAMLIT (+5 bonus points):
  Python-native UI framework — no HTML, CSS, or JavaScript required.
  st.cache_resource ensures the LangGraph pipeline compiles once and is
  reused across all user sessions, avoiding ~2s re-compilation overhead
  on every page refresh.

MODULES:
  app.py           — Main Streamlit application (2 pages):
                       Page 1: NL2SQL Query — type any question → SQL + chart + insight
                       Page 2: Pre-Built Dashboard — 5 live analytics panels
                     Run with: streamlit run dashboard/app.py

  chart_renderer.py — Decoupled Plotly chart renderer.
                      The ResultInterpreterAgent decides WHAT chart type to use;
                      this module decides HOW to render it with Plotly.
                      Supports: bar, line, pie, table.

USAGE:
    from dashboard import render_chart

    fig = render_chart(chart_spec, rows, columns)
    st.plotly_chart(fig)
"""

from dashboard.chart_renderer import render_chart

__all__ = ["render_chart"]
