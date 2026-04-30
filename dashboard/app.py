"""
dashboard/app.py — MediCore NL2SQL Streamlit Dashboard (+5 bonus pts).

HOW TO RUN:
    streamlit run dashboard/app.py

WHY STREAMLIT:
  Python-native UI framework — no HTML, CSS, or JavaScript needed.
  Ideal for data apps and demos. The entire UI is plain Python.
  st.cache_resource ensures the LangGraph pipeline is compiled once
  and reused across all user sessions (not re-compiled per request).

PAGES:
  1. NL2SQL Query  — type any question, get SQL + chart + insight
  2. Pre-Built Dashboard — 5 live panels with direct SQL

PREBUILT PANEL QUERIES come from params.yaml (dashboard.prebuilt_queries)
so they can be changed without touching this file.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is importable when running from the dashboard/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import params, validate
from core.graph import run_pipeline
from dashboard.chart_renderer import render_chart

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MediCore Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .insight-box {
        background: #f0f9ff;
        border-left: 4px solid #2563eb;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        color: #1e3a5f;
    }
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
        margin: 4px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
    }
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Startup validation ────────────────────────────────────────────────────────

@st.cache_resource
def _check_config() -> bool:
    """Validate API key at startup — cached so it only runs once."""
    try:
        validate()
        return True
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        return False


def _get_db_conn():
    return sqlite3.connect(params.database.path)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("MediCore Analytics")
    st.caption("AI-Powered Hospital Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔍 NL2SQL Query", "📊 Pre-Built Dashboard"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Dashboard filters (only shown on dashboard page) ──────────────────────
    if page == "📊 Pre-Built Dashboard":
        st.subheader("Filters")

        selected_year = st.selectbox(
            "Year",
            options=["All Time", "2022", "2023", "2024"],
            index=0,
        )

        @st.cache_data(ttl=600)
        def _load_departments() -> list:
            with sqlite3.connect(params.database.path) as c:
                df = pd.read_sql_query(
                    "SELECT department_name FROM departments ORDER BY department_name", c
                )
            return df["department_name"].tolist()

        all_depts = _load_departments()
        selected_depts = st.multiselect(
            "Departments",
            options=all_depts,
            default=[],
            placeholder="All departments",
        )

        st.divider()

    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    if st.session_state.query_history:
        st.subheader("Recent Queries")
        max_hist = params.dashboard.max_query_history
        for q in reversed(st.session_state.query_history[-5:]):
            label = q[:40] + "…" if len(q) > 40 else q
            if st.button(label, key=f"hist_{q[:20]}"):
                st.session_state.prefill_query = q

    st.divider()
    st.caption("Read-only | SELECT only | Data stays local")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: NL2SQL Query
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🔍 NL2SQL Query":
    st.title("Ask Your Hospital Data")
    st.caption("Type any question in plain English — the AI generates and runs the SQL.")

    if not _check_config():
        st.stop()

    # Example questions from params.yaml
    with st.expander("Example questions", expanded=False):
        examples = [q.query for q in params.dashboard.prebuilt_queries] + [
            "Which patients have the most admissions?",
            "Show doctors by specialty with patient counts",
            "List unpaid invoices over $1000",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.prefill_query = ex

    # Query input
    prefill = st.session_state.get("prefill_query", "")
    query = st.text_input(
        "Your question:",
        value=prefill,
        placeholder="e.g. What are the top 5 revenue-generating departments?",
    )

    col_run, col_clear = st.columns([4, 1])
    run_clicked = col_run.button("Run Query", type="primary", use_container_width=True)
    if col_clear.button("Clear", use_container_width=True):
        st.session_state.prefill_query = ""

    if run_clicked and query.strip():
        with st.spinner("Agents working…"):
            state = run_pipeline(query.strip())

        st.session_state.query_history.append(query.strip())
        st.session_state.pop("prefill_query", None)

        if state["success"]:
            st.success(
                f"Success — {state['exec_row_count']} rows in "
                f"{state['total_latency_ms']:.0f}ms "
                f"({state['sql_attempts']} attempt(s))"
            )

            st.markdown(
                f'<div class="insight-box">💡 <strong>Insight:</strong> {state["insight"]}</div>',
                unsafe_allow_html=True,
            )

            fig = render_chart(
                state["chart_spec"],
                state["exec_rows"],
                state["exec_columns"],
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("View Raw Data"):
                df = pd.DataFrame(state["exec_rows"], columns=state["exec_columns"])
                st.dataframe(df, use_container_width=True)

            with st.expander("Generated SQL"):
                st.code(state["sql"], language="sql")

            if state.get("trace_file"):
                with st.expander("Execution Trace"):
                    try:
                        with open(state["trace_file"]) as f:
                            trace = json.load(f)
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total Latency", f"{trace.get('total_latency_ms', 0):.0f} ms")
                        col_b.metric("SQL Attempts", trace.get("sql_attempts", 0))
                        col_c.metric("Rows Returned", trace.get("row_count", 0))
                        st.caption(
                            f"Intent: `{trace.get('intent', '—')}`  |  "
                            f"Session: `{trace.get('session_id', '—')}`"
                        )
                        if state.get("backup_model_used"):
                            st.warning("Backup model (GPT-4o-mini) was used for this query.")
                    except Exception:
                        st.info(f"Trace saved at: {state['trace_file']}")

        else:
            stage = state.get("failure_stage", "")
            st.error(f"Could not process this query. [{stage}]" if stage else "Could not process this query.")
            st.warning(state.get("fallback_message", ""))
            hint = state.get("fallback_hint", "")
            if hint:
                st.info(f"💡 **Hint:** {hint}")

            suggestions = state.get("fallback_suggestions", [])
            if suggestions:
                st.subheader("Try one of these instead:")
                cols = st.columns(2)
                for i, sug in enumerate(suggestions[:6]):
                    if cols[i % 2].button(sug, key=f"sug_{i}"):
                        st.session_state.prefill_query = sug
                        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Pre-Built Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Pre-Built Dashboard":

    @st.cache_data(ttl=300)
    def _q(sql: str) -> pd.DataFrame:
        """Run a read-only SQL query and cache the result for 5 minutes."""
        with sqlite3.connect(params.database.path) as c:
            return pd.read_sql_query(sql, c)

    # ── Build WHERE snippets from sidebar filters ─────────────────────────────
    # year_w  — appended to queries on date columns
    # dept_in — appended to queries that join departments
    year_w = (
        f" AND strftime('%Y', {{col}}) = '{selected_year}'"
        if selected_year != "All Time" else ""
    )
    if selected_depts:
        dept_list = ", ".join(f"'{d}'" for d in selected_depts)
        dept_in = f" AND dep.department_name IN ({dept_list})"
    else:
        dept_in = ""

    # Helper: format year_w with the right column name
    def _yw(col: str) -> str:
        return year_w.format(col=col)

    _PLOT_LAYOUT = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
    )

    # ── Dashboard header ──────────────────────────────────────────────────────
    hcol, rcol = st.columns([5, 1])
    hcol.title("MediCore Hospital Analytics")
    hcol.caption("Live hospital intelligence · data through Jan 2025 · refreshes every 5 min")
    if rcol.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    try:
        # ── Section 0: KPIs ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)

        total_patients = _q("SELECT COUNT(*) AS n FROM patients").iloc[0, 0]
        k1.metric("Total Patients", f"{total_patients:,}")

        total_rev = _q(
            f"SELECT COALESCE(SUM(total_amount), 0) AS r FROM billing_invoices WHERE 1=1{_yw('invoice_date')}"
        ).iloc[0, 0]
        k2.metric("Total Revenue", f"LKR {total_rev / 1e9:.2f}B")

        inpatients = _q(
            f"SELECT COUNT(*) AS n FROM admissions WHERE discharge_date IS NULL{_yw('admission_date')}"
        ).iloc[0, 0]
        k3.metric("Current Inpatients", f"{inpatients:,}")

        ns_rate = _q(
            f"SELECT ROUND(SUM(CASE WHEN status='No-Show' THEN 1.0 ELSE 0 END) * 100.0 / COUNT(*), 1) AS r "
            f"FROM appointments WHERE 1=1{_yw('appointment_date')}"
        ).iloc[0, 0]
        k4.metric("No-Show Rate", f"{ns_rate}%")

        total_appts = _q(
            f"SELECT COUNT(*) AS n FROM appointments WHERE 1=1{_yw('appointment_date')}"
        ).iloc[0, 0]
        k5.metric("Total Appointments", f"{total_appts:,}")

        st.divider()

        # ── Section 1: Financial Overview ─────────────────────────────────────
        st.markdown('<div class="section-header">💰 Financial Overview</div>', unsafe_allow_html=True)

        rev_df = _q(
            f"SELECT strftime('%Y-%m', invoice_date) AS month, SUM(total_amount) AS revenue "
            f"FROM billing_invoices WHERE 1=1{_yw('invoice_date')} GROUP BY month ORDER BY month"
        )
        if not rev_df.empty:
            fig = px.area(
                rev_df, x="month", y="revenue",
                title="Monthly Revenue Trend (All Time)",
                color_discrete_sequence=["#2563eb"],
            )
            fig.update_traces(fill="tozeroy", fillcolor="rgba(37,99,235,0.08)")
            fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (LKR)",
                              showlegend=False, **_PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        fc1, fc2 = st.columns(2)

        dep_rev_df = _q(
            f"SELECT dep.department_name, SUM(bi.total_amount) AS revenue "
            f"FROM billing_invoices bi "
            f"JOIN admissions adm ON bi.admission_id = adm.admission_id "
            f"JOIN departments dep ON adm.department_id = dep.department_id "
            f"WHERE 1=1{_yw('bi.invoice_date')}{dept_in} "
            f"GROUP BY dep.department_name ORDER BY revenue DESC LIMIT 10"
        )
        if not dep_rev_df.empty:
            fig = px.bar(
                dep_rev_df, x="revenue", y="department_name",
                orientation="h", title="Revenue by Department (Top 10)",
                color_discrete_sequence=["#2563eb"],
            )
            fig.update_layout(yaxis_title="", xaxis_title="Revenue (LKR)", **_PLOT_LAYOUT)
            fc1.plotly_chart(fig, use_container_width=True)

        pay_df = _q(
            f"SELECT payment_method, SUM(amount) AS total "
            f"FROM payments WHERE 1=1{_yw('payment_date')} GROUP BY payment_method ORDER BY total DESC"
        )
        if not pay_df.empty:
            fig = px.pie(
                pay_df, names="payment_method", values="total",
                title="Revenue by Payment Method",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.35,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(**_PLOT_LAYOUT)
            fc2.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Section 2: Patient Operations ─────────────────────────────────────
        st.markdown('<div class="section-header">🏥 Patient Operations</div>', unsafe_allow_html=True)

        adm_df = _q(
            f"SELECT strftime('%Y-%m', admission_date) AS month, COUNT(*) AS admissions "
            f"FROM admissions WHERE 1=1{_yw('admission_date')} GROUP BY month ORDER BY month"
        )
        if not adm_df.empty:
            fig = px.bar(
                adm_df, x="month", y="admissions",
                title="Monthly Hospital Admissions (All Time)",
                color_discrete_sequence=["#059669"],
            )
            fig.update_layout(xaxis_title="Month", yaxis_title="Admissions", **_PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        pc1, pc2 = st.columns(2)

        status_df = _q(
            f"SELECT status, COUNT(*) AS count FROM appointments "
            f"WHERE 1=1{_yw('appointment_date')} GROUP BY status ORDER BY count DESC"
        )
        if not status_df.empty:
            fig = px.pie(
                status_df, names="status", values="count",
                title="Appointment Status Breakdown",
                color_discrete_sequence=["#059669", "#2563eb", "#dc2626", "#f59e0b"],
                hole=0.4,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(**_PLOT_LAYOUT)
            pc1.plotly_chart(fig, use_container_width=True)

        doc_df = _q(
            f"SELECT d.first_name || ' ' || d.last_name AS doctor, "
            f"COUNT(a.appointment_id) AS appointments "
            f"FROM doctors d "
            f"LEFT JOIN appointments a ON d.doctor_id = a.doctor_id "
            f"WHERE 1=1{_yw('a.appointment_date')} "
            f"GROUP BY d.doctor_id ORDER BY appointments DESC LIMIT 10"
        )
        if not doc_df.empty:
            fig = px.bar(
                doc_df, x="appointments", y="doctor",
                orientation="h", title="Top 10 Busiest Doctors",
                color_discrete_sequence=["#0891b2"],
            )
            fig.update_layout(yaxis_title="", xaxis_title="Appointments", **_PLOT_LAYOUT)
            pc2.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Section 3: Clinical Intelligence ──────────────────────────────────
        st.markdown('<div class="section-header">🩺 Clinical Intelligence</div>', unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)

        diag_df = _q(
            f"SELECT diagnosis_description, COUNT(*) AS count "
            f"FROM diagnoses WHERE 1=1{_yw('diagnosis_date')} "
            f"GROUP BY diagnosis_description ORDER BY count DESC LIMIT 10"
        )
        if not diag_df.empty:
            fig = px.bar(
                diag_df, x="count", y="diagnosis_description",
                orientation="h", title="Top 10 Diagnoses",
                color_discrete_sequence=["#7c3aed"],
            )
            fig.update_layout(yaxis_title="", xaxis_title="Patient Count", **_PLOT_LAYOUT)
            cc1.plotly_chart(fig, use_container_width=True)

        noshow_df = _q(
            f"SELECT d.first_name || ' ' || d.last_name AS doctor, COUNT(*) AS no_shows "
            f"FROM appointments a "
            f"JOIN doctors d ON a.doctor_id = d.doctor_id "
            f"WHERE a.status = 'No-Show'{_yw('a.appointment_date')} "
            f"GROUP BY a.doctor_id ORDER BY no_shows DESC LIMIT 10"
        )
        if not noshow_df.empty:
            fig = px.bar(
                noshow_df, x="no_shows", y="doctor",
                orientation="h", title="Top 10 Doctors by No-Show Count",
                color_discrete_sequence=["#dc2626"],
            )
            fig.update_layout(yaxis_title="", xaxis_title="No-Shows", **_PLOT_LAYOUT)
            cc2.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Dashboard error: {exc}")
        st.info(
            f"Make sure the database exists at '{params.database.path}'.\n"
            "Run: python -m core.db_setup --sql medicore_data.sql"
        )
