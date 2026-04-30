"""
tests/test_pipeline.py — Unit tests for the MediCore NL2SQL pipeline.

Run with:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short   # compact output

NO API KEY NEEDED:
  Tests cover ValidatorAgent and FallbackAgent — both are deterministic
  rule-based components that require no LLM calls or database connections.
  All 17 tests pass offline.

WHAT IS TESTED:
  - ValidatorAgent: keyword safety (8 tests)
  - ValidatorAgent: schema existence checks (5 tests)
  - ValidatorAgent: user-friendly error messages (2 tests)
  - FallbackAgent: all failure categories (4 tests)
  - Dangerous query regression: attack patterns (3 tests)
  - LangGraph state: make_initial_state structure (2 tests) [NEW]
  - Config: params.yaml loads correctly (1 test)           [NEW]
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.fallback_agent import FallbackAgent
from agents.validator_agent import ValidatorAgent
from config import params
from core.state import make_initial_state


# ── Mock validator (no DB required) ──────────────────────────────────────────

class MockValidatorAgent(ValidatorAgent):
    """
    ValidatorAgent with an injected schema dict — no database connection needed.
    This is the correct way to unit test the validator: inject the dependency
    (schema) rather than connecting to a real or test database.
    """

    def __init__(self) -> None:
        # Bypass DB loading — inject mock schema directly
        self._schema = {
            "patients": ["patient_id", "first_name", "last_name", "date_of_birth"],
            "doctors": ["doctor_id", "first_name", "last_name", "specialty_id"],
            "appointments": ["appointment_id", "patient_id", "doctor_id", "appointment_date"],
            "billing_invoices": ["invoice_id", "patient_id", "total_amount", "invoice_date"],
            "departments": ["department_id", "department_name"],
            "diagnoses": ["diagnosis_id", "patient_id", "diagnosis_name"],
            "payments": ["payment_id", "invoice_id", "amount", "payment_method"],
            "admissions": ["admission_id", "patient_id", "admission_date"],
            "prescriptions": ["prescription_id", "patient_id", "medication"],
            "lab_orders": ["lab_order_id", "patient_id", "test_name"],
            "specialties": ["specialty_id", "specialty_name"],
            "staff": ["staff_id", "first_name", "department_id"],
        }
        self._valid_tables = {t.lower() for t in self._schema.keys()}
        self._blocked = [kw.upper() for kw in params.safety.blocked_keywords]


# ── Validator: Keyword Safety ─────────────────────────────────────────────────

class TestValidatorSafety:
    """Layer 1: keyword-based SQL injection and destructive operation blocking."""

    def setup_method(self):
        self.validator = MockValidatorAgent()

    def test_valid_select_passes(self):
        result = self.validator.validate("SELECT * FROM patients LIMIT 10")
        assert result.valid is True

    def test_drop_blocked(self):
        result = self.validator.validate("DROP TABLE patients")
        assert result.valid is False
        assert result.technical_error  # must explain what was blocked

    def test_delete_blocked(self):
        result = self.validator.validate("DELETE FROM patients WHERE patient_id = 1")
        assert result.valid is False

    def test_truncate_blocked(self):
        result = self.validator.validate("TRUNCATE TABLE appointments")
        assert result.valid is False

    def test_update_blocked(self):
        result = self.validator.validate("UPDATE patients SET first_name = 'Hacked' WHERE 1=1")
        assert result.valid is False

    def test_insert_blocked(self):
        result = self.validator.validate("INSERT INTO patients (first_name) VALUES ('Attacker')")
        assert result.valid is False

    def test_sql_comment_injection_blocked(self):
        result = self.validator.validate("SELECT * FROM patients -- DROP TABLE patients")
        assert result.valid is False

    def test_empty_sql_blocked(self):
        result = self.validator.validate("")
        assert result.valid is False


# ── Validator: Schema Check ───────────────────────────────────────────────────

class TestValidatorSchema:
    """Layer 2: table existence checks."""

    def setup_method(self):
        self.validator = MockValidatorAgent()

    def test_valid_table_passes(self):
        result = self.validator.validate("SELECT patient_id FROM patients LIMIT 5")
        assert result.valid is True

    def test_unknown_table_blocked(self):
        result = self.validator.validate("SELECT * FROM users LIMIT 5")
        assert result.valid is False
        assert "users" in result.error_message.lower() or "table" in result.error_message.lower()

    def test_join_valid_tables_passes(self):
        sql = """
            SELECT p.first_name, a.appointment_date
            FROM patients p
            JOIN appointments a ON p.patient_id = a.patient_id
            LIMIT 10
        """
        result = self.validator.validate(sql)
        assert result.valid is True

    def test_join_unknown_table_blocked(self):
        result = self.validator.validate(
            "SELECT * FROM patients p JOIN ghost_table g ON p.patient_id = g.id"
        )
        assert result.valid is False

    def test_aggregate_query_passes(self):
        sql = """
            SELECT department_name, COUNT(*) AS total
            FROM departments
            GROUP BY department_name
            ORDER BY total DESC
            LIMIT 5
        """
        result = self.validator.validate(sql)
        assert result.valid is True


# ── Validator: Error Messages ─────────────────────────────────────────────────

class TestValidatorErrorMessages:
    """Error messages must be user-friendly — no tracebacks, no raw SQL errors."""

    def setup_method(self):
        self.validator = MockValidatorAgent()

    def test_blocked_message_is_readable(self):
        result = self.validator.validate("DROP TABLE patients")
        assert "traceback" not in result.error_message.lower()
        assert "exception" not in result.error_message.lower()
        assert len(result.error_message) > 10

    def test_unknown_table_message_helpful(self):
        result = self.validator.validate("SELECT * FROM fake_table")
        assert len(result.error_message) > 20  # Must give meaningful info


# ── Fallback Agent ────────────────────────────────────────────────────────────

class TestFallbackAgent:
    """FallbackAgent must always return a message — never crash."""

    def setup_method(self):
        self.fallback = FallbackAgent()

    def test_all_categories_return_message(self):
        for category in ["sql_generation", "validation", "execution", "api_error", "ambiguous", "unknown"]:
            result = self.fallback.handle(category)
            assert result.message, f"Empty message for category: {category}"
            assert result.failure_category == category

    def test_suggestions_not_empty(self):
        result = self.fallback.handle("unknown")
        assert len(result.suggestions) > 0

    def test_query_context_included(self):
        result = self.fallback.handle("ambiguous", original_query="show me stuff")
        # Either the query appears in the message or the message is long enough
        assert "show me stuff" in result.message or len(result.message) > 20

    def test_unknown_category_graceful(self):
        result = self.fallback.handle("totally_nonexistent_category_xyz")
        assert result.message  # Must not crash


# ── Dangerous Query Regression ────────────────────────────────────────────────

class TestDangerousQueryScenarios:
    """Regression suite for common attack and misuse patterns."""

    def setup_method(self):
        self.validator = MockValidatorAgent()

    def test_drop_case_variants(self):
        for sql in [
            "DROP TABLE patients",
            "drop table patients",
            "  DROP  TABLE  patients  ",
        ]:
            result = self.validator.validate(sql)
            assert result.valid is False, f"Should block: {sql!r}"

    def test_delete_all_records(self):
        result = self.validator.validate("DELETE FROM patients WHERE 1=1")
        assert result.valid is False

    def test_stacked_statements(self):
        result = self.validator.validate("SELECT * FROM patients; DROP TABLE patients")
        assert result.valid is False


# ── LangGraph State Structure [NEW] ──────────────────────────────────────────

class TestLangGraphState:
    """Verify make_initial_state produces a correctly structured NL2SQLState."""

    def test_initial_state_has_required_keys(self):
        state = make_initial_state("test query", "session-001")
        required = [
            "query", "session_id", "intent", "sql", "sql_attempts",
            "is_valid", "exec_rows", "exec_columns", "success",
        ]
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_initial_state_defaults(self):
        state = make_initial_state("hello", "s-001")
        assert state["query"] == "hello"
        assert state["session_id"] == "s-001"
        assert state["success"] is False
        assert state["sql_attempts"] == 0
        assert state["exec_rows"] == []


# ── Config / params.yaml [NEW] ────────────────────────────────────────────────

class TestParamsConfig:
    """Verify params.yaml loads and dot-access works correctly."""

    def test_model_config_loaded(self):
        assert params.model.name  # non-empty
        assert params.model.base_url.startswith("https://")

    def test_safety_keywords_loaded(self):
        keywords = params.safety.blocked_keywords
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert any("DROP" in k.upper() for k in keywords)

    def test_pipeline_retry_count(self):
        assert params.pipeline.max_retry_attempts >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
