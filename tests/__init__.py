"""
tests — pytest test suite for MediCore NL2SQL Platform.

WHAT IS TESTED (17 unit tests, no API key required):
  - ValidatorAgent: keyword safety (8 tests), schema checks (5 tests), error messages (2 tests)
  - FallbackAgent: all failure categories (4 tests)
  - Dangerous query regression tests (3 tests)
  - NL2SQLState structure validation (2 tests)
  - Config loading (1 test)

WHY NO API KEY REQUIRED:
  ValidatorAgent and FallbackAgent are fully deterministic (no LLM calls).
  Tests inject a mock schema dict directly, so no database connection is needed.
  This makes the test suite fast (~1s), free to run in CI, and reliable.

HOW TO RUN:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short   # compact traceback
"""
