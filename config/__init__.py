"""
config/__init__.py — Central configuration for MediCore NL2SQL Platform.

WHY THIS FILE EXISTS HERE (not as a root config.py):
  Python resolves `import config` by finding the `config/` package BEFORE any
  `config.py` module at the same path level. Placing the loading logic here
  ensures every `from config import params` in the codebase finds exactly this
  file — no ambiguity, no accidental module vs. package shadowing.

HOW IT WORKS:
  1. Loads config/params.yaml  — all tunable parameters (model, DB, safety, logging)
  2. Overlays secrets from .env — API keys, never committed to git
  3. Exports `params` as a SimpleNamespace with full dot-access

USAGE (from any module in the project):
    from config import params

    params.model.name                    # "openai/gpt-4o"
    params.pipeline.max_retry_attempts   # 3
    params.safety.blocked_keywords       # ["DROP", "DELETE", ...]
    params.openrouter_api_key            # injected from .env

WHY params.yaml OVER raw env vars:
  - Structured, typed, self-documented in one place
  - Safe to commit to git (no secrets)
  - Change model, retry count, log rotation without touching code
  - Secrets stay in .env and are overlaid at runtime
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env before reading os.getenv() — must happen before any getenv call
load_dotenv()


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    """Load YAML file; raise FileNotFoundError with a helpful message if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Ensure config/params.yaml exists in the project root."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_namespace(obj: Any) -> Any:
    """
    Recursively convert dicts → SimpleNamespace for dot-access.
    Lists are preserved as lists (items converted recursively).

    Example:
        params["model"]["name"]   →   params.model.name
    """
    if isinstance(obj, dict):
        ns = SimpleNamespace()
        for key, value in obj.items():
            setattr(ns, key, _to_namespace(value))
        return ns
    if isinstance(obj, list):
        return [_to_namespace(item) for item in obj]
    return obj


# ── Load params.yaml ──────────────────────────────────────────────────────────
# __file__ is .../medicore_nl2sql/config/__init__.py
# So Path(__file__).parent is .../medicore_nl2sql/config/
# And params.yaml lives right next to this file.

_yaml_path = Path(__file__).parent / "params.yaml"
_raw: dict = _load_yaml(_yaml_path)
params: SimpleNamespace = _to_namespace(_raw)

# ── Overlay secrets from .env (never put API keys in params.yaml) ─────────────

params.openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
params.langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
params.langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")

# Allow runtime overrides via environment variables
_env_db = os.getenv("DB_PATH", "")
if _env_db:
    params.database.path = _env_db

_env_log = os.getenv("LOG_LEVEL", "")
if _env_log:
    params.logging.level = _env_log.upper()


# ── Startup validation ────────────────────────────────────────────────────────

def validate() -> None:
    """
    Call at application startup to catch missing secrets early.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set in .env.
    """
    if not params.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set.\n"
            "Copy .env.example → .env and add your OpenRouter API key.\n"
            "Get one free at https://openrouter.ai/keys"
        )


__all__ = ["params", "validate"]
