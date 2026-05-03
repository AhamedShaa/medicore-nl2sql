"""
core/memory.py - Lightweight multi-turn conversation memory.

The NL2SQL pipeline only needs compact conversational context, not a vector
database. This module stores recent turns in a small SQLite database so follow-up
questions can be rewritten into standalone analytics questions and memory can
survive container restarts when data/ is mounted as a Docker volume.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import params
from utils.log import logger


class ConversationMemory:
    """SQLite-backed store for compact conversation turns."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or params.memory.path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    user_query TEXT NOT NULL,
                    resolved_query TEXT NOT NULL,
                    sql TEXT,
                    insight TEXT,
                    summary_title TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    failure_category TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_turns_lookup
                ON conversation_turns (conversation_id, turn_id)
                """
            )

    def next_turn_id(self, conversation_id: str) -> int:
        """Return the next 1-based turn number for a conversation."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_id), 0) + 1 AS next_id "
                "FROM conversation_turns WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["next_id"])

    def load_recent(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load recent turns in chronological order."""
        limit = limit or params.memory.max_turns
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT turn_id, user_query, resolved_query, sql, insight,
                       summary_title, success, row_count, failure_category, created_at
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY turn_id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()

        turns = [dict(row) for row in rows]
        turns.reverse()
        return turns

    def build_context(self, turns: List[Dict[str, Any]]) -> str:
        """Format recent turns into bounded prompt context."""
        if not turns:
            return ""

        parts: List[str] = []
        for turn in turns:
            status = "success" if turn.get("success") else "failed"
            parts.append(
                "\n".join(
                    [
                        f"Turn {turn.get('turn_id')} ({status})",
                        f"User: {turn.get('user_query', '')}",
                        f"Standalone: {turn.get('resolved_query', '')}",
                        f"SQL: {turn.get('sql') or 'none'}",
                        f"Insight: {turn.get('insight') or 'none'}",
                    ]
                )
            )

        context = "\n\n".join(parts)
        max_chars = params.memory.max_context_chars
        if len(context) > max_chars:
            return context[-max_chars:]
        return context

    def save_turn(self, conversation_id: str, turn_id: int, state: Dict[str, Any]) -> None:
        """Persist a completed pipeline turn."""
        if not conversation_id:
            return

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO conversation_turns (
                        conversation_id, turn_id, user_query, resolved_query, sql,
                        insight, summary_title, success, row_count, failure_category,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        turn_id,
                        state.get("query", ""),
                        state.get("resolved_query") or state.get("query", ""),
                        state.get("sql", ""),
                        state.get("insight", ""),
                        state.get("summary_title", ""),
                        1 if state.get("success") else 0,
                        int(state.get("exec_row_count", 0) or 0),
                        state.get("failure_category", ""),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception as exc:
            logger.warning(f"Failed to save conversation memory: {exc}")


def memory_enabled() -> bool:
    """Return True when memory is configured on."""
    return bool(getattr(params.memory, "enabled", False))

