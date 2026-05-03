"""
agents/contextualizer_agent.py - Rewrite follow-up questions into standalone ones.

This agent is intentionally small: it does not generate SQL and it does not
answer the user. It only resolves conversational references using recent turns,
then hands a clear standalone analytics question to the existing NL2SQL pipeline.
"""

from __future__ import annotations

import re
import time
from typing import Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import BaseAgent
from utils.log import logger


_SYSTEM_PROMPT = """\
You rewrite hospital analytics follow-up questions into standalone questions.

Use the conversation context only to resolve references like "that", "same",
"now", "compare it", "by month", or "for 2024".

Rules:
1. Return ONLY the rewritten standalone user question.
2. Do not generate SQL.
3. Do not answer the question.
4. Preserve the user's latest intent.
5. If the latest query is already standalone, return it unchanged.
6. If context is insufficient, return the latest query unchanged.
"""


class ContextualizerAgent(BaseAgent):
    """LLM-powered query contextualizer with backup-model support."""

    def __init__(self) -> None:
        super().__init__()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                (
                    "human",
                    "Conversation context:\n{memory_context}\n\n"
                    "Latest user query:\n{query}\n\n"
                    "Standalone question:",
                ),
            ]
        )

        self._primary_chain = prompt | self.llm | StrOutputParser()
        self._backup_chain = prompt | self.backup_llm | StrOutputParser()

    def contextualize(self, query: str, memory_context: str) -> Tuple[str, float, bool]:
        """Return a standalone query, latency, and backup-model flag."""
        if not memory_context.strip():
            return query, 0.0, False

        t0 = time.perf_counter()
        raw, used_backup = self.invoke_with_backup(
            self._primary_chain,
            self._backup_chain,
            {"query": query, "memory_context": memory_context},
        )
        latency = (time.perf_counter() - t0) * 1000

        resolved = _clean_question(raw) or query
        logger.info(
            f"Contextualizer -> {len(resolved)} chars ({latency:.0f}ms)"
            + (" [backup]" if used_backup else "")
        )
        return resolved, latency, used_backup


def _clean_question(raw: str) -> str:
    """Remove common LLM wrapping without changing the question content."""
    text = re.sub(r"```(?:text)?", "", raw, flags=re.IGNORECASE).strip("`\n ")
    prefixes = [
        "Standalone question:",
        "Rewritten question:",
        "Standalone:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.strip().strip('"')

