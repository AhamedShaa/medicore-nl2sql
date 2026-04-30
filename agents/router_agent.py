"""
agents/router_agent.py — Intent Router: first gate in the LangGraph pipeline.

ROLE:
  Classifies every user query into one of three intents before any SQL is
  generated. This is the "security guard" at the pipeline entrance.

    analytics → proceed to SQL Generator
    clarify   → ask the user for more detail (ambiguous query)
    blocked   → refuse + explain (dangerous or irrelevant request)

WHY ROUTE FIRST:
  - Saves ~1000 tokens / ~400ms per blocked/ambiguous query
  - Gives users helpful feedback instead of a confusing SQL error
  - Keeps the SQL Generator focused on analytics queries only

LCEL CHAIN:
  ChatPromptTemplate | ChatOpenAI | JsonOutputParser
  The JsonOutputParser handles markdown-fenced JSON and retry on parse failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import BaseAgent
from utils.log import logger


_SYSTEM_PROMPT = """\
You are the Intent Router for MediCore Hospital's analytics platform.
Classify the user query into EXACTLY one of these three intents:

  analytics — The user wants data: reports, counts, lists, comparisons, trends.
  clarify   — The query is too vague or ambiguous to safely turn into SQL.
  blocked   — The query attempts data modification, is harmful, or is irrelevant.

Respond with ONLY a valid JSON object — no other text, no markdown fences:
{{
  "intent": "analytics" | "clarify" | "blocked",
  "reason": "One concise sentence explaining your choice",
  "clarification_needed": "A specific question to ask the user, or null"
}}

Examples:
  "Show top 5 revenue departments"      → analytics
  "Delete all patient records"          → blocked (data modification)
  "DROP TABLE patients"                 → blocked (DDL statement)
  "Tell me something"                   → clarify (too vague)
  "Which doctors see the most patients?" → analytics
  "What is life?"                       → blocked (irrelevant to hospital data)
"""


@dataclass
class RouterResult:
    """Structured output from the Intent Router."""

    intent: str                          # "analytics" | "clarify" | "blocked"
    reason: str                          # Why this intent was chosen
    clarification_needed: Optional[str]  # Question for user (clarify intent only)


class RouterAgent(BaseAgent):
    """
    Classifies user queries before SQL generation using an LCEL chain.

    Chain: system+human prompt → GPT-4o via OpenRouter → JsonOutputParser
    """

    def __init__(self) -> None:
        super().__init__()

        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", _SYSTEM_PROMPT),
                    ("human", "Classify this query: {query}"),
                ]
            )
            | self.llm
            | JsonOutputParser()
        )

    def route(self, query: str) -> Tuple[RouterResult, float]:
        """
        Classify a natural language query.

        Args:
            query: The user's raw input.

        Returns:
            (RouterResult, latency_ms)
        """
        t0 = time.perf_counter()
        try:
            raw: dict = self._chain.invoke({"query": query})
            latency = (time.perf_counter() - t0) * 1000

            result = RouterResult(
                intent=raw.get("intent", "clarify"),
                reason=raw.get("reason", ""),
                clarification_needed=raw.get("clarification_needed"),
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.warning(f"Router parse failed: {exc} — defaulting to clarify")
            result = RouterResult(
                intent="clarify",
                reason="Failed to parse router response.",
                clarification_needed="Could you rephrase your question more specifically?",
            )

        logger.info(f"Router → intent='{result.intent}' ({latency:.0f}ms)")
        return result, latency
