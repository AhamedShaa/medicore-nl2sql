"""
agents — All specialist agents for the MediCore NL2SQL pipeline.

WHY SEPARATE AGENTS (not one big LLM call):
  Each agent has a single, well-defined responsibility. This makes the system:
  - Testable: each agent can be unit-tested in isolation
  - Debuggable: LangFuse traces show exactly which agent produced what output
  - Swappable: replace one agent's prompt or model without touching others
  - Safer: ValidatorAgent and FallbackAgent are LLM-free (deterministic)

AGENTS IN PIPELINE ORDER:
  1. RouterAgent           — classify intent: analytics / clarify / blocked
  2. SQLGeneratorAgent     — NL → SELECT SQL using live schema context
                             .generate(query, validation_error?) — first attempt + validation retries
                             .repair(sql, exec_error)           — execution-error repair (new)
  3. ValidatorAgent        — 3-layer safety: keyword / schema / syntax
  4. ResultInterpreterAgent— raw rows → natural language insight + chart spec
  5. FallbackAgent         — schema-aware error recovery (always deterministic)

USAGE:
    from agents import RouterAgent, SQLGeneratorAgent, ValidatorAgent
    from agents import FallbackAgent, FallbackResult
"""

from agents.base_agent import BaseAgent
from agents.router_agent import RouterAgent, RouterResult
from agents.sql_generator import SQLGeneratorAgent
from agents.validator_agent import ValidatorAgent, ValidationResult
from agents.result_interpreter import ResultInterpreterAgent, ChartSpec, InterpretedResult
from agents.fallback_agent import FallbackAgent, FallbackResult

__all__ = [
    # Base
    "BaseAgent",
    # Intent routing
    "RouterAgent",
    "RouterResult",
    # SQL generation + repair
    "SQLGeneratorAgent",
    # Validation
    "ValidatorAgent",
    "ValidationResult",
    # Result interpretation
    "ResultInterpreterAgent",
    "ChartSpec",
    "InterpretedResult",
    # Fallback / error recovery
    "FallbackAgent",
    "FallbackResult",
]
