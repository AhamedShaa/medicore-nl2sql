"""
agents/base_agent.py — Base class for all LLM-powered agents.

WHY OPENROUTER:
  OpenRouter is an API aggregator with an OpenAI-compatible endpoint.
  A single OPENROUTER_API_KEY gives access to GPT-4o, Claude, Gemini,
  and 200+ other models. To swap models, change params.model.name in
  config/params.yaml — no code changes required.

WHY LANGCHAIN (not raw OpenAI SDK):
  - langchain_openai.ChatOpenAI works unchanged with OpenRouter's endpoint
  - .with_retry() adds automatic exponential backoff on transient API failures
  - ChatPromptTemplate makes prompt variables explicit and unit-testable
  - LCEL (|) pipe syntax makes the chain dataflow readable

BACKUP MODEL DESIGN:
  Every LLM agent has two model instances: self.llm (primary) and self.backup_llm.
  The backup is a cheaper, faster model (e.g. GPT-4o-mini) configured in params.yaml.

  Use invoke_with_backup() in subclasses instead of chain.invoke() directly.
  It transparently falls back to the backup chain if the primary raises an
  API/network exception, sets backup_model_used=True in the return value, and logs
  a warning. The pipeline continues normally — no node fails due to an API blip.

  WHY NOT just use .with_retry() alone:
    .with_retry() retries the SAME model. If the model is overloaded or the key
    is rate-limited, every retry hits the same wall. The backup model is a
    DIFFERENT model — slower/cheaper primary failing → different model succeeds.

USAGE (subclass pattern):
    class MyAgent(BaseAgent):
        def __init__(self):
            super().__init__()
            self._primary_chain = prompt | self.llm | parser
            self._backup_chain  = prompt | self.backup_llm | parser

        def run(self, inputs: dict) -> tuple[result, bool]:
            result, used_backup = self.invoke_with_backup(
                self._primary_chain, self._backup_chain, inputs
            )
            return result, used_backup
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from config import params
from utils.log import logger

# Lazy import — only used when LangFuse keys are set
_get_lf_handler = None
_get_trace_id = None
try:
    from core.tracer import get_langchain_callback_handler as _get_lf_handler
    from core.tracer import get_active_trace_id as _get_trace_id
except Exception:
    pass


class BaseAgent:
    """
    Provides primary and backup ChatOpenAI instances pointed at OpenRouter.

    All LLM agents inherit from this class. Non-LLM agents (ValidatorAgent,
    FallbackAgent) do NOT inherit from this class — they are deterministic.
    """

    def __init__(self) -> None:
        # Primary model: GPT-4o, full token budget, with LangChain retry on transient errors
        self.llm = ChatOpenAI(
            model=params.model.name,
            openai_api_key=params.openrouter_api_key,
            openai_api_base=params.model.base_url,
            max_tokens=params.model.max_tokens,
            temperature=params.model.temperature,
            default_headers={
                "HTTP-Referer": params.model.site_url,
                "X-Title": params.model.site_name,
            },
        ).with_retry(stop_after_attempt=2)

        # Backup model: cheaper/faster, tighter token budget
        # Used automatically by invoke_with_backup() if primary raises an exception.
        self.backup_llm = ChatOpenAI(
            model=params.model.backup_name,
            openai_api_key=params.openrouter_api_key,
            openai_api_base=params.model.base_url,
            max_tokens=params.model.backup_max_tokens,
            temperature=params.model.temperature,
            default_headers={
                "HTTP-Referer": params.model.site_url,
                "X-Title": params.model.site_name,
            },
        ).with_retry(stop_after_attempt=2)

        logger.debug(
            f"{self.__class__.__name__} ready "
            f"(primary={params.model.name}, backup={params.model.backup_name})"
        )

    def invoke_with_backup(
        self,
        primary_chain: Runnable,
        backup_chain: Runnable,
        inputs: Dict[str, Any],
    ) -> Tuple[Any, bool]:
        """
        Invoke primary_chain; fall back to backup_chain on API/network failure.
        If LangFuse keys are set, every LLM call is auto-traced via CallbackHandler.
        The active trace ID is read from thread-local context set by run_pipeline().

        Args:
            primary_chain: LCEL chain built with self.llm.
            backup_chain:  LCEL chain built with self.backup_llm (same prompt/parser).
            inputs:        Dict of template variables to pass to the chain.

        Returns:
            (result, backup_model_used)
            backup_model_used is True when the backup model was activated.

        Raises:
            Exception: Only if BOTH models fail — the caller handles this.
        """
        # LangFuse callback — reads trace_id from thread-local set by run_pipeline()
        lf_callbacks = []
        if _get_lf_handler and _get_trace_id:
            handler = _get_lf_handler(_get_trace_id())
            if handler:
                lf_callbacks = [handler]
        config = {"callbacks": lf_callbacks} if lf_callbacks else {}

        try:
            result = primary_chain.invoke(inputs, config=config)
            return result, False
        except Exception as primary_exc:
            logger.warning(
                f"{self.__class__.__name__} primary model failed "
                f"({params.model.name}): {primary_exc!r} — switching to backup "
                f"({params.model.backup_name})"
            )
            try:
                result = backup_chain.invoke(inputs, config=config)
                logger.info(
                    f"{self.__class__.__name__} backup model succeeded "
                    f"({params.model.backup_name})"
                )
                return result, True
            except Exception as backup_exc:
                logger.error(
                    f"{self.__class__.__name__} backup model also failed: {backup_exc!r}"
                )
                raise
