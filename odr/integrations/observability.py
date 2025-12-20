"""Observability integration using Langfuse.

This module provides automatic LLM tracing via Langfuse's callback handler.
All LangChain/LangGraph operations using the wrapped LLM will be automatically traced.

Configuration:
    Set these environment variables in your .env file:
    - LANGFUSE_PUBLIC_KEY: Your Langfuse public key (required)
    - LANGFUSE_SECRET_KEY: Your Langfuse secret key (required)
    - LANGFUSE_HOST: Langfuse server URL (required, e.g., http://localhost:3000)
    - LANGFUSE_ENABLED: Set to "false" to explicitly disable (optional, default: true)

    See examples/*/.env.example for configuration templates.

Usage:
    from odr.integrations.observability import get_observed_llm

    llm = get_observed_llm(
        model="gpt-5.2",
        base_url="https://api.openai.com/v1",
        api_key="your-key",
    )
    # All calls to this LLM will be traced in Langfuse (if configured)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Lazy import to avoid errors when langfuse is not configured
_langfuse_handler: BaseCallbackHandler | None = None
_langfuse_init_attempted: bool = False


def _get_langfuse_handler() -> BaseCallbackHandler | None:
    """Get or create the Langfuse callback handler.
    
    Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST
    environment variables to be set. If not configured, observability is
    disabled gracefully with a warning log.
    
    Set LANGFUSE_ENABLED=false to disable tracing entirely (no warning).
    
    Returns None if Langfuse is disabled, not configured, or unavailable.
    """
    global _langfuse_handler, _langfuse_init_attempted
    
    # Allow explicit disable (no warning in this case)
    if os.getenv("LANGFUSE_ENABLED", "true").lower() == "false":
        return None
    
    # Return cached handler if already initialized
    if _langfuse_handler is not None:
        return _langfuse_handler
    
    # Only attempt initialization once to avoid repeated warnings
    if _langfuse_init_attempted:
        return None
    _langfuse_init_attempted = True
    
    # Check for required environment variables
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    
    missing_vars = []
    if not public_key:
        missing_vars.append("LANGFUSE_PUBLIC_KEY")
    if not secret_key:
        missing_vars.append("LANGFUSE_SECRET_KEY")
    if not host:
        missing_vars.append("LANGFUSE_HOST")
    
    if missing_vars:
        logger.warning(
            "Langfuse observability disabled: missing environment variables: %s. "
            "See .env.example for configuration.",
            ", ".join(missing_vars)
        )
        return None
    
    try:
        from langfuse.langchain import CallbackHandler
        
        _langfuse_handler = CallbackHandler()
        logger.info("Langfuse observability enabled (host: %s)", host)
        return _langfuse_handler
    except ImportError:
        logger.warning(
            "Langfuse observability disabled: langfuse package not installed. "
            "Install with: poetry add langfuse"
        )
        return None
    except Exception as e:
        logger.warning("Langfuse observability disabled: initialization failed: %s", e)
        return None


def get_langfuse_callbacks() -> list[BaseCallbackHandler]:
    """Get Langfuse callbacks for manual injection into LangChain calls.
    
    Returns:
        List containing the Langfuse handler if configured, empty list otherwise.
    
    Example:
        llm.invoke(messages, config={"callbacks": get_langfuse_callbacks()})
    """
    handler = _get_langfuse_handler()
    return [handler] if handler else []


def get_observed_llm(
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a ChatOpenAI instance with Langfuse observability built-in.
    
    If Langfuse credentials are configured (via environment variables),
    all LLM calls will be automatically traced. If not configured,
    returns a regular ChatOpenAI without tracing.
    
    Args:
        model: Model name/identifier.
        base_url: API endpoint URL. Defaults to OpenAI if not set.
        api_key: API key. Falls back to OPENAI_API_KEY env var.
        temperature: Sampling temperature.
        **kwargs: Additional arguments passed to ChatOpenAI.
    
    Returns:
        Configured ChatOpenAI with observability callbacks attached.
    
    Example:
        from odr.integrations.observability import get_observed_llm
        
        llm = get_observed_llm(
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    """
    callbacks = get_langfuse_callbacks()
    
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        **kwargs,
    }
    
    if base_url:
        llm_kwargs["base_url"] = base_url
    
    if api_key:
        llm_kwargs["api_key"] = api_key
    
    if callbacks:
        llm_kwargs["callbacks"] = callbacks
    
    return ChatOpenAI(**llm_kwargs)


def is_observability_enabled() -> bool:
    """Check if Langfuse observability is configured and available.
    
    Returns:
        True if Langfuse is configured and ready, False otherwise.
    """
    return _get_langfuse_handler() is not None

