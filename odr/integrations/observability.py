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

# Lazy import for Google Gemini to avoid errors when not needed
_ChatGoogleGenerativeAI: Any | None = None

logger = logging.getLogger(__name__)

class _QuietOtelHandler(logging.Handler):
    """Handler that shows one-liner per OTEL error instead of verbose stack traces."""

    def emit(self, record: logging.LogRecord) -> None:
        # Only warn if Langfuse is supposed to be enabled
        if os.getenv("LANGFUSE_ENABLED", "true").lower() == "false":
            return
        # Check if this is a connection/export error
        msg = str(record.msg).lower() if record.msg else ""
        exc_text = str(record.exc_info[1]).lower() if record.exc_info and record.exc_info[1] else ""
        if "export" in msg or "connection" in exc_text or "refused" in exc_text:
            print("⚠ Langfuse tracing failed (connection refused)")


# Replace default handlers with our quiet one for OpenTelemetry loggers
_quiet_handler = _QuietOtelHandler()
for _logger_name in (
    "opentelemetry",
    "opentelemetry.exporter",
    "opentelemetry.exporter.otlp",
    "opentelemetry.exporter.otlp.proto.http",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.sdk",
    "opentelemetry.sdk._shared_internal",
):
    _otel_logger = logging.getLogger(_logger_name)
    _otel_logger.handlers = [_quiet_handler]
    _otel_logger.propagate = False

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
    name: str | None = None,
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
        name: Name to assign to the LLM for tracing identification.
        **kwargs: Additional arguments passed to ChatOpenAI.
    
    Returns:
        Configured ChatOpenAI with observability callbacks attached.
    
    Example:
        from odr.integrations.observability import get_observed_llm
        
        llm = get_observed_llm(
            model="gpt-5",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            name="my-agent-llm",
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

    if name:
        llm_kwargs["name"] = name
    
    return ChatOpenAI(**llm_kwargs)


def get_observed_gemini_llm(
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0,
    name: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a Google Gemini instance with Langfuse observability built-in.
    
    If Langfuse credentials are configured (via environment variables),
    all LLM calls will be automatically traced. If not configured,
    returns a regular ChatGoogleGenerativeAI without tracing.
    
    Args:
        model: Model name/identifier. Falls back to GEMINI_MODEL env var.
        api_key: Google API key. Falls back to GEMINI_API_KEY env var.
        temperature: Sampling temperature.
        name: Name to assign to the LLM for tracing identification.
        **kwargs: Additional arguments passed to ChatGoogleGenerativeAI.
    
    Returns:
        Configured ChatGoogleGenerativeAI with observability callbacks attached.
    
    Example:
        from odr.integrations.observability import get_observed_gemini_llm
        
        llm = get_observed_gemini_llm(
            model="gemini-3-flash-preview",
            api_key=os.getenv("GEMINI_API_KEY"),
            name="planner-gemini",
        )
    """
    global _ChatGoogleGenerativeAI
    
    if _ChatGoogleGenerativeAI is None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai not installed. Install with:\n"
                "  poetry add langchain-google-genai"
            ) from e
    
    callbacks = get_langfuse_callbacks()
    
    # Get model and API key from env if not provided.
    # Per langchain-google-genai docs: GOOGLE_API_KEY is primary, GEMINI_API_KEY is fallback.
    model_name = (model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")).strip()
    gemini_api_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    
    if not gemini_api_key:
        raise ValueError(
            "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass api_key=..."
        )
    
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "api_key": gemini_api_key,
        **kwargs,
    }
    
    if callbacks:
        llm_kwargs["callbacks"] = callbacks

    if name:
        llm_kwargs["name"] = name
    
    return _ChatGoogleGenerativeAI(**llm_kwargs)


def is_observability_enabled() -> bool:
    """Check if Langfuse observability is configured and available.
    
    Returns:
        True if Langfuse is configured and ready, False otherwise.
    """
    return _get_langfuse_handler() is not None


def get_observed_browser_use_llm(
    model: str,
    api_key: str | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a Browser Use ChatOpenAI instance with Langfuse observability built-in.
    
    This wraps Browser Use's ChatOpenAI to use Langfuse's AsyncOpenAI client,
    which automatically traces all OpenAI SDK calls to Langfuse.
    
    If Langfuse credentials are not configured, returns a regular BrowserUseChatOpenAI
    without tracing.
    
    Args:
        model: Model name/identifier.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        name: Name to assign to the LLM (if supported by underlying class).
        **kwargs: Additional arguments passed to BrowserUseChatOpenAI (e.g., reasoning_effort).
    
    Returns:
        BrowserUseChatOpenAI instance with Langfuse-wrapped OpenAI client if configured,
        otherwise regular BrowserUseChatOpenAI.
    
    Example:
        from odr.integrations.observability import get_observed_browser_use_llm
        
        llm = get_observed_browser_use_llm(
            model="gpt-4.1-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            reasoning_effort="low",
            name="browser-use-worker",
        )
    """
    # Lazy import to avoid errors when browser_use is not installed
    try:
        from browser_use.llm import ChatOpenAI as BrowserUseChatOpenAI
    except ImportError as e:
        raise ImportError(
            "browser-use not installed. Install with:\n"
            "  poetry add browser-use"
        ) from e
    
    # If Langfuse is not enabled, return regular BrowserUseChatOpenAI
    if not is_observability_enabled():
        return BrowserUseChatOpenAI(model=model, api_key=api_key, **kwargs)
    
    # Import Langfuse's OpenAI wrapper
    try:
        from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI
    except ImportError as e:
        logger.warning(
            "Langfuse OpenAI integration not available. Install with:\n"
            "  poetry add langfuse[openai]\n"
            "Falling back to regular BrowserUseChatOpenAI without tracing."
        )
        return BrowserUseChatOpenAI(model=model, api_key=api_key, **kwargs)
    
    # Create a custom class that wraps the OpenAI client with Langfuse
    class LangfuseChatOpenAI(BrowserUseChatOpenAI):
        """Browser Use ChatOpenAI with Langfuse observability."""
        
        _client: LangfuseAsyncOpenAI | None = None
        
        def get_client(self) -> LangfuseAsyncOpenAI:
            """Override to return Langfuse-wrapped AsyncOpenAI client."""
            if self._client is None:
                # Try to get client params from parent class
                # Browser Use's ChatOpenAI may expose _get_client_params() or similar
                client_params: dict[str, Any] = {}
                try:
                    # Try the method name from user's example (instance method)
                    if hasattr(self, "_get_client_params"):
                        client_params = self._get_client_params()  # type: ignore[misc]
                    else:
                        # Fallback: construct params from instance attributes
                        # Browser Use ChatOpenAI typically stores these as instance attrs
                        client_params = {
                            "api_key": getattr(self, "api_key", api_key),
                            "base_url": getattr(self, "base_url", None),
                            "timeout": getattr(self, "timeout", None),
                            "max_retries": getattr(self, "max_retries", None),
                        }
                        # Remove None values
                        client_params = {k: v for k, v in client_params.items() if v is not None}
                except Exception as e:
                    logger.warning(
                        "Could not extract client params from BrowserUseChatOpenAI: %s. "
                        "Using minimal config.",
                        e
                    )
                    # Minimal fallback
                    client_params = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
                
                # We can't easily pass 'name' to the AsyncOpenAI client directly 
                # but we can try to use it if Langfuse wrapper supports it in future.
                self._client = LangfuseAsyncOpenAI(**client_params)
            return self._client
    
    return LangfuseChatOpenAI(model=model, api_key=api_key, **kwargs)
