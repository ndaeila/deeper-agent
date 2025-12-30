"""LLM Factory for centralized model creation and configuration."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel

from odr.integrations.observability import (
    get_observed_browser_use_llm,
    get_observed_gemini_llm,
    get_observed_llm,
)


class DefaultLLMFactory:
    """Factory for creating configured LLM instances with observability."""

    def __init__(self, prefer_gemini: bool = False, agent_config: dict[str, dict[str, Any]] | None = None):
        """Initialize the factory.
        
        Args:
            prefer_gemini: Whether to prefer Gemini models when 'default' provider is requested
                          and Gemini credentials are available.
            agent_config: Optional configuration overrides for specific agents/nodes.
                          Map from agent name to kwargs dict (e.g. {"judge-counsel": {"model": "gpt-4"}}).
        """
        self.prefer_gemini = prefer_gemini
        self.agent_config = agent_config or {}
        self._check_credentials()

    def _check_credentials(self) -> None:
        """Cache credential availability."""
        self.has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self.has_openai = bool(os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))

    def get_llm(
        self,
        name: str,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Get an LLM instance with the specified configuration.

        Args:
            name: The name for the agent/node (used for trace naming).
            provider: 'openai', 'gemini', or None (auto-detect based on preference/availability).
            model: Specific model name to use.
            temperature: Sampling temperature.
            **kwargs: Additional model arguments.

        Returns:
            Configured BaseChatModel.
        """
        # Apply agent-specific overrides
        overrides = self.agent_config.get(name, {})
        
        # Resolve parameters (kwargs > overrides > defaults)
        resolved_provider = kwargs.get("provider") or overrides.get("provider") or provider
        resolved_model = kwargs.get("model") or overrides.get("model") or model
        resolved_temp = kwargs.get("temperature") if "temperature" in kwargs else overrides.get("temperature", temperature)
        
        # Merge other kwargs
        combined_kwargs = {**kwargs, **overrides}
        # Remove handled keys from kwargs to avoid conflicts or double passing
        for k in ["provider", "model", "temperature"]:
            combined_kwargs.pop(k, None)

        # Determine effective provider
        effective_provider = resolved_provider
        if not effective_provider:
            if self.prefer_gemini and self.has_gemini:
                effective_provider = "gemini"
            else:
                effective_provider = "openai"

        # Handle Gemini
        if effective_provider in ("gemini", "google"):
            if not self.has_gemini:
                # Fallback to OpenAI if Gemini requested but missing
                if self.has_openai:
                    effective_provider = "openai"
                else:
                    raise ValueError("No Gemini or OpenAI credentials found.")
            else:
                return get_observed_gemini_llm(
                    model=resolved_model or "gemini-3-pro-preview",
                    temperature=resolved_temp,
                    name=name,
                    **combined_kwargs,
                )

        # Handle OpenAI / Default
        return get_observed_llm(
            model=resolved_model or os.getenv("MODEL_NAME"),
            temperature=resolved_temp,
            name=name,
            **combined_kwargs,
        )

    def get_browser_use_llm(
        self,
        name: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get an LLM instance for Browser Use (wraps ChatOpenAI)."""
        return get_observed_browser_use_llm(
            model=model or os.getenv("BROWSER_USE_MODEL", "gpt-5-nano-2025-08-07"),
            name=name,
            **kwargs,
        )

