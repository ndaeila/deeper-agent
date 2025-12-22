"""External service integrations.

This package provides thin wrappers around external services (e.g., observability).
"""

from odr.integrations.observability import (
    get_langfuse_callbacks,
    get_observed_llm,
    is_observability_enabled,
)

__all__ = [
    "get_langfuse_callbacks",
    "get_observed_llm",
    "is_observability_enabled",
]


