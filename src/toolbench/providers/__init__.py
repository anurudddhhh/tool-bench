"""Provider abstractions and implementations for LLM interaction (F11d)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from toolbench.providers.base import ProviderBase, ProviderResponse
from toolbench.providers.gemini import GeminiProvider
from toolbench.providers.mock import MockProvider
from toolbench.rate_limit import RateLimiter

if TYPE_CHECKING:
    from toolbench.cache import DiskCache
    from toolbench.config import ToolbenchConfig

__all__ = [
    "GeminiProvider",
    "MockProvider",
    "ProviderBase",
    "ProviderResponse",
    "get_provider",
]


def get_provider(
    config: ToolbenchConfig,
    cache: DiskCache | None = None,
) -> ProviderBase:
    """Instantiate and return the configured LLM provider.

    Args:
        config: Loaded ToolbenchConfig instance.
        cache: Optional DiskCache instance to connect to the provider.

    Returns:
        ProviderBase subclass instance (GeminiProvider or MockProvider).

    Raises:
        ValueError: If config specifies an unsupported provider.
    """
    provider_type = config.model.provider.lower().strip()

    if provider_type == "gemini":
        rate_limiter = RateLimiter(config=config.rate_limit)
        return GeminiProvider(
            model_name=config.model.name,
            api_key_env=config.model.api_key_env,
            rate_limiter=rate_limiter,
            cache=cache,
        )

    if provider_type == "mock":
        return MockProvider(model_name=config.model.name)

    raise ValueError(
        f"Unsupported provider: '{config.model.provider}'. "
        "Supported providers are 'gemini' and 'mock'."
    )
