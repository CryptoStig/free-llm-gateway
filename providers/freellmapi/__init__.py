"""FreeLLMAPI (OpenAI-compat) adapter."""

from providers.defaults import FREETHEMIND_DEFAULT_BASE

from .client import FreeLLMAPIProvider

__all__ = ["FREETHEMIND_DEFAULT_BASE", "FreeLLMAPIProvider"]
