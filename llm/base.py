"""Shared language-model provider contract."""

from __future__ import annotations

from typing import Protocol


class LLMProviderError(RuntimeError):
    """Raised when a configured model provider cannot return a completion."""


class LLMClient(Protocol):
    """Minimal interface required by the narrative agent."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return one completion for the supplied system and user prompts."""
