"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    @abstractmethod
    def available_models(self) -> list[str]:
        """List of available models for this provider."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated content and metadata
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and healthy."""
        ...
