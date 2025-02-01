"""OpenAI provider implementation."""

import logging

from openai import AsyncOpenAI

from cascade.config import get_settings
from cascade.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]

    def __init__(self) -> None:
        """Initialize OpenAI provider."""
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    @property
    def available_models(self) -> list[str]:
        """Available OpenAI models."""
        return self.MODELS

    async def complete(
        self,
        messages: list[dict],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate chat completion using OpenAI API.

        Args:
            messages: List of message dicts
            model: OpenAI model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated content
        """
        logger.debug(f"OpenAI request: model={model}, messages={len(messages)}")

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            # Simple models list check
            await self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI availability check failed: {e}")
            return False
