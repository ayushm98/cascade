"""Ollama provider implementation for local LLM inference."""

import logging

import httpx

from cascade.config import get_settings
from cascade.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self) -> None:
        """Initialize Ollama provider."""
        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._default_model = settings.ollama_model
        self._client = httpx.AsyncClient(timeout=120.0)

    @property
    def name(self) -> str:
        """Provider name."""
        return "ollama"

    @property
    def available_models(self) -> list[str]:
        """Available Ollama models (configured default)."""
        return [self._default_model]

    async def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate chat completion using Ollama API.

        Args:
            messages: List of message dicts
            model: Ollama model name (defaults to configured model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (num_predict in Ollama)

        Returns:
            LLMResponse with generated content
        """
        model = model or self._default_model
        logger.debug(f"Ollama request: model={model}, messages={len(messages)}")

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        # Ollama response format
        message = data.get("message", {})
        content = message.get("content", "")

        # Ollama provides token counts in different fields
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
        )

    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]

            # Check if our configured model is available
            model_base = self._default_model.split(":")[0]
            available = model_base in models

            if not available:
                logger.warning(
                    f"Ollama model '{self._default_model}' not found. "
                    f"Available: {models}. Run: ollama pull {self._default_model}"
                )

            return available

        except httpx.ConnectError:
            logger.warning("Ollama not running. Start with: ollama serve")
            return False
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
