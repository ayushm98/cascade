"""Gemini provider implementation."""

import logging

import google.generativeai as genai

from cascade.config import get_settings
from cascade.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    MODELS = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    def __init__(self) -> None:
        """Initialize Gemini provider."""
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        # Default model
        self._model = genai.GenerativeModel("gemini-2.0-flash-exp")

    @property
    def name(self) -> str:
        """Provider name."""
        return "gemini"

    @property
    def available_models(self) -> list[str]:
        """Available Gemini models."""
        return self.MODELS

    def _convert_messages_to_gemini(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-style messages to Gemini format."""
        system_instruction = None
        history = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        return system_instruction, history

    async def complete(
        self,
        messages: list[dict],
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate chat completion using Gemini API.

        Args:
            messages: List of message dicts
            model: Gemini model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated content
        """
        logger.debug(f"Gemini request: model={model}, messages={len(messages)}")

        # Convert messages to Gemini format
        system_instruction, history = self._convert_messages_to_gemini(messages)

        # Configure generation
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        # Create model with system instruction if exists
        if system_instruction:
            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction
            )
        else:
            gemini_model = genai.GenerativeModel(model_name=model)

        # Generate response
        try:
            if len(history) > 1:
                # Use chat session for multi-turn conversations
                chat = gemini_model.start_chat(history=history[:-1])
                response = chat.send_message(
                    history[-1]["parts"],
                    generation_config=generation_config
                )
            else:
                # Single message
                response = gemini_model.generate_content(
                    history[0]["parts"] if history else [""],
                    generation_config=generation_config
                )

            # Extract content
            content = response.text if hasattr(response, 'text') else ""

            # Extract token usage
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', 0)
                completion_tokens = getattr(usage, 'candidates_token_count', 0)

            return LLMResponse(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason="stop",
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if Gemini API is available."""
        try:
            # Try a simple generation
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content("test")
            return bool(response.text)
        except Exception as e:
            logger.warning(f"Gemini availability check failed: {e}")
            return False
