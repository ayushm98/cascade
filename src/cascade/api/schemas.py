"""OpenAI-compatible request/response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message author"
    )
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="gpt-4o", description="Model to use for completion")
    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream the response")

    # Cascade-specific options
    bypass_cache: bool = Field(default=False, description="Skip semantic cache lookup")
    force_model: str | None = Field(default=None, description="Force specific model (bypass routing)")


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int = Field(..., description="Index of the choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: Literal["stop", "length", "content_filter"] | None = Field(
        default="stop", description="Reason for completion"
    )


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(..., description="Unique identifier for the completion")
    object: Literal["chat.completion"] = Field(default="chat.completion")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: list[ChatCompletionChoice] = Field(..., description="List of completion choices")
    usage: UsageInfo = Field(..., description="Token usage information")

    # Cascade-specific metadata
    cascade_metadata: dict | None = Field(
        default=None,
        description="Cascade routing metadata (cache_hit, routed_model, cost, etc.)",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: dict[str, bool] = Field(..., description="Component health status")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: dict | None = Field(default=None, description="Additional error details")
