"""API routes for Cascade proxy."""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from cascade.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
    ChatCompletionChoice,
)
from cascade.router import route_query, RoutingDecision
from cascade.providers import GeminiProvider, OllamaProvider
from cascade.cache.redis_client import get_redis_client
from cascade.cache.semantic_cache import get_semantic_cache, CacheHit
from cascade.cost.tracker import CostTracker, get_cost_tracker


router = APIRouter()


async def get_provider(model: str):
    """Get the appropriate provider for a model."""
    if model.startswith("gemini-"):
        return GeminiProvider()
    else:
        return OllamaProvider()


async def try_cache_lookup(
    query: str,
    model: str,
) -> tuple[Optional[dict], str]:
    """
    Try to find a cached response.

    Returns:
        Tuple of (cached_response, cache_type)
        cache_type is 'exact', 'semantic', or 'miss'
    """
    # Try exact match first
    try:
        redis = await get_redis_client()
        cached = await redis.get_cached_response(query, model)
        if cached:
            return cached, "exact"
    except Exception:
        pass

    # Try semantic match
    try:
        semantic_cache = get_semantic_cache()
        hit: Optional[CacheHit] = await semantic_cache.search(query, model)
        if hit:
            return hit.response, "semantic"
    except Exception:
        pass

    return None, "miss"


async def store_in_cache(
    query: str,
    model: str,
    response: dict,
) -> None:
    """Store response in both caches."""
    # Store in exact match cache
    try:
        redis = await get_redis_client()
        await redis.cache_response(query, model, response)
    except Exception:
        pass

    # Store in semantic cache
    try:
        semantic_cache = get_semantic_cache()
        await semantic_cache.store(query, response, model)
    except Exception:
        pass


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Routes requests to appropriate models based on complexity.
    """
    start_time = time.time()

    # Extract the user query for routing
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    query = user_messages[-1].content
    original_model = request.model

    # Route the query
    routing: RoutingDecision = await route_query(query)

    # Determine final model
    if request.model == "auto" or request.model is None:
        final_model = routing.model
    else:
        final_model = request.model

    # Check cache
    cached_response, cache_type = await try_cache_lookup(query, final_model)
    if cached_response:
        # Track cache hit
        tracker = get_cost_tracker()
        tracker.record_cache_hit(cache_type)
        return ChatCompletionResponse(**cached_response)

    # Get provider and make request
    provider = await get_provider(final_model)

    try:
        llm_response = await provider.complete(
            model=final_model,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        # Convert LLMResponse to dict format for compatibility
        response = {
            "id": f"cascade-{int(time.time())}",
            "choices": [{"message": {"content": llm_response.content}}],
            "usage": {
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "total_tokens": llm_response.prompt_tokens + llm_response.completion_tokens,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build response
    usage = UsageInfo(
        prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
        completion_tokens=response.get("usage", {}).get("completion_tokens", 0),
        total_tokens=response.get("usage", {}).get("total_tokens", 0),
    )

    chat_response = ChatCompletionResponse(
        id=response.get("id", f"cascade-{int(time.time())}"),
        object="chat.completion",
        created=int(time.time()),
        model=final_model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", ""),
                ),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )

    # Store in cache
    await store_in_cache(query, final_model, chat_response.model_dump())

    # Track cost
    tracker = get_cost_tracker()
    latency = time.time() - start_time
    tracker.record_request(
        model=final_model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        latency=latency,
        routing_decision=routing,
    )

    return chat_response


@router.get("/stats")
async def get_stats():
    """Get usage statistics and cost savings."""
    tracker = get_cost_tracker()
    return tracker.get_summary()


@router.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {"id": "auto", "description": "Automatic routing based on query complexity"},
            {"id": "gemini-2.5-flash", "description": "Most capable Gemini model"},
            {"id": "gemini-1.5-flash", "description": "Fast and efficient Gemini model"},
            {"id": "gemini-1.5-pro", "description": "Gemini Pro model"},
            {"id": "llama3.2", "description": "Local Llama model via Ollama"},
            {"id": "mistral", "description": "Local Mistral model via Ollama"},
        ]
    }
