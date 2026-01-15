"""Model pricing constants and cost calculation."""

from typing import TypedDict


class ModelPricing(TypedDict):
    """Pricing for a model (per 1K tokens)."""
    input: float
    output: float


# Pricing per 1K tokens (as of Jan 2025)
PRICING: dict[str, ModelPricing] = {
    # Gemini models
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0012},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    # Local models (free)
    "llama3.2": {"input": 0.0, "output": 0.0},
    "llama3.1": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
}

# Default model for baseline cost calculation
BASELINE_MODEL = "gemini-2.5-flash"


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Calculate the cost for a request.

    Args:
        model: Model name
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in dollars
    """
    pricing = PRICING.get(model, PRICING["gemini-2.5-flash"])

    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]

    return input_cost + output_cost


def calculate_baseline_cost(
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Calculate what the cost would be using the baseline model.

    This is used to calculate savings.

    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in dollars if using baseline model
    """
    return calculate_cost(BASELINE_MODEL, prompt_tokens, completion_tokens)


def calculate_savings(
    actual_cost: float,
    baseline_cost: float,
) -> tuple[float, float]:
    """
    Calculate cost savings.

    Args:
        actual_cost: What was actually spent
        baseline_cost: What would have been spent with baseline model

    Returns:
        Tuple of (dollars saved, percentage saved)
    """
    dollars_saved = baseline_cost - actual_cost
    percentage_saved = (dollars_saved / baseline_cost * 100) if baseline_cost > 0 else 0.0

    return dollars_saved, percentage_saved


def get_model_price(model: str) -> ModelPricing:
    """Get pricing for a model."""
    return PRICING.get(model, PRICING["gemini-2.5-flash"])


def is_free_model(model: str) -> bool:
    """Check if a model is free (local)."""
    pricing = PRICING.get(model)
    if pricing is None:
        return False
    return pricing["input"] == 0.0 and pricing["output"] == 0.0
