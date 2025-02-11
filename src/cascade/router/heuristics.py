"""Heuristic-based complexity classification as fallback."""

from typing import Literal

# Keywords indicating complex queries
COMPLEX_KEYWORDS = frozenset([
    "write", "implement", "create", "build", "develop", "design",
    "code", "function", "class", "algorithm", "program",
    "step by step", "analyze", "compare", "evaluate", "explain how",
    "debug", "optimize", "refactor", "fix the bug",
    "parse", "generate", "convert", "transform",
    "api", "database", "sql", "query",
    "machine learning", "neural network", "model",
])

# Keywords indicating simple queries
SIMPLE_KEYWORDS = frozenset([
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
    "what is", "who is", "when is", "where is", "how old",
    "define", "meaning of", "definition",
    "yes", "no", "ok", "okay",
    "capital of", "population of",
])


def classify_by_heuristics(query: str) -> tuple[float, Literal["simple", "medium", "complex"]]:
    """
    Classify query complexity using simple heuristics.

    This is used as a fallback when the ML classifier is unavailable.

    Args:
        query: The query text to classify

    Returns:
        Tuple of (score, label) where score is 0.0-1.0
    """
    query_lower = query.lower().strip()
    words = query_lower.split()
    word_count = len(words)

    # Check for complex patterns
    for keyword in COMPLEX_KEYWORDS:
        if keyword in query_lower:
            return 0.85, "complex"

    # Check for simple patterns
    for keyword in SIMPLE_KEYWORDS:
        if keyword in query_lower:
            return 0.15, "simple"

    # Check for code blocks or technical indicators
    if "```" in query or "def " in query or "class " in query:
        return 0.90, "complex"

    # Length-based heuristics
    if word_count < 5:
        return 0.20, "simple"
    elif word_count < 20:
        return 0.45, "medium"
    elif word_count < 50:
        return 0.60, "medium"
    else:
        return 0.75, "complex"


def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count.

    Uses the approximation of ~4 characters per token for English text.

    Args:
        text: The text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4 + 1
