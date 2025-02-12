"""Routing engine for intelligent model selection."""

import logging
from dataclasses import dataclass
from typing import Literal

from cascade.router.classifier import ONNXComplexityClassifier

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    model: str
    complexity_score: float
    complexity_label: Literal["simple", "medium", "complex"]
    reason: str


class RoutingEngine:
    """
    Routes queries to appropriate LLM models based on complexity.

    Uses a trained complexity classifier to determine query difficulty
    and routes to cost-appropriate models:
    - Simple queries → Ollama (free) or GPT-3.5 (cheap)
    - Medium queries → GPT-4o-mini
    - Complex queries → GPT-4o

    Attributes:
        classifier: The complexity classifier
        simple_threshold: Score below this is "simple"
        complex_threshold: Score above this is "complex"
    """

    # Default model mapping
    DEFAULT_MODELS = {
        "simple": "llama3.2",  # Ollama (free)
        "medium": "gpt-4o-mini",  # Cheap OpenAI
        "complex": "gpt-4o",  # Best OpenAI
    }

    # Fallback models when primary is unavailable
    FALLBACK_MODELS = {
        "llama3.2": "gpt-3.5-turbo",
        "gpt-4o-mini": "gpt-4o",
        "gpt-4o": "gpt-4o",  # No fallback for top model
    }

    def __init__(
        self,
        classifier: ONNXComplexityClassifier | None = None,
        model_path: str = "models/model.onnx",
        simple_threshold: float = 0.35,
        complex_threshold: float = 0.70,
        models: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the routing engine.

        Args:
            classifier: Pre-loaded classifier (optional)
            model_path: Path to ONNX model (used if classifier not provided)
            simple_threshold: Score below this is "simple"
            complex_threshold: Score above this is "complex"
            models: Custom model mapping (optional)
        """
        self._classifier = classifier
        self._model_path = model_path
        self.simple_threshold = simple_threshold
        self.complex_threshold = complex_threshold
        self.models = models or self.DEFAULT_MODELS.copy()
        self._classifier_available = True

    @property
    def classifier(self) -> ONNXComplexityClassifier | None:
        """Lazy-load the classifier."""
        if self._classifier is None and self._classifier_available:
            try:
                self._classifier = ONNXComplexityClassifier(self._model_path)
                self._classifier.load()
                logger.info(f"Loaded complexity classifier from {self._model_path}")
            except FileNotFoundError:
                logger.warning(f"Classifier not found at {self._model_path}, using heuristics")
                self._classifier_available = False
            except Exception as e:
                logger.warning(f"Failed to load classifier: {e}, using heuristics")
                self._classifier_available = False
        return self._classifier

    def _get_complexity_label(self, score: float) -> Literal["simple", "medium", "complex"]:
        """Convert score to complexity label."""
        if score < self.simple_threshold:
            return "simple"
        elif score > self.complex_threshold:
            return "complex"
        else:
            return "medium"

    async def route(self, query: str) -> RoutingDecision:
        """
        Route a query to the appropriate model.

        Args:
            query: The user query to route

        Returns:
            RoutingDecision with model and metadata
        """
        # Try to use ML classifier
        if self.classifier is not None:
            try:
                score = await self.classifier.classify(query)
                label = self._get_complexity_label(score)
                model = self.models.get(label, self.DEFAULT_MODELS[label])

                return RoutingDecision(
                    model=model,
                    complexity_score=score,
                    complexity_label=label,
                    reason="ml_classifier",
                )
            except Exception as e:
                logger.warning(f"Classifier failed: {e}, falling back to heuristics")

        # Fallback to simple heuristics
        score, label = self._heuristic_classify(query)
        model = self.models.get(label, self.DEFAULT_MODELS[label])

        return RoutingDecision(
            model=model,
            complexity_score=score,
            complexity_label=label,
            reason="heuristic_fallback",
        )

    def _heuristic_classify(
        self, query: str
    ) -> tuple[float, Literal["simple", "medium", "complex"]]:
        """
        Simple heuristic classification as fallback.

        Args:
            query: The query to classify

        Returns:
            Tuple of (score, label)
        """
        query_lower = query.lower()
        word_count = len(query.split())

        # Complex indicators
        complex_keywords = [
            "write", "implement", "create", "build", "develop",
            "code", "function", "class", "algorithm",
            "explain step by step", "analyze", "compare", "evaluate",
            "debug", "optimize", "refactor",
        ]

        # Simple indicators
        simple_keywords = [
            "hello", "hi", "hey", "thanks", "thank you",
            "what is", "who is", "when is", "where is",
            "define", "meaning of",
        ]

        # Check for complex keywords
        for keyword in complex_keywords:
            if keyword in query_lower:
                return 0.85, "complex"

        # Check for simple keywords
        for keyword in simple_keywords:
            if keyword in query_lower:
                return 0.15, "simple"

        # Use length as a proxy
        if word_count < 10:
            return 0.25, "simple"
        elif word_count < 50:
            return 0.50, "medium"
        else:
            return 0.75, "complex"

    def route_sync(self, query: str) -> RoutingDecision:
        """
        Synchronous version of route().

        Args:
            query: The user query to route

        Returns:
            RoutingDecision with model and metadata
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.route(query))

    def get_fallback_model(self, model: str) -> str:
        """Get fallback model if primary is unavailable."""
        return self.FALLBACK_MODELS.get(model, model)

    def update_thresholds(
        self,
        simple_threshold: float | None = None,
        complex_threshold: float | None = None,
    ) -> None:
        """Update routing thresholds."""
        if simple_threshold is not None:
            self.simple_threshold = simple_threshold
        if complex_threshold is not None:
            self.complex_threshold = complex_threshold

    def __repr__(self) -> str:
        classifier_status = "loaded" if self.classifier else "not loaded"
        return (
            f"RoutingEngine("
            f"simple<{self.simple_threshold}, "
            f"complex>{self.complex_threshold}, "
            f"classifier={classifier_status})"
        )


# Global routing engine instance
_default_engine: RoutingEngine | None = None


def get_routing_engine() -> RoutingEngine:
    """Get the default routing engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = RoutingEngine()
    return _default_engine


async def route_query(query: str) -> RoutingDecision:
    """
    Convenience function to route a query.

    Args:
        query: The query to route

    Returns:
        RoutingDecision
    """
    engine = get_routing_engine()
    return await engine.route(query)
