"""Cost tracking and analytics service."""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from cascade.cost.pricing import (
    calculate_cost,
    calculate_baseline_cost,
    calculate_savings,
    is_free_model,
)
from cascade.router.routing_engine import RoutingDecision


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    baseline_cost: float
    latency: float
    timestamp: float
    complexity_score: float
    cache_hit: bool = False


@dataclass
class CostTracker:
    """
    Tracks costs, savings, and usage statistics.

    Thread-safe for use in async context.
    """

    total_requests: int = 0
    total_cost: float = 0.0
    total_baseline_cost: float = 0.0
    total_tokens: int = 0

    cache_hits_exact: int = 0
    cache_hits_semantic: int = 0
    cache_misses: int = 0

    requests_by_model: dict = field(default_factory=lambda: defaultdict(int))
    costs_by_model: dict = field(default_factory=lambda: defaultdict(float))
    latencies: list = field(default_factory=list)

    _recent_requests: list = field(default_factory=list)
    _max_recent: int = 1000

    def record_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        routing_decision: Optional[RoutingDecision] = None,
    ) -> RequestMetrics:
        """
        Record a completed request.

        Args:
            model: Model used for the request
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            latency: Request latency in seconds
            routing_decision: Routing decision if available

        Returns:
            RequestMetrics for the recorded request
        """
        cost = calculate_cost(model, prompt_tokens, completion_tokens)
        baseline_cost = calculate_baseline_cost(prompt_tokens, completion_tokens)
        complexity_score = routing_decision.complexity_score if routing_decision else 0.5

        metrics = RequestMetrics(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            baseline_cost=baseline_cost,
            latency=latency,
            timestamp=time.time(),
            complexity_score=complexity_score,
        )

        # Update totals
        self.total_requests += 1
        self.total_cost += cost
        self.total_baseline_cost += baseline_cost
        self.total_tokens += prompt_tokens + completion_tokens

        # Update per-model stats
        self.requests_by_model[model] += 1
        self.costs_by_model[model] += cost

        # Track latency
        self.latencies.append(latency)
        if len(self.latencies) > self._max_recent:
            self.latencies = self.latencies[-self._max_recent:]

        # Store recent request
        self._recent_requests.append(metrics)
        if len(self._recent_requests) > self._max_recent:
            self._recent_requests = self._recent_requests[-self._max_recent:]

        return metrics

    def record_cache_hit(self, cache_type: str) -> None:
        """
        Record a cache hit.

        Args:
            cache_type: Type of cache hit ('exact' or 'semantic')
        """
        if cache_type == "exact":
            self.cache_hits_exact += 1
        elif cache_type == "semantic":
            self.cache_hits_semantic += 1
        else:
            self.cache_misses += 1

    def get_savings(self) -> tuple[float, float]:
        """
        Get total cost savings.

        Returns:
            Tuple of (dollars_saved, percentage_saved)
        """
        return calculate_savings(self.total_cost, self.total_baseline_cost)

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate as a percentage."""
        total_cache_ops = self.cache_hits_exact + self.cache_hits_semantic + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return (self.cache_hits_exact + self.cache_hits_semantic) / total_cache_ops * 100

    def get_average_latency(self) -> float:
        """Get average request latency in seconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def get_p95_latency(self) -> float:
        """Get 95th percentile latency in seconds."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def get_summary(self) -> dict:
        """
        Get a summary of all tracked metrics.

        Returns:
            Dictionary with all metrics
        """
        dollars_saved, percentage_saved = self.get_savings()

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "cost": {
                "actual": round(self.total_cost, 6),
                "baseline": round(self.total_baseline_cost, 6),
                "saved_dollars": round(dollars_saved, 6),
                "saved_percentage": round(percentage_saved, 2),
            },
            "cache": {
                "exact_hits": self.cache_hits_exact,
                "semantic_hits": self.cache_hits_semantic,
                "misses": self.cache_misses,
                "hit_rate": round(self.get_cache_hit_rate(), 2),
            },
            "latency": {
                "average_ms": round(self.get_average_latency() * 1000, 2),
                "p95_ms": round(self.get_p95_latency() * 1000, 2),
            },
            "models": {
                "requests_by_model": dict(self.requests_by_model),
                "costs_by_model": {k: round(v, 6) for k, v in self.costs_by_model.items()},
            },
        }

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.total_requests = 0
        self.total_cost = 0.0
        self.total_baseline_cost = 0.0
        self.total_tokens = 0
        self.cache_hits_exact = 0
        self.cache_hits_semantic = 0
        self.cache_misses = 0
        self.requests_by_model = defaultdict(int)
        self.costs_by_model = defaultdict(float)
        self.latencies = []
        self._recent_requests = []


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
