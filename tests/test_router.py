"""Tests for the routing module."""

import pytest
from cascade.router.heuristics import classify_by_heuristics, estimate_token_count
from cascade.router.routing_engine import RoutingEngine, RoutingDecision


class TestHeuristics:
    """Tests for heuristic-based classification."""

    def test_simple_greetings(self):
        """Simple greetings should be classified as simple."""
        simple_queries = ["Hello", "Hi there", "Thanks!", "yes", "no"]
        for query in simple_queries:
            score, label = classify_by_heuristics(query)
            assert label == "simple" or score < 0.5, f"Failed for: {query}"

    def test_complex_coding_queries(self, sample_queries):
        """Coding queries should be classified as complex."""
        for query in sample_queries["complex"]:
            score, label = classify_by_heuristics(query)
            assert label == "complex" or score > 0.7, f"Failed for: {query}"

    def test_code_block_detection(self):
        """Queries with code blocks should be complex."""
        query = "```python\ndef foo():\n    pass\n```"
        score, label = classify_by_heuristics(query)
        assert label == "complex"
        assert score >= 0.85

    def test_keyword_detection(self):
        """Keywords should trigger appropriate classification."""
        # Complex keywords
        assert classify_by_heuristics("write a function")[1] == "complex"
        assert classify_by_heuristics("implement an algorithm")[1] == "complex"

        # Simple keywords
        assert classify_by_heuristics("hello there")[1] == "simple"
        assert classify_by_heuristics("what is Python")[1] == "simple"

    def test_length_based_classification(self):
        """Very short queries should be simple."""
        score, label = classify_by_heuristics("Hi")
        assert label == "simple"
        assert score < 0.3

    def test_estimate_token_count(self):
        """Token estimation should be reasonable."""
        text = "Hello world"  # ~3 tokens
        estimate = estimate_token_count(text)
        assert 2 <= estimate <= 5


class TestRoutingEngine:
    """Tests for the routing engine."""

    def test_routing_decision_creation(self):
        """RoutingDecision should be created correctly."""
        decision = RoutingDecision(
            complexity_score=0.8,
            complexity_label="complex",
            model="gpt-4o",
            reason="High complexity query",
        )
        assert decision.complexity_score == 0.8
        assert decision.complexity_label == "complex"
        assert decision.model == "gpt-4o"

    def test_complexity_label_thresholds(self):
        """Complexity labels should be determined by thresholds."""
        engine = RoutingEngine()

        # Simple -> score < 0.35
        assert engine._get_complexity_label(0.2) == "simple"

        # Medium -> 0.35 <= score <= 0.70
        assert engine._get_complexity_label(0.5) == "medium"

        # Complex -> score > 0.70
        assert engine._get_complexity_label(0.85) == "complex"

    def test_threshold_boundaries(self):
        """Test exact threshold boundaries."""
        engine = RoutingEngine()

        # At lower boundary - still medium
        assert engine._get_complexity_label(0.35) == "medium"

        # Just above upper boundary - complex
        assert engine._get_complexity_label(0.71) == "complex"

    @pytest.mark.asyncio
    async def test_route_query_returns_decision(self):
        """route_query should return a valid RoutingDecision."""
        from cascade.router import route_query

        decision = await route_query("Hello world")

        assert isinstance(decision, RoutingDecision)
        assert 0 <= decision.complexity_score <= 1
        assert decision.complexity_label in ["simple", "medium", "complex"]
        assert decision.model in ["llama3.2", "gpt-4o-mini", "gpt-4o"]
