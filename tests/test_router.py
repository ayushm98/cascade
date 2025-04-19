"""Tests for the routing module."""

import pytest
from cascade.router.heuristics import classify_by_heuristics, estimate_token_count
from cascade.router.routing_engine import RoutingEngine, RoutingDecision


class TestHeuristics:
    """Tests for heuristic-based classification."""

    def test_simple_greetings(self, sample_queries):
        """Simple greetings should be classified as simple."""
        for query in sample_queries["simple"]:
            score, label = classify_by_heuristics(query)
            assert label == "simple" or score < 0.5, f"Failed for: {query}"

    def test_complex_coding_queries(self, sample_queries):
        """Coding queries should be classified as complex."""
        for query in sample_queries["complex"]:
            score, label = classify_by_heuristics(query)
            assert label == "complex" or score > 0.7, f"Failed for: {query}"

    def test_code_block_detection(self):
        """Queries with code blocks should be complex."""
        query = "Can you fix this?\n```python\ndef foo():\n    pass\n```"
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
            recommended_model="gpt-4o",
            routing_reason="High complexity query",
        )
        assert decision.complexity_score == 0.8
        assert decision.complexity_label == "complex"
        assert decision.recommended_model == "gpt-4o"

    def test_model_selection_by_threshold(self):
        """Models should be selected based on complexity thresholds."""
        engine = RoutingEngine()

        # Simple -> local model
        assert engine._select_model(0.2) == "llama3.2"

        # Medium -> mini model
        assert engine._select_model(0.5) == "gpt-4o-mini"

        # Complex -> full model
        assert engine._select_model(0.85) == "gpt-4o"

    def test_threshold_boundaries(self):
        """Test exact threshold boundaries."""
        engine = RoutingEngine()

        # At lower boundary
        assert engine._select_model(0.35) == "gpt-4o-mini"

        # At upper boundary
        assert engine._select_model(0.70) == "gpt-4o"

    @pytest.mark.asyncio
    async def test_route_query_returns_decision(self):
        """route_query should return a valid RoutingDecision."""
        from cascade.router import route_query

        decision = await route_query("Hello world")

        assert isinstance(decision, RoutingDecision)
        assert 0 <= decision.complexity_score <= 1
        assert decision.complexity_label in ["simple", "medium", "complex"]
        assert decision.recommended_model in ["llama3.2", "gpt-4o-mini", "gpt-4o"]
