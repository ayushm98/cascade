"""Tests for the API module."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from cascade.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
    ChatCompletionChoice,
)


class TestSchemas:
    """Tests for API schemas."""

    def test_chat_message_creation(self):
        """ChatMessage should be created correctly."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_roles(self):
        """ChatMessage should accept valid roles."""
        for role in ["system", "user", "assistant"]:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role

    def test_chat_request_defaults(self):
        """ChatCompletionRequest should have correct defaults."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert request.model == "gpt-4o"
        assert request.temperature == 0.7
        assert request.max_tokens is None
        assert request.stream is False

    def test_chat_request_custom_values(self):
        """ChatCompletionRequest should accept custom values."""
        request = ChatCompletionRequest(
            model="gpt-4o-mini",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.5,
            max_tokens=100,
        )
        assert request.model == "gpt-4o-mini"
        assert request.temperature == 0.5
        assert request.max_tokens == 100

    def test_usage_info(self):
        """UsageInfo should track token usage."""
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_chat_response_creation(self):
        """ChatCompletionResponse should be created correctly."""
        response = ChatCompletionResponse(
            id="test-123",
            created=1234567890,
            model="gpt-4o",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi there!"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=5,
                completion_tokens=10,
                total_tokens=15,
            ),
        )
        assert response.id == "test-123"
        assert response.model == "gpt-4o"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hi there!"


class TestCostTracking:
    """Tests for cost tracking."""

    def test_cost_calculation(self):
        """Cost should be calculated correctly."""
        from cascade.cost.pricing import calculate_cost

        # GPT-4o: $0.03/1K input, $0.06/1K output
        cost = calculate_cost("gpt-4o", 1000, 1000)
        assert cost == 0.09  # 0.03 + 0.06

    def test_free_model_cost(self):
        """Free models should have zero cost."""
        from cascade.cost.pricing import calculate_cost, is_free_model

        cost = calculate_cost("llama3.2", 1000, 1000)
        assert cost == 0.0
        assert is_free_model("llama3.2")

    def test_savings_calculation(self):
        """Savings should be calculated correctly."""
        from cascade.cost.pricing import calculate_savings

        dollars_saved, percentage_saved = calculate_savings(1.0, 10.0)
        assert dollars_saved == 9.0
        assert percentage_saved == 90.0

    def test_savings_zero_baseline(self):
        """Savings with zero baseline should not error."""
        from cascade.cost.pricing import calculate_savings

        dollars_saved, percentage_saved = calculate_savings(0.0, 0.0)
        assert dollars_saved == 0.0
        assert percentage_saved == 0.0


class TestCostTracker:
    """Tests for the cost tracker service."""

    def test_tracker_initialization(self):
        """CostTracker should initialize with zero values."""
        from cascade.cost.tracker import CostTracker

        tracker = CostTracker()
        assert tracker.total_requests == 0
        assert tracker.total_cost == 0.0
        assert tracker.cache_hits_exact == 0

    def test_record_request(self):
        """Recording requests should update totals."""
        from cascade.cost.tracker import CostTracker

        tracker = CostTracker()
        tracker.record_request(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency=0.5,
        )

        assert tracker.total_requests == 1
        assert tracker.total_cost > 0
        assert tracker.total_tokens == 150

    def test_cache_hit_tracking(self):
        """Cache hits should be tracked correctly."""
        from cascade.cost.tracker import CostTracker

        tracker = CostTracker()
        tracker.record_cache_hit("exact")
        tracker.record_cache_hit("semantic")
        tracker.record_cache_hit("miss")

        assert tracker.cache_hits_exact == 1
        assert tracker.cache_hits_semantic == 1
        assert tracker.cache_misses == 1

    def test_get_summary(self):
        """Summary should contain all metrics."""
        from cascade.cost.tracker import CostTracker

        tracker = CostTracker()
        tracker.record_request("gpt-4o", 100, 50, 0.5)
        tracker.record_cache_hit("exact")

        summary = tracker.get_summary()

        assert "total_requests" in summary
        assert "cost" in summary
        assert "cache" in summary
        assert "latency" in summary
        assert "models" in summary

    def test_reset(self):
        """Reset should clear all metrics."""
        from cascade.cost.tracker import CostTracker

        tracker = CostTracker()
        tracker.record_request("gpt-4o", 100, 50, 0.5)
        tracker.reset()

        assert tracker.total_requests == 0
        assert tracker.total_cost == 0.0
