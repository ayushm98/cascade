"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "simple": [
            "Hello",
            "What is 2 + 2?",
            "What is the capital of France?",
            "Hi there",
            "Thanks!",
        ],
        "medium": [
            "Explain the difference between TCP and UDP.",
            "What are the main features of Python 3.11?",
            "How does garbage collection work in Java?",
            "Describe the MVC architecture pattern.",
        ],
        "complex": [
            "Write a Python function that implements a binary search tree.",
            "Create a REST API with authentication using FastAPI.",
            "Implement a neural network from scratch in Python.",
            "Design a distributed caching system with Redis.",
            "Write a compiler for a simple programming language.",
        ],
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = AsyncMock()
    client.get.return_value = None
    client.setex.return_value = True
    client.delete.return_value = 1
    client.ping.return_value = True
    return client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = MagicMock()
    client.search.return_value = []
    client.upsert.return_value = None
    return client
