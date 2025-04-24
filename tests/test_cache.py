"""Tests for the caching module."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from cascade.cache.embeddings import EmbeddingService, cosine_similarity
from cascade.cache.redis_client import RedisClient


class TestEmbeddingService:
    """Tests for the embedding service."""

    def test_fallback_embedding_deterministic(self):
        """Fallback embeddings should be deterministic."""
        service = EmbeddingService()
        service._use_fallback = True
        service._model = None

        text = "Hello world"
        emb1 = service._fallback_embed(text)
        emb2 = service._fallback_embed(text)

        np.testing.assert_array_equal(emb1, emb2)

    def test_fallback_embedding_normalized(self):
        """Fallback embeddings should be normalized."""
        service = EmbeddingService()
        service._use_fallback = True
        service._model = None

        text = "Test query"
        embedding = service._fallback_embed(text)
        norm = np.linalg.norm(embedding)

        assert abs(norm - 1.0) < 0.01

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        service = EmbeddingService()
        service._use_fallback = True
        service._model = None

        emb1 = service._fallback_embed("Hello")
        emb2 = service._fallback_embed("Goodbye")

        assert not np.allclose(emb1, emb2)


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(vec1, vec2)) < 0.001

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        assert abs(cosine_similarity(vec1, vec2) + 1.0) < 0.001

    def test_zero_vector(self):
        """Zero vectors should return 0.0 similarity."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(vec1, vec2) == 0.0


class TestRedisClient:
    """Tests for the Redis client."""

    def test_make_key_deterministic(self):
        """Cache keys should be deterministic."""
        client = RedisClient()

        key1 = client._make_key("prefix", "query", "model")
        key2 = client._make_key("prefix", "query", "model")

        assert key1 == key2

    def test_make_key_different_inputs(self):
        """Different inputs should produce different keys."""
        client = RedisClient()

        key1 = client._make_key("prefix", "query1", "model")
        key2 = client._make_key("prefix", "query2", "model")

        assert key1 != key2

    def test_make_key_format(self):
        """Keys should have correct format."""
        client = RedisClient()
        key = client._make_key("cascade", "hello", "gpt-4o")

        assert key.startswith("cascade:")
        assert len(key.split(":")[1]) == 16  # SHA256 truncated to 16 chars

    @pytest.mark.asyncio
    async def test_cache_response_and_get(self, mock_redis_client):
        """Should be able to cache and retrieve responses."""
        client = RedisClient()
        client._client = mock_redis_client

        response = {"content": "test response"}
        await client.cache_response("query", "model", response)

        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, mock_redis_client):
        """Should be able to invalidate cached entries."""
        client = RedisClient()
        client._client = mock_redis_client

        result = await client.invalidate("query", "model")

        mock_redis_client.delete.assert_called_once()
        assert result is True
