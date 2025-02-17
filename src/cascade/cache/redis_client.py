"""Redis client wrapper for caching."""

import json
import hashlib
from typing import Any, Optional

import redis.asyncio as redis

from cascade.config import settings


class RedisClient:
    """Async Redis client for exact-match caching."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600,
    ):
        """
        Initialize Redis client.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            ttl: Default TTL for cached items in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl = ttl
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True,
        )
        # Test connection
        await self._client.ping()

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> redis.Redis:
        """Get the Redis client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    def _make_key(self, prefix: str, query: str, model: str) -> str:
        """Generate a cache key from query and model."""
        content = f"{query}:{model}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_value}"

    async def get_cached_response(
        self,
        query: str,
        model: str,
        prefix: str = "cascade:exact",
    ) -> Optional[dict[str, Any]]:
        """
        Get cached response for exact query match.

        Args:
            query: The query text
            model: The model name
            prefix: Cache key prefix

        Returns:
            Cached response dict or None if not found
        """
        key = self._make_key(prefix, query, model)
        cached = await self.client.get(key)

        if cached:
            return json.loads(cached)
        return None

    async def cache_response(
        self,
        query: str,
        model: str,
        response: dict[str, Any],
        prefix: str = "cascade:exact",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache a response.

        Args:
            query: The query text
            model: The model name
            response: The response to cache
            prefix: Cache key prefix
            ttl: TTL in seconds (uses default if not specified)
        """
        key = self._make_key(prefix, query, model)
        ttl = ttl or self.ttl
        await self.client.setex(key, ttl, json.dumps(response))

    async def invalidate(
        self,
        query: str,
        model: str,
        prefix: str = "cascade:exact",
    ) -> bool:
        """
        Invalidate a cached response.

        Args:
            query: The query text
            model: The model name
            prefix: Cache key prefix

        Returns:
            True if key was deleted, False if not found
        """
        key = self._make_key(prefix, query, model)
        result = await self.client.delete(key)
        return result > 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        info = await self.client.info("stats")
        return {
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": (
                info.get("keyspace_hits", 0)
                / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
            ),
        }


# Global client instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """Get or create the global Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient(
            host=settings.redis_host,
            port=settings.redis_port,
            ttl=settings.cache_ttl,
        )
        await _redis_client.connect()
    return _redis_client
