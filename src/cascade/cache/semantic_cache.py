"""Semantic cache using vector similarity search."""

import uuid
import json
from typing import Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from cascade.cache.embeddings import EmbeddingService, get_embedding_service
from cascade.config import settings


@dataclass
class CacheHit:
    """Represents a cache hit with similarity score."""
    query: str
    response: dict[str, Any]
    similarity: float
    model: str


class SemanticCache:
    """
    Semantic cache using Qdrant for vector similarity search.

    Caches responses and retrieves them based on semantic similarity
    rather than exact match.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "cascade_cache",
        embedding_service: Optional[EmbeddingService] = None,
        similarity_threshold: float = 0.92,
    ):
        """
        Initialize semantic cache.

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Name of the vector collection
            embedding_service: Embedding service instance
            similarity_threshold: Minimum similarity for cache hit (0.0-1.0)
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_service = embedding_service or get_embedding_service()
        self.similarity_threshold = similarity_threshold
        self._client: Optional[QdrantClient] = None

    def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(url=self.qdrant_url)
        return self._client

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        client = self._get_client()
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_service.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    async def search(
        self,
        query: str,
        model: Optional[str] = None,
        top_k: int = 1,
    ) -> Optional[CacheHit]:
        """
        Search for semantically similar cached responses.

        Args:
            query: Query text to search for
            model: Optional model filter
            top_k: Number of results to consider

        Returns:
            CacheHit if found above threshold, None otherwise
        """
        embedding = await self.embedding_service.embed_async(query)
        client = self._get_client()

        # Build filter if model specified
        search_filter = None
        if model:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="model",
                        match=models.MatchValue(value=model),
                    )
                ]
            )

        results = client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            query_filter=search_filter,
            limit=top_k,
        )

        if not results:
            return None

        best_match = results[0]
        if best_match.score < self.similarity_threshold:
            return None

        payload = best_match.payload
        return CacheHit(
            query=payload["query"],
            response=json.loads(payload["response"]),
            similarity=best_match.score,
            model=payload["model"],
        )

    async def store(
        self,
        query: str,
        response: dict[str, Any],
        model: str,
    ) -> str:
        """
        Store a response in the semantic cache.

        Args:
            query: Original query text
            response: Response to cache
            model: Model that generated the response

        Returns:
            ID of the stored point
        """
        embedding = await self.embedding_service.embed_async(query)
        client = self._get_client()
        point_id = str(uuid.uuid4())

        client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "query": query,
                        "response": json.dumps(response),
                        "model": model,
                    },
                )
            ],
        )

        return point_id

    async def delete(self, point_id: str) -> bool:
        """
        Delete a cached response by ID.

        Args:
            point_id: ID of the point to delete

        Returns:
            True if deleted successfully
        """
        client = self._get_client()
        client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[point_id]),
        )
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        client = self._get_client()
        info = client.get_collection(self.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
        }


# Global semantic cache instance
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create the global semantic cache."""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            similarity_threshold=settings.similarity_threshold,
        )
        _semantic_cache.ensure_collection()
    return _semantic_cache
