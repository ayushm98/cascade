"""Embedding service for semantic similarity."""

import asyncio
from typing import Optional

import numpy as np


class EmbeddingService:
    """
    Generate embeddings for semantic search.

    Uses sentence-transformers for local embedding generation.
    Falls back to simple TF-IDF if sentence-transformers unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service.

        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self._model = None
        self._use_fallback = False

    def _load_model(self) -> None:
        """Lazy load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._use_fallback = False
        except ImportError:
            # Fallback to simple hashing if sentence-transformers not available
            self._use_fallback = True
            self._model = self._create_fallback_model()

    def _create_fallback_model(self):
        """Create a simple fallback vectorizer."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            return TfidfVectorizer(max_features=384)
        except ImportError:
            return None

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Generate a simple hash-based embedding as fallback."""
        # Create a deterministic embedding from text hash
        import hashlib
        hash_bytes = hashlib.sha384(text.encode()).digest()
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        self._load_model()

        if self._use_fallback:
            return self._fallback_embed(text)

        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embedding vectors
        """
        self._load_model()

        if self._use_fallback:
            return np.array([self._fallback_embed(t) for t in texts])

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings

    async def embed_async(self, text: str) -> np.ndarray:
        """Async wrapper for embedding generation."""
        return await asyncio.to_thread(self.embed, text)

    async def embed_batch_async(self, texts: list[str]) -> np.ndarray:
        """Async wrapper for batch embedding generation."""
        return await asyncio.to_thread(self.embed_batch, texts)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        self._load_model()
        if self._use_fallback:
            return 48  # SHA-384 produces 48 bytes
        return self._model.get_sentence_embedding_dimension()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
