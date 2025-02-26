"""FastAPI application for Cascade LLM proxy."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cascade.config import settings
from cascade.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    from cascade.cache.redis_client import get_redis_client
    from cascade.cache.semantic_cache import get_semantic_cache

    # Initialize cache connections
    try:
        await get_redis_client()
    except Exception as e:
        print(f"Warning: Redis connection failed: {e}")

    try:
        get_semantic_cache()
    except Exception as e:
        print(f"Warning: Qdrant connection failed: {e}")

    yield

    # Shutdown
    from cascade.cache.redis_client import _redis_client
    if _redis_client:
        await _redis_client.disconnect()


app = FastAPI(
    title="Cascade",
    description="Intelligent LLM request router with cost optimization",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Cascade",
        "version": "0.1.0",
        "description": "Intelligent LLM request router",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
