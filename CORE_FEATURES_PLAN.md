# Core Features Implementation Plan

This document outlines the core features being added to Cascade on Dec 29, 2025.

## Current State

✅ **Working:**
- Heuristic-based routing (keyword matching + query length)
- Exact-match caching (Redis)
- Semantic caching (Qdrant + embeddings)
- Cost tracking and analytics
- OpenAI & Ollama provider support
- Streamlit UI dashboard

❌ **Missing:**
- Streaming responses (SSE)
- Retry logic with fallbacks
- Rate limiting
- Trained ML classifier (ONNX model)

## Planned Improvements

### 1. Streaming Support
**Goal:** Add SSE (Server-Sent Events) streaming for real-time token generation

**Implementation:**
- Add `stream_complete()` method to `LLMProvider` base class
- Implement streaming in OpenAI and Ollama providers
- Add streaming endpoint `/v1/chat/completions` with `stream=true` parameter
- Return SSE format: `data: {json}\n\n`

**Benefits:**
- Better UX - users see tokens as they're generated
- Lower perceived latency
- Standard OpenAI API compatibility

### 2. Retry Logic with Fallbacks
**Goal:** Automatic retries and model fallbacks on failures

**Implementation:**
- Add retry decorator with exponential backoff (tenacity library)
- Implement fallback chain: `gpt-4o` → `gpt-4o-mini` → `gpt-3.5-turbo`
- Handle rate limits, timeouts, and provider errors
- Track retry/fallback metrics

**Benefits:**
- Higher reliability
- Graceful degradation
- Better error handling

### 3. Rate Limiting
**Goal:** Prevent abuse and manage costs

**Implementation:**
- Add `slowapi` middleware for request rate limiting
- Implement token bucket algorithm
- Configurable limits per IP/API key
- Return `429 Too Many Requests` with `Retry-After` header

**Limits:**
- Default: 100 requests/minute per IP
- Configurable via environment variables

### 4. ML Classifier Training
**Goal:** Replace heuristics with trained DistilBERT model

**Implementation:**
- Create training dataset from existing queries
- Fine-tune DistilBERT for 3-class classification (simple/medium/complex)
- Convert to ONNX for fast inference
- Add fallback to heuristics if model unavailable

**Benefits:**
- More accurate routing decisions
- Better cost optimization
- Learns from actual query patterns

## Implementation Order

1. **Streaming** - High impact, moderate complexity
2. **Retry logic** - Critical for reliability
3. **Rate limiting** - Quick win for production readiness
4. **ML classifier** - Lower priority, can use heuristics for now

## Success Metrics

- Streaming: Response time perceived latency < 500ms
- Retry: 99.9% request success rate
- Rate limiting: Zero abuse incidents
- ML classifier: >85% accuracy vs manual labels
