# Product Requirements Document: Cascade Data Generation Agent

**Project**: Cascade - Intelligent LLM Router
**Document Type**: PRD - Data Generation Agent
**Author**: AI Assistant
**Date**: January 15, 2026
**Status**: Draft

---

## 1. Executive Summary

Build an autonomous agent that continuously sends diverse, realistic queries to the Cascade API to populate the dashboard with meaningful analytics data. The agent should simulate real-world usage patterns, including simple queries (routed to free llama3.2), complex queries (routed to Gemini), and duplicate queries (demonstrating cache effectiveness).

---

## 2. Project Context

### 2.1 What is Cascade?

**Cascade** is an intelligent LLM request router that reduces API costs by 60%+ through:
- **ML-based routing**: Uses a DistilBERT classifier (or heuristics fallback) to route queries by complexity
- **Semantic caching**: Redis (exact match) + Qdrant (vector similarity) for response caching
- **Cost optimization**: Routes simple queries to free local Ollama (llama3.2), complex queries to Gemini API

### 2.2 Current System Architecture

```
Client → Cascade API (FastAPI) → Routing Engine → LLM Provider
                ↓
         Cache Layer (Redis + Qdrant)
                ↓
         Cost Tracker & Metrics
                ↓
         Streamlit Dashboard
```

**Components:**
- **API Server**: `http://localhost:8000` (FastAPI)
- **UI Dashboard**: `http://localhost:8501` (Streamlit)
- **Models Available**:
  - `llama3.2` - Free local Ollama model (simple queries)
  - `gemini-2.0-flash-exp` - Google Gemini (complex queries)
  - `gemini-1.5-flash` - Google Gemini (medium queries)
  - `gemini-1.5-pro` - Google Gemini (advanced)
  - `auto` - Automatic routing based on complexity

**Current Routing Logic:**
- Complexity score < 0.35 → llama3.2 (free)
- Complexity score 0.35-0.70 → gemini-2.0-flash-exp (medium)
- Complexity score > 0.70 → gemini-2.0-flash-exp (complex)

Uses **heuristic fallback** (keyword + length analysis) when ONNX model not available.

---

## 3. API Specifications

### 3.1 Available Endpoints

#### `POST /v1/chat/completions`
**Purpose**: Send chat completion requests (OpenAI-compatible)

**Request Schema:**
```json
{
  "model": "auto",  // "auto" | "llama3.2" | "gemini-2.0-flash-exp" | etc.
  "messages": [
    {
      "role": "user",  // "user" | "assistant" | "system"
      "content": "Your query here"
    }
  ],
  "temperature": 0.7,  // Optional, 0.0-1.0
  "max_tokens": null   // Optional
}
```

**Response Schema:**
```json
{
  "id": "cascade-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama3.2",  // Actual model used
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response text here"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 32,
    "completion_tokens": 15,
    "total_tokens": 47
  },
  "cascade_metadata": null  // May contain routing info
}
```

#### `GET /v1/stats`
**Purpose**: Retrieve usage statistics

**Response:**
```json
{
  "total_requests": 10,
  "total_tokens": 5000,
  "cost": {
    "actual": 0.0053,
    "baseline": 0.0070,
    "saved_dollars": 0.0017,
    "saved_percentage": 24.4
  },
  "cache": {
    "exact_hits": 2,
    "semantic_hits": 6,
    "misses": 2,
    "hit_rate": 80.0
  },
  "latency": {
    "average_ms": 1200.5,
    "p95_ms": 3500.0
  },
  "models": {
    "requests_by_model": {
      "llama3.2": 8,
      "gemini-2.0-flash-exp": 2
    },
    "costs_by_model": {
      "llama3.2": 0.0,
      "gemini-2.0-flash-exp": 0.0053
    }
  }
}
```

#### `GET /v1/models`
**Purpose**: List available models

#### `GET /health`
**Purpose**: Health check endpoint

---

## 4. Requirements

### 4.1 Functional Requirements

**FR-1: Query Diversity**
- Generate queries spanning multiple categories:
  - Simple math/arithmetic (5+3, 100-47)
  - Basic factual questions (capital of France, what is Python)
  - Greetings/casual chat (Hello, How are you)
  - Medium complexity (short explanations, definitions)
  - Complex queries (detailed explanations, essays, analysis)
  - Technical questions (programming, ML concepts)
  - Creative tasks (write stories, poems)

**FR-2: Routing Distribution**
- Target distribution:
  - 70% simple queries → llama3.2 (free)
  - 20% medium queries → gemini models
  - 10% complex queries → gemini models
- This maximizes cost savings demonstration

**FR-3: Cache Testing**
- Include intentional duplicates (20-30% of queries)
- Vary queries slightly to test semantic similarity
- Example: "What is 5+3?" then "Calculate 5 + 3"

**FR-4: Continuous Operation**
- Run indefinitely until stopped
- Configurable delay between requests (default 2-5 seconds)
- Graceful handling of API errors

**FR-5: Progress Reporting**
- Print statistics every N requests (e.g., every 10)
- Show:
  - Total requests sent
  - Current cost savings %
  - Cache hit rate
  - Model distribution

### 4.2 Non-Functional Requirements

**NFR-1: Rate Limiting**
- Don't overwhelm the API
- Default: 2-5 second delay between requests
- Should be configurable

**NFR-2: Error Handling**
- Retry failed requests (max 3 retries)
- Continue on errors, don't crash
- Log errors for debugging

**NFR-3: Realistic Patterns**
- Queries should look like real user traffic
- Mix of query types in random order
- Variable query lengths

**NFR-4: Extensibility**
- Easy to add new query templates
- Configurable via command-line args or config file

---

## 5. Technical Specifications

### 5.1 Technology Stack
- **Language**: Python 3.11+
- **HTTP Client**: `httpx` or `requests`
- **Async Support**: Optional but recommended
- **Dependencies**: Minimal (only HTTP library needed)

### 5.2 Query Templates

**Simple Query Templates (70% of traffic):**
```python
SIMPLE_QUERIES = [
    "What is {a} + {b}?",
    "Calculate {a} - {b}",
    "What is {a} times {b}?",
    "Divide {a} by {b}",
    "What is the capital of {country}?",
    "Hello!",
    "How are you?",
    "Good morning",
    "What is Python?",
    "Define {simple_term}",
    "Translate: {simple_phrase}",
    "What color is the sky?",
    "Who is the president of {country}?",
]
```

**Medium Query Templates (20% of traffic):**
```python
MEDIUM_QUERIES = [
    "Explain what {concept} means",
    "What are the benefits of {topic}?",
    "How does {technology} work?",
    "Compare {item1} and {item2}",
    "What is the difference between {term1} and {term2}?",
    "List 5 facts about {subject}",
    "Summarize {topic} in 3 sentences",
]
```

**Complex Query Templates (10% of traffic):**
```python
COMPLEX_QUERIES = [
    "Write a detailed explanation of {advanced_topic} including its history, principles, and applications",
    "Analyze the philosophical implications of {concept}",
    "Explain {scientific_theory} and how it changed our understanding of the universe",
    "Write a comprehensive guide to {technical_subject}",
    "Discuss the ethical considerations surrounding {controversial_topic}",
    "Compare and contrast {theory1} and {theory2} in detail",
    "Explain the mathematical foundations of {math_topic}",
]
```

**Suggested Variable Pools:**
```python
COUNTRIES = ["France", "Germany", "Japan", "Brazil", "Canada"]
SIMPLE_TERMS = ["API", "HTTP", "JSON", "database", "algorithm"]
CONCEPTS = ["machine learning", "blockchain", "cloud computing", "REST API"]
TECHNOLOGIES = ["Docker", "Kubernetes", "React", "PostgreSQL"]
ADVANCED_TOPICS = ["quantum mechanics", "general relativity", "neural networks",
                   "distributed systems", "quantum computing"]
SCIENTIFIC_THEORIES = ["theory of relativity", "quantum mechanics",
                       "evolution", "thermodynamics"]
```

### 5.3 Configuration Options

```python
CONFIG = {
    "api_url": "http://localhost:8000",
    "delay_min": 2.0,  # seconds
    "delay_max": 5.0,  # seconds
    "simple_ratio": 0.70,
    "medium_ratio": 0.20,
    "complex_ratio": 0.10,
    "duplicate_chance": 0.25,  # 25% chance of duplicate
    "report_interval": 10,  # print stats every 10 requests
    "max_requests": None,  # None = infinite, or set limit
}
```

### 5.4 Script Structure

```python
"""
Suggested structure:

1. Imports (httpx, random, time, etc.)
2. Configuration constants
3. Query templates and variable pools
4. Helper functions:
   - generate_simple_query()
   - generate_medium_query()
   - generate_complex_query()
   - select_query_type() # based on distribution ratios
   - send_request(query)
   - get_stats()
   - print_stats()
5. Main loop:
   - Generate query
   - Send request
   - Track history for duplicates
   - Sleep with random delay
   - Print periodic reports
6. Signal handling for graceful shutdown (Ctrl+C)
"""
```

---

## 6. Success Criteria

### 6.1 Functional Success
- ✅ Agent successfully sends 100+ requests without crashing
- ✅ Query distribution matches target (70/20/10)
- ✅ Cache hit rate reaches 25%+ (from duplicates)
- ✅ All API endpoints respond successfully
- ✅ No rate limiting errors

### 6.2 Dashboard Validation
After running agent, the dashboard should show:
- ✅ **Total Requests**: 100+
- ✅ **Cost Savings**: 15-30% (from routing to llama3.2)
- ✅ **Cache Hit Rate**: 25%+ (exact + semantic)
- ✅ **Model Distribution**:
  - llama3.2: ~70% of requests
  - gemini models: ~30% of requests
- ✅ **Charts Populated**: Cost comparison, model pie chart, cache breakdown

### 6.3 Performance Success
- ✅ Average latency < 15 seconds per request
- ✅ No memory leaks over 1000+ requests
- ✅ Error rate < 5%

---

## 7. Implementation Guidelines

### 7.1 Execution Flow

1. **Initialize**:
   - Load configuration
   - Initialize HTTP client
   - Create empty history list for duplicate tracking

2. **Main Loop**:
   ```python
   while True:
       # Decide query type (simple/medium/complex)
       query_type = select_query_type()

       # Check if should duplicate
       if should_duplicate() and history:
           query = random.choice(history)
       else:
           query = generate_query(query_type)
           history.append(query)

       # Send request
       response = send_request(query)

       # Update counters
       request_count += 1

       # Print periodic stats
       if request_count % report_interval == 0:
           print_stats()

       # Sleep
       time.sleep(random.uniform(delay_min, delay_max))
   ```

3. **Error Handling**:
   - Wrap API calls in try-except
   - Retry on connection errors (max 3 times)
   - Log but continue on failures

4. **Stats Reporting**:
   - Fetch `/v1/stats` endpoint
   - Pretty-print key metrics
   - Show progress bar or counter

### 7.2 Example Output

```
=== Cascade Data Generation Agent ===
API Endpoint: http://localhost:8000
Target: 70% simple, 20% medium, 10% complex

[Request 1] Simple: "What is 5 + 3?" → llama3.2 ✓
[Request 2] Medium: "Explain machine learning" → gemini-2.0-flash-exp ✓
[Request 3] Simple: "Hello!" → llama3.2 ✓
[Request 4] DUPLICATE: "What is 5 + 3?" → CACHED ⚡
...

--- Stats Report (10 requests) ---
Total Requests: 10
Cost Savings: $0.0045 (32.1%)
Cache Hit Rate: 30.0% (3/10)
Model Distribution:
  - llama3.2: 7 (70%)
  - gemini-2.0-flash-exp: 3 (30%)
Average Latency: 8,234ms
----------------------------------
```

---

## 8. Current System State

### 8.1 Deployed Services
```bash
# All services running via docker-compose
cascade-api:        http://localhost:8000 (FastAPI)
cascade-ui:         http://localhost:8501 (Streamlit)
redis:              localhost:6379 (cache)
qdrant:             localhost:6333 (vector DB)
ollama:             http://host.docker.internal:11434 (local LLM)
```

### 8.2 Environment Variables
```bash
GEMINI_API_KEY=AIzaSyCGwcpk4vT1DP7nu3YDwp2cU9HnIsUsMaM
OLLAMA_BASE_URL=http://host.docker.internal:11434
REDIS_HOST=redis
QDRANT_URL=http://qdrant:6333
SIMILARITY_THRESHOLD=0.92
CACHE_TTL=3600
```

### 8.3 Known Limitations
- ONNX classifier not loaded (using heuristic fallback)
- Heuristics use keyword matching + length analysis
- Semantic cache works but similarity threshold is conservative (0.92)
- Gemini has deprecated `google.generativeai` package (still works, shows warning)

---

## 9. Acceptance Criteria

### Must Have
- [ ] Script runs without crashing for 100+ requests
- [ ] Supports configurable query distribution (70/20/10)
- [ ] Includes duplicate queries for cache testing
- [ ] Prints periodic statistics
- [ ] Handles API errors gracefully
- [ ] Uses `model: "auto"` for intelligent routing

### Should Have
- [ ] Command-line arguments for configuration
- [ ] Colorful terminal output (optional)
- [ ] Progress bar (optional)
- [ ] Async requests for better performance
- [ ] Save history to file for analysis

### Nice to Have
- [ ] Web UI for controlling the agent
- [ ] Real-time dashboard updates
- [ ] Query generation using LLM itself (meta!)
- [ ] A/B testing different routing strategies

---

## 10. Testing & Validation

### 10.1 Manual Testing
1. Start Cascade services: `docker compose up -d`
2. Run data generation agent
3. Observe dashboard at http://localhost:8501
4. Verify metrics update in real-time
5. Check `/v1/stats` endpoint for accuracy

### 10.2 Expected Results (after 100 requests)
```json
{
  "total_requests": 100,
  "total_tokens": ~30000,
  "cost": {
    "actual": ~0.015,
    "baseline": ~0.025,
    "saved_percentage": ~40%
  },
  "cache": {
    "hit_rate": ~30%
  },
  "models": {
    "requests_by_model": {
      "llama3.2": 70,
      "gemini-2.0-flash-exp": 30
    }
  }
}
```

---

## 11. File Location

**Recommended Path**: `/Users/ayush/PROJECTS/AI_ML/AI_Tools/cascade/scripts/generate_data.py`

**Usage**:
```bash
cd /Users/ayush/PROJECTS/AI_ML/AI_Tools/cascade
python scripts/generate_data.py

# Or with options
python scripts/generate_data.py --requests 500 --delay 3 --verbose
```

---

## 12. Additional Resources

### 12.1 Curl Examples

**Simple Query:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

**Complex Query:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain quantum mechanics"}]}'
```

**Get Stats:**
```bash
curl http://localhost:8000/v1/stats | jq
```

### 12.2 Related Files
- API Routes: `/Users/ayush/PROJECTS/AI_ML/AI_Tools/cascade/src/cascade/api/routes.py`
- Routing Engine: `/Users/ayush/PROJECTS/AI_ML/AI_Tools/cascade/src/cascade/router/routing_engine.py`
- Cost Tracker: `/Users/ayush/PROJECTS/AI_ML/AI_Tools/cascade/src/cascade/cost/tracker.py`
- Dashboard: `/Users/ayush/PROJECTS/AI_ML/AI_Tools/cascade/src/cascade/ui/components/dashboard.py`

---

## 13. Notes

- **Why this agent?** Dashboard looks empty with only 10-20 requests. Need 100+ requests to show meaningful trends and validate cost savings.
- **Realistic data**: Agent should simulate real user behavior, not just spam random queries.
- **Demo-ready**: After running agent, dashboard should be impressive enough to demo to stakeholders.

---

## Appendix A: Query Categories

### Simple Queries (Target: 70%)
- Basic arithmetic
- Greetings
- Yes/no questions
- Single-word answers
- Common facts (capitals, dates)

### Medium Queries (Target: 20%)
- Short explanations (2-3 sentences)
- Comparisons
- Lists (top 5, best practices)
- How-to questions (simple)

### Complex Queries (Target: 10%)
- Detailed analysis
- Essays (200+ words)
- Multi-step reasoning
- Technical deep-dives
- Philosophical discussions

---

**End of PRD**
