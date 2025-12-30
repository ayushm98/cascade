---
title: Cascade - Intelligent LLM Router
emoji: 🌊
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.31.0"
app_file: app.py
pinned: false
---

# Cascade 🌊

**Intelligent LLM Request Router** - Reduce API costs by 60%+ through smart routing and semantic caching.

[![CI](https://github.com/ayushm98/cascade/actions/workflows/ci.yml/badge.svg)](https://github.com/ayushm98/cascade/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 🚀 Try It Live

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/ayushm98/cascade)

Experience Cascade's intelligent routing and cost optimization in action!

## Overview

Cascade is an intelligent LLM proxy that automatically routes requests to the most cost-effective model based on query complexity. Simple queries go to free local models (Ollama), while complex queries are routed to powerful cloud models (GPT-4o).

### Key Features

- **ML-Powered Routing**: Fine-tuned DistilBERT classifier predicts query complexity in <20ms
- **Semantic Caching**: Vector similarity search finds cached responses for similar queries
- **OpenAI Compatible**: Drop-in replacement for OpenAI API
- **Cost Analytics**: Real-time dashboard showing savings and usage metrics
- **60%+ Cost Reduction**: Typical savings by routing simple queries to free models

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Cascade                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Request ──► Semantic Cache ──► Cache Hit? ──► Return      │
│                     │                                        │
│                     ▼ (miss)                                 │
│              ML Classifier                                   │
│                     │                                        │
│         ┌──────────┼──────────┐                             │
│         ▼          ▼          ▼                             │
│      Simple     Medium     Complex                          │
│         │          │          │                             │
│         ▼          ▼          ▼                             │
│     Llama3.2  GPT-4o-mini  GPT-4o                          │
│      (free)   ($0.15/1M)  ($2.50/1M)                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- Ollama (for local models)
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/ayushm98/cascade.git
cd cascade

# Install dependencies
pip install poetry
poetry install

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Running with Docker

```bash
# Start all services
docker-compose up -d

# API available at http://localhost:8000
# UI available at http://localhost:8501
```

### Running Locally

```bash
# Start the API server
poetry run uvicorn cascade.api.main:app --reload

# Start the Streamlit UI (in another terminal)
poetry run streamlit run src/cascade/ui/app.py
```

## Usage

### API Usage

Cascade is OpenAI-compatible. Just change your base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Uses your configured key
)

# Automatic routing based on complexity
response = client.chat.completions.create(
    model="auto",  # Let Cascade choose the best model
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

### Forcing a Specific Model

```python
# Force GPT-4o for complex tasks
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a compiler..."}]
)
```

### Checking Stats

```bash
curl http://localhost:8000/v1/stats
```

```json
{
  "total_requests": 1247,
  "cost": {
    "actual": 2.34,
    "baseline": 7.89,
    "saved_dollars": 5.55,
    "saved_percentage": 70.3
  },
  "cache": {
    "hit_rate": 42.6
  }
}
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `REDIS_HOST` | `localhost` | Redis host |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `SIMILARITY_THRESHOLD` | `0.92` | Semantic cache threshold |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |

## Project Structure

```
cascade/
├── src/cascade/
│   ├── api/           # FastAPI application
│   ├── cache/         # Redis + Qdrant caching
│   ├── cost/          # Cost tracking & analytics
│   ├── providers/     # LLM provider adapters
│   ├── router/        # ML classifier & routing
│   └── ui/            # Streamlit dashboard
├── ml/                # ML training pipeline
│   ├── data/          # Dataset loading
│   ├── training/      # Model training
│   └── export/        # ONNX conversion
├── tests/             # Test suite
└── docker-compose.yml
```

## How It Works

1. **Request Arrives**: User sends a chat completion request
2. **Cache Check**: Check semantic cache for similar previous queries
3. **Complexity Classification**: ML model predicts query complexity (0-1)
4. **Routing Decision**:
   - Score < 0.35 → Ollama (free)
   - Score 0.35-0.70 → GPT-4o-mini ($0.15/1M tokens)
   - Score > 0.70 → GPT-4o ($2.50/1M tokens)
5. **Response**: Forward to selected model, cache result, return

## Development

```bash
# Run tests
make test

# Run linting
make lint

# Format code
make format

# Train the classifier
make train

# Export to ONNX
make export-onnx
```

## Deployment

### Railway (Recommended)

Railway offers the easiest deployment with automatic builds:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Set environment variables in Railway dashboard:
# - OPENAI_API_KEY
# - REDIS_URL (add Redis plugin)
```

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/cascade)

### Fly.io

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login and deploy
fly auth login
fly launch
fly secrets set OPENAI_API_KEY=sk-your-key
fly deploy
```

### Render

1. Fork this repository
2. Connect to Render
3. Use the `render.yaml` blueprint
4. Set `OPENAI_API_KEY` in environment variables

### Docker (Self-hosted)

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t cascade .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx cascade
```

### Environment Variables for Production

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `REDIS_URL` | No | Redis connection URL (for caching) |
| `QDRANT_URL` | No | Qdrant URL (for semantic cache) |
| `PORT` | No | Server port (default: 8000) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/models` | GET | List available models |
| `/v1/stats` | GET | Usage statistics |

## Contributing

Contributions are welcome! Please read our contributing guidelines first.

## License

MIT License - see [LICENSE](LICENSE) for details.
