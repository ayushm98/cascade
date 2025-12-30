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

## What is Cascade?

Cascade is an intelligent proxy that automatically routes LLM requests to the most cost-effective model based on query complexity:

- **Simple queries** → Free local models (Ollama) or GPT-3.5
- **Medium queries** → GPT-4o-mini ($0.15/1M tokens)
- **Complex queries** → GPT-4o ($2.50/1M tokens)

## Features

- 🎯 **ML-Powered Routing** - Predicts query complexity in <20ms
- 💰 **60%+ Cost Savings** - Routes simple queries to cheaper models
- ⚡ **Semantic Caching** - Vector similarity search for cached responses
- 📊 **Real-Time Analytics** - Dashboard showing savings and usage metrics
- 🔌 **OpenAI Compatible** - Drop-in replacement for OpenAI API

## Using This Space

This Space provides a demo UI where you can:

1. **Test Routing** - See how different queries get routed to different models
2. **View Analytics** - Track cost savings and cache hit rates
3. **Interactive Chat** - Try example queries and see real-time responses

## Configuration

To use this with real LLM APIs, set these environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key
- `OLLAMA_BASE_URL` - Ollama server URL (optional)
- `REDIS_URL` - Redis connection for caching (optional)
- `QDRANT_URL` - Qdrant server for semantic cache (optional)

## Learn More

- [GitHub Repository](https://github.com/ayushm98/cascade)
- [Documentation](https://github.com/ayushm98/cascade#readme)
- [Contributing Guide](https://github.com/ayushm98/cascade/blob/main/CONTRIBUTING.md)

## Architecture

```
Request → Cache Check → ML Classifier → Route to Model
           ↓              ↓                ↓
       Semantic       Simple/Med/      Ollama/GPT-4o-mini/
         Cache        Complex          GPT-4o
```

## Tech Stack

- **Backend**: FastAPI, Python 3.11+
- **ML**: DistilBERT (ONNX), Sentence Transformers
- **Caching**: Redis (exact match), Qdrant (semantic)
- **UI**: Streamlit, Plotly
- **Deployment**: Docker, Hugging Face Spaces

---

Built with ❤️ by [ayushm98](https://github.com/ayushm98)
