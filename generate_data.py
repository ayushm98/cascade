#!/usr/bin/env python3
"""
Cascade Data Generation Script

Uses Gemini API to generate diverse queries and send them to Cascade API
to populate the dashboard with realistic data.

Usage:
    python generate_data.py
"""

import time
import random
import httpx
import google.generativeai as genai
from typing import List, Dict

# Configuration
GEMINI_API_KEY = "AIzaSyCGwcpk4vT1DP7nu3YDwp2cU9HnIsUsMaM"
CASCADE_API_URL = "http://localhost:8000"
DELAY_MIN = 2.0  # seconds between requests
DELAY_MAX = 5.0
REPORT_INTERVAL = 10  # print stats every N requests
TARGET_REQUESTS = 100  # total requests to send

# Query generation prompts for Gemini
QUERY_PROMPTS = {
    "simple": [
        "Generate a simple math question (single operation like 5+3)",
        "Generate a basic factual question (like 'What is the capital of France?')",
        "Generate a casual greeting or small talk phrase",
        "Generate a simple yes/no question",
        "Generate a one-word definition request (like 'Define API')",
    ],
    "medium": [
        "Generate a question asking for a brief explanation (2-3 sentences) of a common concept",
        "Generate a question comparing two similar things",
        "Generate a question asking for a list of 5 items",
        "Generate a how-to question about a simple task",
        "Generate a question about the benefits of something",
    ],
    "complex": [
        "Generate a question asking for a detailed explanation of an advanced scientific concept",
        "Generate a question about philosophical implications of a technology",
        "Generate a request for a comprehensive analysis or essay",
        "Generate a question about the history and evolution of a complex topic",
        "Generate a request to compare and contrast multiple theories in detail",
    ],
}

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Tracking
query_history = []
request_count = 0
start_time = time.time()


def generate_query(complexity: str) -> str:
    """Generate a query using Gemini based on complexity level."""
    prompt_templates = QUERY_PROMPTS[complexity]
    prompt = random.choice(prompt_templates)

    full_prompt = f"{prompt}. Return ONLY the question, nothing else. No explanation, no quotes, just the raw question."

    try:
        response = model.generate_content(full_prompt)
        query = response.text.strip().strip('"').strip("'")
        return query
    except Exception as e:
        print(f"⚠️  Gemini generation error: {e}")
        # Fallback to hardcoded queries
        fallbacks = {
            "simple": ["What is 5 + 3?", "Hello!", "What is Python?"],
            "medium": ["Explain what an API is", "What are the benefits of Docker?"],
            "complex": ["Explain the theory of relativity in detail", "Write an essay on AI ethics"],
        }
        return random.choice(fallbacks[complexity])


def send_to_cascade(query: str) -> Dict:
    """Send a query to Cascade API and return response."""
    url = f"{CASCADE_API_URL}/v1/chat/completions"

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
    }

    try:
        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Cascade API error: {e}")
        return None


def get_stats() -> Dict:
    """Fetch current stats from Cascade."""
    try:
        response = httpx.get(f"{CASCADE_API_URL}/v1/stats", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️  Stats fetch error: {e}")
        return None


def print_stats():
    """Print formatted statistics."""
    stats = get_stats()
    if not stats:
        return

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"📊 STATISTICS (after {request_count} requests, {elapsed:.0f}s)")
    print("="*60)
    print(f"💰 Cost Savings: ${stats['cost']['saved_dollars']:.4f} ({stats['cost']['saved_percentage']:.1f}%)")
    print(f"📦 Cache Hit Rate: {stats['cache']['hit_rate']:.1f}%")
    print(f"   - Exact Hits: {stats['cache']['exact_hits']}")
    print(f"   - Semantic Hits: {stats['cache']['semantic_hits']}")
    print(f"   - Misses: {stats['cache']['misses']}")
    print(f"⚡ Avg Latency: {stats['latency']['average_ms']:.0f}ms")
    print(f"🤖 Model Distribution:")
    for model, count in stats['models']['requests_by_model'].items():
        cost = stats['models']['costs_by_model'][model]
        percentage = (count / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
        print(f"   - {model}: {count} ({percentage:.1f}%) | ${cost:.4f}")
    print("="*60 + "\n")


def select_complexity() -> str:
    """Select query complexity based on target distribution (70/20/10)."""
    rand = random.random()
    if rand < 0.70:
        return "simple"
    elif rand < 0.90:
        return "medium"
    else:
        return "complex"


def should_duplicate() -> bool:
    """Decide if we should send a duplicate query (25% chance)."""
    return random.random() < 0.25 and len(query_history) > 0


def main():
    """Main execution loop."""
    global request_count

    print("="*60)
    print("🌊 CASCADE DATA GENERATION AGENT")
    print("="*60)
    print(f"Target: {TARGET_REQUESTS} requests")
    print(f"Distribution: 70% simple, 20% medium, 10% complex")
    print(f"Duplicate rate: 25%")
    print(f"Delay: {DELAY_MIN}-{DELAY_MAX}s between requests")
    print("="*60 + "\n")

    # Check API health
    try:
        health = httpx.get(f"{CASCADE_API_URL}/health", timeout=5.0)
        if health.status_code != 200:
            print("❌ Cascade API is not healthy!")
            return
        print("✅ Cascade API is healthy\n")
    except Exception as e:
        print(f"❌ Cannot connect to Cascade API: {e}")
        return

    try:
        while request_count < TARGET_REQUESTS:
            request_count += 1

            # Decide if duplicate or new query
            if should_duplicate():
                query = random.choice(query_history)
                query_type = "DUPLICATE"
                icon = "🔁"
            else:
                complexity = select_complexity()
                print(f"[{request_count}/{TARGET_REQUESTS}] Generating {complexity} query...", end=" ")
                query = generate_query(complexity)
                query_history.append(query)
                query_type = complexity.upper()
                icon = "🔵" if complexity == "simple" else "🟡" if complexity == "medium" else "🔴"

            # Truncate query for display
            display_query = query[:60] + "..." if len(query) > 60 else query
            print(f"\n[{request_count}/{TARGET_REQUESTS}] {icon} {query_type}: \"{display_query}\"")

            # Send to Cascade
            response = send_to_cascade(query)

            if response:
                model_used = response.get("model", "unknown")
                tokens = response.get("usage", {}).get("total_tokens", 0)
                print(f"    ✓ Routed to: {model_used} | Tokens: {tokens}")
            else:
                print(f"    ✗ Request failed")

            # Print stats periodically
            if request_count % REPORT_INTERVAL == 0:
                print_stats()

            # Sleep between requests
            if request_count < TARGET_REQUESTS:
                delay = random.uniform(DELAY_MIN, DELAY_MAX)
                print(f"    ⏳ Sleeping {delay:.1f}s...\n")
                time.sleep(delay)

        # Final stats
        print("\n" + "="*60)
        print("🎉 COMPLETE! All requests sent.")
        print("="*60)
        print_stats()
        print("📊 View dashboard at: http://localhost:8501")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Showing final stats...\n")
        print_stats()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
