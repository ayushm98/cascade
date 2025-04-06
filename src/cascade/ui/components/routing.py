"""Routing visualization component."""

import streamlit as st
import plotly.graph_objects as go
import httpx
from typing import Optional

# Import local router for demo
try:
    from cascade.router import route_query, classify_by_heuristics
    HAS_ROUTER = True
except ImportError:
    HAS_ROUTER = False


def classify_query_demo(query: str) -> dict:
    """Classify query using local router or heuristics."""
    if HAS_ROUTER:
        try:
            import asyncio
            result = asyncio.run(route_query(query))
            return {
                "score": result.complexity_score,
                "label": result.complexity_label,
                "model": result.recommended_model,
                "reason": result.routing_reason,
            }
        except Exception:
            pass

    # Fallback to simple heuristics
    score, label = classify_by_heuristics(query) if HAS_ROUTER else (0.5, "medium")
    models = {"simple": "llama3.2", "medium": "gpt-4o-mini", "complex": "gpt-4o"}
    return {
        "score": score,
        "label": label,
        "model": models.get(label, "gpt-4o-mini"),
        "reason": "Classified using heuristics",
    }


def render_complexity_gauge(score: float):
    """Render a gauge chart for complexity score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Complexity Score"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#667eea"},
            "steps": [
                {"range": [0, 35], "color": "#27ae60"},
                {"range": [35, 70], "color": "#f39c12"},
                {"range": [70, 100], "color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def render_routing_demo():
    """Render the routing demonstration page."""
    st.markdown('<h1 class="main-header">Routing Demo</h1>', unsafe_allow_html=True)
    st.markdown(
        "See how Cascade classifies query complexity and routes to the optimal model."
    )

    # Example queries
    st.markdown("### Try Example Queries")
    examples = {
        "Simple": "What is the capital of France?",
        "Medium": "Explain the difference between TCP and UDP protocols.",
        "Complex": "Write a Python function that implements a binary search tree with insert, delete, and search operations, including balancing.",
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🟢 Simple Query"):
            st.session_state["demo_query"] = examples["Simple"]
    with col2:
        if st.button("🟡 Medium Query"):
            st.session_state["demo_query"] = examples["Medium"]
    with col3:
        if st.button("🔴 Complex Query"):
            st.session_state["demo_query"] = examples["Complex"]

    st.divider()

    # Query input
    query = st.text_area(
        "Enter a query to classify",
        value=st.session_state.get("demo_query", ""),
        height=100,
        placeholder="Type or select an example query above...",
    )

    if st.button("Analyze Query", type="primary") or query:
        if query:
            with st.spinner("Analyzing..."):
                result = classify_query_demo(query)

            # Display results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Classification Result")
                st.plotly_chart(
                    render_complexity_gauge(result["score"]),
                    use_container_width=True,
                )

            with col2:
                st.markdown("### Routing Decision")
                st.markdown(f"**Complexity Label:** `{result['label'].upper()}`")
                st.markdown(f"**Recommended Model:** `{result['model']}`")
                st.markdown(f"**Reasoning:** {result['reason']}")

                # Model info
                model_info = {
                    "llama3.2": ("🟢", "Free (Local)", "~50ms"),
                    "gpt-4o-mini": ("🟡", "$0.15/1M tokens", "~200ms"),
                    "gpt-4o": ("🔴", "$2.50/1M tokens", "~500ms"),
                }
                info = model_info.get(result["model"], ("⚪", "Unknown", "Unknown"))
                st.markdown(f"""
                    **Model Details:**
                    - Status: {info[0]}
                    - Cost: {info[1]}
                    - Typical Latency: {info[2]}
                """)

            # Explanation
            st.divider()
            st.markdown("### How It Works")
            st.markdown("""
                1. **Query Analysis**: The ML classifier (DistilBERT) analyzes the query text
                2. **Complexity Score**: Outputs a score from 0.0 (simple) to 1.0 (complex)
                3. **Threshold Routing**:
                   - Score < 0.35 → Route to local Llama (free)
                   - Score 0.35-0.70 → Route to GPT-4o-mini (cheap)
                   - Score > 0.70 → Route to GPT-4o (powerful)
                4. **Cost Savings**: Simple queries use free/cheap models, saving 60%+ on API costs
            """)
