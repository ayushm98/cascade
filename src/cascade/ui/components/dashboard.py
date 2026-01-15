"""Dashboard component with cost metrics and analytics."""

import streamlit as st
import httpx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta


def fetch_stats(api_url: str) -> dict:
    """Fetch stats from Cascade API."""
    try:
        response = httpx.get(f"{api_url}/v1/stats", timeout=5.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def render_dashboard():
    """Render the main dashboard with cost metrics."""
    st.markdown('<h1 class="main-header">Dashboard</h1>', unsafe_allow_html=True)

    api_url = st.session_state.get("api_url", "http://localhost:8000")
    stats = fetch_stats(api_url)

    if stats is None:
        # Show demo data if API not available
        stats = {
            "total_requests": 1247,
            "total_tokens": 523400,
            "cost": {
                "actual": 2.34,
                "baseline": 7.89,
                "saved_dollars": 5.55,
                "saved_percentage": 70.3,
            },
            "cache": {
                "exact_hits": 342,
                "semantic_hits": 189,
                "misses": 716,
                "hit_rate": 42.6,
            },
            "latency": {
                "average_ms": 245.3,
                "p95_ms": 892.1,
            },
            "models": {
                "requests_by_model": {
                    "llama3.2": 534,
                    "gpt-4o-mini": 412,
                    "gpt-4o": 301,
                },
                "costs_by_model": {
                    "llama3.2": 0.0,
                    "gpt-4o-mini": 0.89,
                    "gpt-4o": 1.45,
                },
            },
        }
        st.info("📊 Showing demo data. Connect to Cascade API for live metrics.")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Requests",
            value=f"{stats['total_requests']:,}",
            delta=None,
        )

    with col2:
        # Show more decimal places for small amounts
        saved = stats['cost']['saved_dollars']
        if saved < 0.01:
            saved_str = f"${saved:.4f}"
        else:
            saved_str = f"${saved:.2f}"

        st.metric(
            label="Cost Savings",
            value=saved_str,
            delta=f"{stats['cost']['saved_percentage']:.1f}%",
        )

    with col3:
        st.metric(
            label="Cache Hit Rate",
            value=f"{stats['cache']['hit_rate']:.1f}%",
            delta=None,
        )

    with col4:
        st.metric(
            label="Avg Latency",
            value=f"{stats['latency']['average_ms']:.0f}ms",
            delta=None,
        )

    st.divider()

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Cost Comparison")
        cost_data = pd.DataFrame({
            "Type": ["Actual Cost", "Baseline Cost"],
            "Cost ($)": [stats["cost"]["actual"], stats["cost"]["baseline"]],
        })
        fig = px.bar(
            cost_data,
            x="Type",
            y="Cost ($)",
            color="Type",
            color_discrete_map={
                "Actual Cost": "#667eea",
                "Baseline Cost": "#e74c3c",
            },
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Requests by Model")
        model_data = stats["models"]["requests_by_model"]
        fig = px.pie(
            values=list(model_data.values()),
            names=list(model_data.keys()),
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Purples_r,
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Cache breakdown
    st.markdown("### Cache Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Exact Hits", stats["cache"]["exact_hits"])
    with col2:
        st.metric("Semantic Hits", stats["cache"]["semantic_hits"])
    with col3:
        st.metric("Cache Misses", stats["cache"]["misses"])

    # Cache hit types chart
    cache_data = pd.DataFrame({
        "Type": ["Exact Hits", "Semantic Hits", "Misses"],
        "Count": [
            stats["cache"]["exact_hits"],
            stats["cache"]["semantic_hits"],
            stats["cache"]["misses"],
        ],
    })
    fig = px.bar(
        cache_data,
        x="Type",
        y="Count",
        color="Type",
        color_discrete_map={
            "Exact Hits": "#27ae60",
            "Semantic Hits": "#3498db",
            "Misses": "#95a5a6",
        },
    )
    fig.update_layout(showlegend=False, height=250)
    st.plotly_chart(fig, use_container_width=True)
