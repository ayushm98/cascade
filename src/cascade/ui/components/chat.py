"""Chat interface component for testing Cascade."""

import streamlit as st
import httpx
import time
from typing import Optional


def send_message(api_url: str, message: str, model: str) -> Optional[dict]:
    """Send a message to Cascade API."""
    try:
        response = httpx.post(
            f"{api_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "temperature": 0.7,
            },
            timeout=30.0,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


def render_chat():
    """Render the chat interface."""
    st.markdown('<h1 class="main-header">Chat</h1>', unsafe_allow_html=True)
    st.markdown("Test the Cascade router with real queries.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_stats" not in st.session_state:
        st.session_state.chat_stats = {
            "total_cost": 0.0,
            "total_tokens": 0,
            "requests": 0,
        }

    # Settings
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("### Session Stats")
        st.metric("Requests", st.session_state.chat_stats["requests"])
        st.metric("Tokens", st.session_state.chat_stats["total_tokens"])
        st.metric("Cost", f"${st.session_state.chat_stats['total_cost']:.4f}")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_stats = {
                "total_cost": 0.0,
                "total_tokens": 0,
                "requests": 0,
            }
            st.rerun()

    with col1:
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and "metadata" in msg:
                        meta = msg["metadata"]
                        st.caption(
                            f"Model: {meta.get('model', 'unknown')} | "
                            f"Tokens: {meta.get('tokens', 0)} | "
                            f"Latency: {meta.get('latency_ms', 0):.0f}ms"
                        )

        # Chat input
        if prompt := st.chat_input("Type your message..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            api_url = st.session_state.get("api_url", "http://localhost:8000")
            model = st.session_state.get("default_model", "auto")

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    response = send_message(api_url, prompt, model)
                    latency = (time.time() - start_time) * 1000

                if response:
                    content = response["choices"][0]["message"]["content"]
                    st.markdown(content)

                    # Extract metadata
                    usage = response.get("usage", {})
                    tokens = usage.get("total_tokens", 0)
                    model_used = response.get("model", model)

                    metadata = {
                        "model": model_used,
                        "tokens": tokens,
                        "latency_ms": latency,
                    }

                    st.caption(
                        f"Model: {model_used} | "
                        f"Tokens: {tokens} | "
                        f"Latency: {latency:.0f}ms"
                    )

                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": content,
                        "metadata": metadata,
                    })

                    # Update stats
                    st.session_state.chat_stats["requests"] += 1
                    st.session_state.chat_stats["total_tokens"] += tokens
                else:
                    # Demo response if API not available
                    demo_response = "I'm a demo response. Connect to the Cascade API to get real responses!"
                    st.markdown(demo_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": demo_response,
                        "metadata": {"model": "demo", "tokens": 0, "latency_ms": 0},
                    })
