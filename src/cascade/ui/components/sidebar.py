"""Sidebar component for navigation and settings."""

import streamlit as st
import httpx


def check_api_connection(api_url: str) -> bool:
    """Check if API is reachable."""
    try:
        response = httpx.get(f"{api_url}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def render_sidebar() -> str:
    """
    Render the sidebar with navigation and settings.

    Returns:
        Selected page name
    """
    with st.sidebar:
        st.markdown("## 🌊 Cascade")
        st.markdown("*Intelligent LLM Router*")
        st.divider()

        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Chat", "Routing Demo"],
            label_visibility="collapsed",
        )

        st.divider()

        # Settings
        st.markdown("### Settings")

        # API endpoint configuration
        api_url = st.text_input(
            "API Endpoint",
            value="http://localhost:8889",
            help="Cascade API server URL",
        )
        st.session_state["api_url"] = api_url

        # Connection status
        is_connected = check_api_connection(api_url)
        if is_connected:
            st.success("🟢 Connected")
        else:
            st.error("🔴 Disconnected")

        # Model selection
        default_model = st.selectbox(
            "Default Model",
            ["auto", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "llama3.2"],
            help="Select default model or 'auto' for smart routing",
        )
        st.session_state["default_model"] = default_model

        # Cache settings
        st.markdown("#### Cache")
        use_cache = st.toggle("Enable Semantic Cache", value=True)
        st.session_state["use_cache"] = use_cache

        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.8,
            max_value=0.99,
            value=0.92,
            step=0.01,
            help="Minimum similarity for cache hits",
        )
        st.session_state["similarity_threshold"] = similarity_threshold

        st.divider()

        # Footer
        st.markdown("---")
        st.markdown(
            "<small>Built with Streamlit</small>",
            unsafe_allow_html=True,
        )

    return page
