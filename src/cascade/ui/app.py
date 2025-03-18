"""Main Streamlit application for Cascade demo."""

import streamlit as st

from cascade.ui.components.sidebar import render_sidebar
from cascade.ui.components.dashboard import render_dashboard
from cascade.ui.components.chat import render_chat
from cascade.ui.components.routing import render_routing_demo


def main():
    """Main entry point for Streamlit app."""
    st.set_page_config(
        page_title="Cascade - LLM Router",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 1rem;
            color: white;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Render sidebar for navigation and settings
    page = render_sidebar()

    # Main content based on selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Chat":
        render_chat()
    elif page == "Routing Demo":
        render_routing_demo()


if __name__ == "__main__":
    main()
