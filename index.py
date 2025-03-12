import streamlit as st

st.set_page_config(
    page_title="Overview",
    page_icon="🔍",
)

# This hack tries to change how the page appears in the sidebar
import streamlit.components.v1 as components

# Inject custom CSS to change the sidebar text
components.html(
    """
    <style>
    [data-testid="stSidebarNav"] li:first-child a div:first-child {
        display: none;
    }
    [data-testid="stSidebarNav"] li:first-child a div:nth-child(2)::before {
        content: "Overview";
    }
    </style>
    """,
    height=0
)