"""Health check page."""

import streamlit as st
from utils.display import show_response

st.title("Health Check")
st.markdown("`GET /api/v1/health` -- No authentication required")

if st.button("Check Health"):
    client = st.session_state.get("client")
    if not client:
        st.error("Configure API connection in the sidebar first.")
    else:
        status, body, elapsed = client.health()
        show_response(status, body, elapsed)
