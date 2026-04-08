"""
display.py

Helper functions for rendering API responses in Streamlit.
"""

import json

import streamlit as st


def _status_color(status_code: int) -> str:
    """Return a CSS-friendly color string for the given HTTP status code."""
    if 200 <= status_code < 300:
        return "green"
    if 400 <= status_code < 500:
        return "orange"
    if status_code >= 500:
        return "red"
    if status_code == 0:
        return "red"
    return "gray"


def show_response(status_code: int, body: dict, elapsed: float):
    """Display an API response with color-coded status badge and formatted JSON."""
    color = _status_color(status_code)
    st.markdown(
        f"**Status:** :{color}[{status_code}] &nbsp; | &nbsp; "
        f"**Time:** {elapsed:.0f} ms"
    )
    if "error" in body:
        st.error(f"Error: {body['error'].get('code', 'UNKNOWN')} -- {body['error'].get('detail', '')}")
    st.json(body)


def show_error(status_code: int, body: dict):
    """Display an error response with red highlight."""
    st.error(f"HTTP {status_code}")
    if "error" in body:
        err = body["error"]
        st.markdown(f"**Code:** `{err.get('code', 'N/A')}`")
        st.markdown(f"**Detail:** {err.get('detail', 'N/A')}")
    st.json(body)


def show_sse_events(events: list[dict]):
    """Display SSE events in an expandable list."""
    if not events:
        st.warning("No SSE events received.")
        return

    for i, evt in enumerate(events):
        event_type = evt.get("event", "unknown")
        data = evt.get("data", {})

        if event_type == "result":
            st.success(f"Event {i + 1}: **{event_type}**")
            st.json(data)
        elif event_type == "error":
            st.error(f"Event {i + 1}: **{event_type}**")
            st.json(data)
        elif event_type == "done":
            st.info(f"Event {i + 1}: **{event_type}** (stream complete)")
        else:
            with st.expander(f"Event {i + 1}: {event_type}", expanded=False):
                st.json(data)
