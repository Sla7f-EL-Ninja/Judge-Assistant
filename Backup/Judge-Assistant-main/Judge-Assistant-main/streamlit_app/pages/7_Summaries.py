"""Summaries page."""

import streamlit as st
from utils.display import show_response

st.title("Case Summaries")
st.markdown("`GET /api/v1/cases/{case_id}/summary`")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

case_id = st.text_input(
    "Case ID", value=st.session_state.get("last_case_id", "")
)

if st.button("Fetch Summary"):
    if not case_id.strip():
        st.error("Case ID is required.")
    else:
        status, body, elapsed = client.get_summary(case_id.strip())
        show_response(status, body, elapsed)

        if status == 200 and "summary" in body:
            st.divider()
            st.subheader("Summary")
            st.markdown(body["summary"])

            if body.get("sources"):
                st.subheader("Sources")
                for src in body["sources"]:
                    st.markdown(f"- {src}")
