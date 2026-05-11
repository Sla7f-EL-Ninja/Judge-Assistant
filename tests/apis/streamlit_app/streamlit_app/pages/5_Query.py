"""Query + SSE streaming page."""

import streamlit as st
from utils.display import show_sse_events

st.title("Supervisor Query")
st.markdown("`POST /api/v1/query` -- Server-Sent Events stream")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

query_text = st.text_area("Query", height=100, placeholder="Enter your legal question...")

col1, col2 = st.columns(2)
case_id = col1.text_input("Case ID", value=st.session_state.get("last_case_id", ""))
conversation_id = col2.text_input(
    "Conversation ID (optional)",
    value=st.session_state.get("last_conversation_id", ""),
)

if st.button("Submit Query"):
    if not query_text.strip():
        st.error("Query text is required.")
    elif not case_id.strip():
        st.error("Case ID is required.")
    else:
        with st.spinner("Running supervisor graph..."):
            status, events, elapsed = client.query(
                query_text.strip(),
                case_id.strip(),
                conversation_id.strip() or None,
            )

        st.markdown(f"**Status:** {status} &nbsp; | &nbsp; **Time:** {elapsed:.0f} ms")
        st.divider()

        show_sse_events(events)

        # Extract conversation_id from result event
        for evt in events:
            if evt.get("event") == "result":
                data = evt.get("data", {})
                if isinstance(data, dict) and "conversation_id" in data:
                    st.session_state["last_conversation_id"] = data["conversation_id"]
                    st.info(f"Conversation ID stored: {data['conversation_id']}")

                # Show the final response prominently
                if isinstance(data, dict) and "final_response" in data:
                    st.divider()
                    st.subheader("Final Response")
                    st.markdown(data["final_response"])
