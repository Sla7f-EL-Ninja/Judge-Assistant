"""Conversations page."""

import streamlit as st
from utils.display import show_response

st.title("Conversations")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

tab_list, tab_get, tab_delete = st.tabs(["List", "Get", "Delete"])

# -- List ---------------------------------------------------------------------
with tab_list:
    st.subheader("List Conversations for Case")
    case_id = st.text_input(
        "Case ID", value=st.session_state.get("last_case_id", ""), key="list_conv_case"
    )
    col1, col2 = st.columns(2)
    skip = col1.number_input("Skip", min_value=0, value=0, step=1, key="conv_skip")
    limit = col2.number_input("Limit", min_value=1, max_value=100, value=20, step=1, key="conv_limit")

    if st.button("List Conversations"):
        if not case_id.strip():
            st.error("Case ID is required.")
        else:
            status, body, elapsed = client.list_conversations(
                case_id.strip(), skip=int(skip), limit=int(limit)
            )
            show_response(status, body, elapsed)

# -- Get ----------------------------------------------------------------------
with tab_get:
    st.subheader("Get Conversation")
    conv_id = st.text_input(
        "Conversation ID",
        value=st.session_state.get("last_conversation_id", ""),
        key="get_conv_id",
    )
    if st.button("Get Conversation"):
        if not conv_id.strip():
            st.error("Conversation ID is required.")
        else:
            status, body, elapsed = client.get_conversation(conv_id.strip())
            show_response(status, body, elapsed)

            # Display turns nicely if present
            if status == 200 and "turns" in body:
                st.divider()
                st.subheader("Conversation Turns")
                for i, turn in enumerate(body["turns"]):
                    with st.expander(f"Turn {turn.get('turn_number', i + 1)}"):
                        st.markdown(f"**Query:** {turn.get('query', '')}")
                        st.markdown(f"**Response:** {turn.get('response', '')}")
                        st.markdown(f"**Intent:** {turn.get('intent', 'N/A')}")
                        st.markdown(f"**Agents:** {', '.join(turn.get('agents_used', []))}")

# -- Delete -------------------------------------------------------------------
with tab_delete:
    st.subheader("Delete Conversation")
    conv_id_del = st.text_input(
        "Conversation ID",
        value=st.session_state.get("last_conversation_id", ""),
        key="del_conv_id",
    )
    if st.button("Delete Conversation", type="primary"):
        if not conv_id_del.strip():
            st.error("Conversation ID is required.")
        else:
            status, body, elapsed = client.delete_conversation(conv_id_del.strip())
            show_response(status, body, elapsed)
