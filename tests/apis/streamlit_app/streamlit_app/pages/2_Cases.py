"""Cases CRUD page."""

import json
import streamlit as st
from utils.display import show_response

st.title("Cases")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

tab_create, tab_list, tab_get, tab_update, tab_delete = st.tabs(
    ["Create", "List", "Get", "Update", "Delete"]
)

# -- Create -------------------------------------------------------------------
with tab_create:
    st.subheader("Create Case")
    title = st.text_input("Title", value="Test Case")
    description = st.text_area("Description", value="")
    metadata_str = st.text_area("Metadata (JSON)", value="{}", height=80)

    if st.button("Create Case"):
        try:
            metadata = json.loads(metadata_str) if metadata_str.strip() else {}
        except json.JSONDecodeError:
            st.error("Invalid JSON in metadata field")
            metadata = {}

        status, body, elapsed = client.create_case(title, description, metadata)
        show_response(status, body, elapsed)

        if status == 201 and "_id" in body:
            st.session_state["last_case_id"] = body["_id"]
            st.success(f"Case ID stored: {body['_id']}")

# -- List ---------------------------------------------------------------------
with tab_list:
    st.subheader("List Cases")
    col1, col2 = st.columns(2)
    skip = col1.number_input("Skip", min_value=0, value=0, step=1)
    limit = col2.number_input("Limit", min_value=1, max_value=100, value=20, step=1)

    if st.button("List Cases"):
        status, body, elapsed = client.list_cases(skip=int(skip), limit=int(limit))
        show_response(status, body, elapsed)

# -- Get ----------------------------------------------------------------------
with tab_get:
    st.subheader("Get Case")
    case_id = st.text_input(
        "Case ID", value=st.session_state.get("last_case_id", ""), key="get_case_id"
    )
    if st.button("Get Case"):
        status, body, elapsed = client.get_case(case_id)
        show_response(status, body, elapsed)

# -- Update -------------------------------------------------------------------
with tab_update:
    st.subheader("Update Case")
    case_id_upd = st.text_input(
        "Case ID", value=st.session_state.get("last_case_id", ""), key="upd_case_id"
    )
    new_title = st.text_input("New Title (leave empty to skip)", key="upd_title")
    new_status = st.selectbox("New Status", ["(no change)", "active", "archived", "closed"])
    new_desc = st.text_input("New Description (leave empty to skip)", key="upd_desc")

    if st.button("Update Case"):
        updates = {}
        if new_title.strip():
            updates["title"] = new_title
        if new_status != "(no change)":
            updates["status"] = new_status
        if new_desc.strip():
            updates["description"] = new_desc

        if not updates:
            st.warning("No fields to update.")
        else:
            status, body, elapsed = client.update_case(case_id_upd, updates)
            show_response(status, body, elapsed)

# -- Delete -------------------------------------------------------------------
with tab_delete:
    st.subheader("Delete Case")
    case_id_del = st.text_input(
        "Case ID", value=st.session_state.get("last_case_id", ""), key="del_case_id"
    )
    if st.button("Delete Case", type="primary"):
        status, body, elapsed = client.delete_case(case_id_del)
        show_response(status, body, elapsed)
