"""Document ingestion page."""

import streamlit as st
from utils.display import show_response

st.title("Document Ingestion")
st.markdown("`POST /api/v1/cases/{case_id}/documents`")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

case_id = st.text_input(
    "Case ID", value=st.session_state.get("last_case_id", "")
)

file_ids_str = st.text_area(
    "File IDs (one per line)",
    value=st.session_state.get("last_file_id", ""),
    height=100,
    help="Enter file IDs to ingest, one per line.",
)

if st.button("Ingest Documents"):
    file_ids = [fid.strip() for fid in file_ids_str.strip().splitlines() if fid.strip()]

    if not case_id.strip():
        st.error("Case ID is required.")
    elif not file_ids:
        st.error("At least one file ID is required.")
    else:
        st.info(f"Ingesting {len(file_ids)} file(s) into case {case_id}...")

        with st.spinner("Processing (this may take 10-30s per file)..."):
            status, body, elapsed = client.ingest_documents(case_id, file_ids)

        show_response(status, body, elapsed)

        if status == 200:
            ingested = body.get("ingested", [])
            errors = body.get("errors", [])
            if ingested:
                st.success(f"Successfully ingested: {len(ingested)} file(s)")
            if errors:
                st.error(f"Errors: {len(errors)} file(s) failed")
                for err in errors:
                    st.warning(f"File {err.get('file_id', '?')}: {err.get('error', '?')}")
