"""Document endpoints page — 6 tabs."""

import streamlit as st
from utils.display import show_response

st.title("Documents")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

tab_ingest, tab_list, tab_detail, tab_ocr_get, tab_ocr_correct, tab_delete = st.tabs(
    ["Ingest", "List", "Detail", "OCR — Get", "OCR — Correct", "Delete"]
)

# -- Ingest -------------------------------------------------------------------
with tab_ingest:
    st.subheader("Ingest Documents")
    st.markdown("`POST /api/v1/cases/{case_id}/documents`")

    case_id = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="ingest_case_id")
    file_ids_str = st.text_area(
        "File IDs (one per line)",
        value=st.session_state.get("last_file_id", ""),
        height=100,
    )

    if st.button("Ingest Documents"):
        file_ids = [f.strip() for f in file_ids_str.strip().splitlines() if f.strip()]
        if not case_id.strip():
            st.error("Case ID is required.")
        elif not file_ids:
            st.error("At least one file ID is required.")
        else:
            st.info(f"Ingesting {len(file_ids)} file(s) into case {case_id}...")
            with st.spinner("Processing (10–30s per file)..."):
                status, body, elapsed = client.ingest_documents(case_id.strip(), file_ids)
            show_response(status, body, elapsed)
            if status == 201:
                ingested = body.get("ingested", [])
                errors = body.get("errors", [])
                if ingested:
                    st.success(f"Ingested: {len(ingested)} file(s)")
                if errors:
                    st.error(f"Errors: {len(errors)} file(s) failed")
                    for err in errors:
                        st.warning(f"File {err.get('file_id', '?')}: {err.get('error', '?')}")

# -- List ---------------------------------------------------------------------
with tab_list:
    st.subheader("List Documents")
    st.markdown("`GET /api/v1/cases/{case_id}/documents`")

    case_id_list = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="list_doc_case_id")

    if st.button("List Documents"):
        if not case_id_list.strip():
            st.error("Case ID is required.")
        else:
            status, body, elapsed = client.list_documents(case_id_list.strip())
            show_response(status, body, elapsed)
            if status == 200 and body.get("documents"):
                first_id = body["documents"][0].get("id", "")
                if first_id:
                    st.session_state["last_doc_id"] = first_id
                    st.info(f"First doc ID stored: {first_id}")

# -- Detail -------------------------------------------------------------------
with tab_detail:
    st.subheader("Get Document Detail")
    st.markdown("`GET /api/v1/cases/{case_id}/documents/{doc_id}`")

    case_id_det = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="det_case_id")
    doc_id_det = st.text_input("Document ID", value=st.session_state.get("last_doc_id", ""), key="det_doc_id")

    if st.button("Get Document"):
        if not case_id_det.strip() or not doc_id_det.strip():
            st.error("Case ID and Document ID are required.")
        else:
            status, body, elapsed = client.get_document(case_id_det.strip(), doc_id_det.strip())
            show_response(status, body, elapsed)
            if status == 200:
                st.session_state["last_doc_id"] = doc_id_det.strip()

# -- OCR Get ------------------------------------------------------------------
with tab_ocr_get:
    st.subheader("Get OCR Text")
    st.markdown("`GET /api/v1/cases/{case_id}/documents/{doc_id}/ocr`")

    case_id_ocr = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="ocr_get_case_id")
    doc_id_ocr = st.text_input("Document ID", value=st.session_state.get("last_doc_id", ""), key="ocr_get_doc_id")

    if st.button("Get OCR Text"):
        if not case_id_ocr.strip() or not doc_id_ocr.strip():
            st.error("Case ID and Document ID are required.")
        else:
            status, body, elapsed = client.get_document_ocr(case_id_ocr.strip(), doc_id_ocr.strip())
            show_response(status, body, elapsed)

            if status == 200:
                if body.get("corrected"):
                    st.badge("corrected", color="green")
                else:
                    st.badge("original", color="gray")

                if body.get("classification"):
                    cl = body["classification"]
                    st.markdown(
                        f"**Type:** `{cl.get('final_type', 'N/A')}` | "
                        f"**Confidence:** {cl.get('confidence', 0):.2f}"
                    )

                if body.get("text"):
                    st.divider()
                    st.subheader("OCR Text")
                    st.text_area("Text", value=body["text"], height=300, disabled=True)

# -- OCR Correct --------------------------------------------------------------
with tab_ocr_correct:
    st.subheader("Correct OCR Text")
    st.markdown("`PATCH /api/v1/cases/{case_id}/documents/{doc_id}/ocr`")

    case_id_fix = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="ocr_fix_case_id")
    doc_id_fix = st.text_input("Document ID", value=st.session_state.get("last_doc_id", ""), key="ocr_fix_doc_id")

    # Pre-fetch existing text
    prefill = ""
    if st.button("Load Existing Text"):
        if case_id_fix.strip() and doc_id_fix.strip():
            s, b, _ = client.get_document_ocr(case_id_fix.strip(), doc_id_fix.strip())
            if s == 200:
                prefill = b.get("text", "")
                st.session_state["_ocr_prefill"] = prefill
                st.success("Text loaded — edit below and submit.")
        else:
            st.error("Case ID and Document ID are required.")

    corrected_text = st.text_area(
        "Corrected Text",
        value=st.session_state.get("_ocr_prefill", prefill),
        height=300,
    )
    corrected_by = st.text_input("Corrected By (optional)", key="ocr_corrected_by")

    if st.button("Submit Correction", type="primary"):
        if not case_id_fix.strip() or not doc_id_fix.strip():
            st.error("Case ID and Document ID are required.")
        elif not corrected_text.strip():
            st.error("Corrected text cannot be empty.")
        else:
            status, body, elapsed = client.correct_document_ocr(
                case_id_fix.strip(), doc_id_fix.strip(),
                corrected_text, corrected_by.strip()
            )
            show_response(status, body, elapsed)
            if status == 200 and body.get("corrected"):
                st.success("Document marked as corrected.")
                st.session_state.pop("_ocr_prefill", None)

# -- Delete -------------------------------------------------------------------
with tab_delete:
    st.subheader("Delete Document")
    st.markdown("`DELETE /api/v1/cases/{case_id}/documents/{doc_id}`")

    case_id_del = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="del_doc_case_id")
    doc_id_del = st.text_input("Document ID", value=st.session_state.get("last_doc_id", ""), key="del_doc_id")

    confirm = st.checkbox("I confirm I want to delete this document (irreversible)")

    if st.button("Delete Document", type="primary", disabled=not confirm):
        if not case_id_del.strip() or not doc_id_del.strip():
            st.error("Case ID and Document ID are required.")
        else:
            status, body, elapsed = client.delete_document(case_id_del.strip(), doc_id_del.strip())
            show_response(status, body, elapsed)
