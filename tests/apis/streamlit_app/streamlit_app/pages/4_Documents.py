"""Document endpoints page — 7 tabs."""

import streamlit as st
from utils.display import show_response

st.title("Documents")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

tab_ingest, tab_list, tab_detail, tab_ocr_get, tab_ocr_correct, tab_ocr_bulk, tab_delete = st.tabs(
    ["Ingest", "List", "Detail", "OCR — Get", "OCR — Correct", "OCR — Bulk Correct", "Delete"]
)

# -- Ingest -------------------------------------------------------------------
with tab_ingest:
    st.subheader("Ingest Documents")
    st.markdown("`POST /api/v1/cases/{case_id}/documents`")

    case_id = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="ingest_case_id")

    ingest_mode = st.radio(
        "Ingest mode",
        ["Legacy — one document per file", "Groups — merge files into one document"],
        key="ingest_mode",
        horizontal=True,
    )

    if ingest_mode.startswith("Legacy"):
        file_ids_str = st.text_area(
            "File IDs (one per line)",
            value=st.session_state.get("last_file_id", ""),
            height=100,
            key="ingest_file_ids_str",
        )
    else:
        st.caption("Each line = one document. Example: `id1, id2` → one multi-file doc.")
        file_ids_str = st.text_area(
            "Groups (one group per line, comma-separated file IDs)",
            height=120,
            key="ingest_groups_str",
        )

    if st.button("Ingest Documents"):
        if not case_id.strip():
            st.error("Case ID is required.")
        elif ingest_mode.startswith("Legacy"):
            file_ids = [f.strip() for f in file_ids_str.strip().splitlines() if f.strip()]
            if not file_ids:
                st.error("At least one file ID is required.")
            else:
                st.info(f"Ingesting {len(file_ids)} file(s) into case {case_id}...")
                with st.spinner("Processing (10–30s per file)..."):
                    status, body, elapsed = client.ingest_documents(case_id.strip(), file_ids=file_ids)
                show_response(status, body, elapsed)
                if status == 201:
                    ingested = body.get("ingested", [])
                    errors = body.get("errors", [])
                    if ingested:
                        st.success(f"Ingested: {len(ingested)} document(s)")
                    if errors:
                        st.error(f"Errors: {len(errors)} group(s) failed")
                        for err in errors:
                            ids = err.get("file_ids") or [err.get("file_id", "?")]
                            st.warning(f"Files {ids}: {err.get('error', '?')}")
        else:
            raw_lines = [l for l in file_ids_str.strip().splitlines() if l.strip()]
            if not raw_lines:
                st.error("At least one group is required.")
            else:
                groups = []
                parse_error = False
                for i, line in enumerate(raw_lines, 1):
                    ids = [tok.strip() for tok in line.split(",") if tok.strip()]
                    if not ids:
                        st.error(f"Line {i} has no valid file IDs after parsing.")
                        parse_error = True
                    else:
                        groups.append(ids)
                if not parse_error:
                    st.info(f"Submitting {len(groups)} group(s) into case {case_id}...")
                    with st.spinner("Processing (10–30s per group)..."):
                        status, body, elapsed = client.ingest_documents(case_id.strip(), groups=groups)
                    show_response(status, body, elapsed)
                    if status == 201:
                        ingested = body.get("ingested", [])
                        errors = body.get("errors", [])
                        if ingested:
                            st.success(f"Ingested: {len(ingested)} document(s)")
                        if errors:
                            st.error(f"Errors: {len(errors)} group(s) failed")
                            for err in errors:
                                ids = err.get("file_ids") or [err.get("file_id", "?")]
                                st.warning(f"Files {ids}: {err.get('error', '?')}")

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
                file_ids = body.get("file_ids", [])
                if len(file_ids) > 1:
                    st.multiselect("Linked file IDs", options=file_ids, default=file_ids, disabled=True, key="det_file_ids_display")

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

                file_ids = body.get("file_ids", [])
                if len(file_ids) > 1:
                    st.multiselect("Linked file IDs", options=file_ids, default=file_ids, disabled=True, key="ocr_get_file_ids_display")

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

# -- OCR Bulk Correct ---------------------------------------------------------
with tab_ocr_bulk:
    st.subheader("Bulk Correct OCR Text")
    st.markdown("`POST /api/v1/cases/{case_id}/documents/ocr/bulk`")

    bulk_case_id = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="bulk_ocr_case_id")
    bulk_corrected_by = st.text_input("Corrected By (optional — applies to all)", key="bulk_ocr_corrected_by")

    col_ids, col_texts = st.columns(2)
    with col_ids:
        doc_ids_str = st.text_area("Document IDs (one per line)", height=200, key="bulk_ocr_doc_ids")
    with col_texts:
        texts_str = st.text_area("Corrected Texts (one per line, matching order)", height=200, key="bulk_ocr_texts")

    if st.button("Submit Bulk Correction", type="primary"):
        if not bulk_case_id.strip():
            st.error("Case ID is required.")
        else:
            doc_ids = [l.strip() for l in doc_ids_str.strip().splitlines() if l.strip()]
            texts = [l.strip() for l in texts_str.strip().splitlines() if l.strip()]
            if not doc_ids:
                st.error("At least one Document ID is required.")
            elif len(doc_ids) != len(texts):
                st.error(f"Line count mismatch: {len(doc_ids)} doc IDs vs {len(texts)} texts. Must be equal.")
            else:
                corrections = [{"doc_id": d, "text": t} for d, t in zip(doc_ids, texts)]
                with st.spinner(f"Correcting {len(corrections)} document(s)..."):
                    status, body, elapsed = client.bulk_correct_document_ocr(
                        bulk_case_id.strip(), corrections, bulk_corrected_by.strip() or None
                    )
                show_response(status, body, elapsed)
                if status == 207:
                    m1, m2 = st.columns(2)
                    m1.metric("Succeeded", body.get("succeeded", 0))
                    m2.metric("Failed", body.get("failed", 0))

                    succeeded_ids = [r["doc_id"] for r in body.get("results", []) if r.get("status") == "success"]
                    failed_items = [r for r in body.get("results", []) if r.get("status") == "failed"]

                    if succeeded_ids and not failed_items:
                        st.success("All corrections applied.")
                    if succeeded_ids and failed_items:
                        st.success(f"Succeeded: {', '.join(succeeded_ids)}")
                    if failed_items:
                        with st.expander(f"Failures ({len(failed_items)})"):
                            for item in failed_items:
                                err = item.get("error") or {}
                                st.warning(
                                    f"`{item['doc_id']}` — {err.get('code', '?')}: {err.get('message', '?')}"
                                )
                else:
                    st.error(f"Unexpected status {status}. Check raw response above.")

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
