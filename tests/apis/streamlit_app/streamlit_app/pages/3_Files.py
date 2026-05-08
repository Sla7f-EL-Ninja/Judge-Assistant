"""File upload and delete page."""

import streamlit as st
from utils.display import show_response

st.title("Files")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

tab_upload, tab_view, tab_delete = st.tabs(["Upload", "View / Download", "Delete"])

# -- Upload -------------------------------------------------------------------
with tab_upload:
    st.subheader("Upload File")
    st.markdown("`POST /api/v1/files/upload`")
    st.caption("Allowed: PDF, PNG, JPEG, TIFF, BMP, WebP, HEIC, HEIF, TXT, MD. Max 20 MB.")

    uploaded = st.file_uploader(
        "Choose a file",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp", "heic", "heif", "txt", "md"],
    )

    if uploaded and st.button("Upload"):
        content = uploaded.read()
        mime = uploaded.type or "application/octet-stream"
        filename = uploaded.name or "unnamed"

        st.info(f"Uploading **{filename}** ({len(content):,} bytes, {mime})")
        status, body, elapsed = client.upload_file(filename, content, mime)
        show_response(status, body, elapsed)

        if status == 201 and "file_id" in body:
            st.session_state["last_file_id"] = body["file_id"]
            st.success(f"File ID stored: {body['file_id']}")

# -- View / Download ----------------------------------------------------------
with tab_view:
    st.subheader("View / Download File")
    st.markdown("`GET /api/v1/files/{file_id}`")

    view_file_id = st.text_input(
        "File ID", value=st.session_state.get("last_file_id", ""), key="view_file_id"
    )
    force_download = st.checkbox("Force download instead of inline view", key="view_file_download")

    if st.button("Fetch File"):
        if not view_file_id.strip():
            st.error("File ID is required.")
        else:
            status, data, content_type, elapsed = client.get_file(view_file_id.strip(), download=force_download)
            st.caption(f"HTTP {status} · {content_type or '—'} · {elapsed:.0f} ms · {len(data):,} bytes")

            if status != 200:
                st.error(f"Request failed with status {status}.")
                try:
                    st.code(data.decode("utf-8", errors="replace"))
                except Exception:
                    st.code(repr(data[:500]))
            elif force_download or not content_type:
                st.download_button(
                    "Download file",
                    data=data,
                    file_name=f"file_{view_file_id.strip()}",
                    mime=content_type or "application/octet-stream",
                )
            elif content_type.startswith("image/"):
                st.image(data)
            elif "pdf" in content_type:
                st.info("PDF rendering not supported inline — use the download button.")
                st.download_button(
                    "Download PDF",
                    data=data,
                    file_name=f"file_{view_file_id.strip()}.pdf",
                    mime="application/pdf",
                )
            else:
                st.download_button(
                    "Download file",
                    data=data,
                    file_name=f"file_{view_file_id.strip()}",
                    mime=content_type or "application/octet-stream",
                )

# -- Delete -------------------------------------------------------------------
with tab_delete:
    st.subheader("Delete File")
    st.markdown("`DELETE /api/v1/files/{file_id}`")

    file_id = st.text_input(
        "File ID", value=st.session_state.get("last_file_id", ""), key="del_file_id"
    )

    if st.button("Delete File", type="primary"):
        if not file_id.strip():
            st.error("File ID is required.")
        else:
            status, body, elapsed = client.delete_file(file_id.strip())
            show_response(status, body, elapsed)
