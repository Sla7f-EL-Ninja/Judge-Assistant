"""File upload page."""

import streamlit as st
from utils.display import show_response

st.title("File Upload")
st.markdown("`POST /api/v1/files/upload`")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

uploaded = st.file_uploader(
    "Choose a file",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    help="Allowed: PDF, PNG, JPEG, TIFF, BMP, WebP. Max 20 MB.",
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
