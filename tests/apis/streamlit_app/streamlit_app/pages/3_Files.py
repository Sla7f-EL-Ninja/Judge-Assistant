"""File upload and delete page."""

import base64

import streamlit as st
import streamlit.components.v1 as components
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
                b64 = base64.b64encode(data).decode("utf-8")
                pdf_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#525659; font-family:sans-serif; }}
  #toolbar {{
    display:flex; align-items:center; gap:8px;
    padding:6px 12px; background:#404040; color:#fff; font-size:13px;
  }}
  #toolbar button {{
    background:#666; border:none; color:#fff; padding:3px 10px;
    border-radius:3px; cursor:pointer; font-size:13px;
  }}
  #toolbar button:hover {{ background:#888; }}
  #canvas-container {{
    overflow-y:auto; height:calc(100vh - 38px);
    display:flex; flex-direction:column; align-items:center; gap:8px; padding:12px 0;
  }}
  canvas {{ box-shadow:0 2px 8px rgba(0,0,0,0.5); }}
</style></head>
<body>
<div id="toolbar">
  <button onclick="changePage(-1)">&#9664;</button>
  <span id="page-info">Page 1 of ?</span>
  <button onclick="changePage(1)">&#9654;</button>
  <span style="margin-left:12px;">Zoom:</span>
  <button onclick="changeZoom(-0.25)">&#8722;</button>
  <span id="zoom-info">150%</span>
  <button onclick="changeZoom(0.25)">+</button>
</div>
<div id="canvas-container"></div>
<script>
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  const raw = atob("{b64}");
  const arr = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
  let pdfDoc = null, currentPage = 1, scale = 1.5;
  const container = document.getElementById("canvas-container");
  const pageInfo  = document.getElementById("page-info");
  const zoomInfo  = document.getElementById("zoom-info");
  pdfjsLib.getDocument({{ data: arr }}).promise.then(pdf => {{
    pdfDoc = pdf;
    renderPage(currentPage);
  }});
  function renderPage(num) {{
    container.innerHTML = "";
    pdfDoc.getPage(num).then(page => {{
      const vp = page.getViewport({{ scale }});
      const canvas = document.createElement("canvas");
      canvas.width = vp.width; canvas.height = vp.height;
      container.appendChild(canvas);
      page.render({{ canvasContext: canvas.getContext("2d"), viewport: vp }});
      pageInfo.textContent = "Page " + currentPage + " of " + pdfDoc.numPages;
      zoomInfo.textContent = Math.round(scale * 100) + "%";
    }});
  }}
  function changePage(d) {{
    const n = currentPage + d;
    if (n < 1 || n > pdfDoc.numPages) return;
    currentPage = n; renderPage(currentPage);
  }}
  function changeZoom(d) {{
    scale = Math.max(0.5, Math.min(3.0, scale + d));
    renderPage(currentPage);
  }}
</script></body></html>"""
                components.html(pdf_html, height=820, scrolling=False)
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