"""
Judge Assistant API -- Streamlit Testing App

Main entry point. Configure the API connection in the sidebar,
then navigate to endpoint pages to test them.
"""

import streamlit as st
from utils.api_client import JudgeAssistantClient, _generate_jwt

st.set_page_config(
    page_title="Judge Assistant API Tester",
    page_icon="⚖️",
    layout="wide",
)

# -- Sidebar: connection settings ---------------------------------------------

st.sidebar.title("API Connection")

import os

base_url = st.sidebar.text_input(
    "Base URL", value=st.session_state.get("base_url", os.environ.get("API_BASE_URL", "http://localhost:8000"))
)
st.session_state["base_url"] = base_url

jwt_secret = st.sidebar.text_input(
    "JWT Secret", value=st.session_state.get("jwt_secret", "change-me-in-production"), type="password"
)
st.session_state["jwt_secret"] = jwt_secret

user_id = st.sidebar.text_input(
    "User ID", value=st.session_state.get("user_id", "test_user_001")
)
st.session_state["user_id"] = user_id

auto_gen = st.sidebar.checkbox("Auto-generate JWT", value=True)

if auto_gen:
    token = _generate_jwt(user_id, jwt_secret)
    st.session_state["jwt_token"] = token
else:
    token = st.sidebar.text_area(
        "JWT Token", value=st.session_state.get("jwt_token", ""), height=80
    )
    st.session_state["jwt_token"] = token

if token:
    st.session_state["client"] = JudgeAssistantClient(base_url, token)

st.sidebar.divider()
st.sidebar.subheader("Session State")
for key in ("last_case_id", "last_file_id", "last_conversation_id", "last_doc_id", "last_report_id"):
    val = st.session_state.get(key, "N/A")
    st.sidebar.text(f"{key}: {val}")

# -- Main page ----------------------------------------------------------------

st.title("Judge Assistant API Tester")

st.markdown("""
Use the sidebar to configure your API connection, then navigate to the endpoint
pages using the sidebar menu.

### Endpoints

| Page | Method | Endpoint | Description |
|------|--------|----------|-------------|
| Health | GET | `/api/v1/health` | Service health check |
| Cases | POST | `/api/v1/cases` | Create case |
| Cases | GET | `/api/v1/cases` | List cases |
| Cases | GET | `/api/v1/cases/{id}` | Get case |
| Cases | PATCH | `/api/v1/cases/{id}` | Update case |
| Cases | DELETE | `/api/v1/cases/{id}` | Delete case |
| Files | POST | `/api/v1/files/upload` | Upload file |
| Files | GET | `/api/v1/files/{id}` | View or download raw file |
| Files | DELETE | `/api/v1/files/{id}` | Delete file |
| Documents | POST | `/api/v1/cases/{id}/documents` | Ingest documents (legacy file_ids or grouped) |
| Documents | GET | `/api/v1/cases/{id}/documents` | List documents |
| Documents | GET | `/api/v1/cases/{id}/documents/{doc_id}` | Get document detail |
| Documents | DELETE | `/api/v1/cases/{id}/documents/{doc_id}` | Delete document |
| Documents | GET | `/api/v1/cases/{id}/documents/{doc_id}/ocr` | Get OCR text |
| Documents | PATCH | `/api/v1/cases/{id}/documents/{doc_id}/ocr` | Correct OCR text |
| Documents | POST | `/api/v1/cases/{id}/documents/ocr/bulk` | Bulk correct OCR text |
| Query | POST | `/api/v1/query` | Supervisor query (SSE) |
| Conversations | GET | `/api/v1/cases/{id}/conversations` | List conversations |
| Conversations | GET | `/api/v1/conversations/{id}` | Get conversation |
| Conversations | DELETE | `/api/v1/conversations/{id}` | Delete conversation |
| Summaries | GET | `/api/v1/cases/{id}/summary` | Get summary |
| Summaries | POST | `/api/v1/cases/{id}/summary/generate` | Generate summary |
| Summaries | GET | `/api/v1/cases/{id}/case-brief` | Get case brief |
| Case Reasoning | GET | `/api/v1/cases/{id}/case-reasoning` | Get case reasoning |
| Reports | POST | `/api/v1/cases/{id}/reports/generate` | Generate report |
| Reports | GET | `/api/v1/cases/{id}/reports/{report_id}` | Poll report status |
| Reports | GET | `/api/v1/cases/{id}/reports` | List reports |
| Legal Search | GET | `/api/v1/legal/search` | Search legal corpus |

### Getting Started

1. Make sure the API server is running at the configured base URL
2. Set the JWT secret to match the server's `JWT_SECRET` env var
3. Click on a page in the sidebar to start testing
""")
