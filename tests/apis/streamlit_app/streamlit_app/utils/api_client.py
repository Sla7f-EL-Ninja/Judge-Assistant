"""
api_client.py

HTTP client wrapper for the Judge Assistant API.
All methods return a ``(status_code, body_dict, elapsed_ms)`` tuple.
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import requests
from jose import jwt


def _generate_jwt(user_id: str, secret: str, algorithm: str = "HS256") -> str:
    """Create a JWT token with the given user_id claim."""
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=4),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


class JudgeAssistantClient:
    """HTTP client for the Judge Assistant API."""

    def __init__(self, base_url: str, jwt_token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        })

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        json: Any = None,
        files: Any = None,
        params: Any = None,
        timeout: int = 700,
    ) -> tuple[int, dict, float]:
        """Execute a request and return (status, body, elapsed_ms)."""
        headers = dict(self.session.headers)
        if files:
            headers["Content-Type"] = None

        start = time.time()
        try:
            r = self.session.request(
                method,
                self._url(path),
                json=json,
                files=files,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            elapsed = (time.time() - start) * 1000
            try:
                body = r.json()
            except Exception:
                body = {"_raw": r.text}
            return r.status_code, body, elapsed
        except requests.RequestException as exc:
            elapsed = (time.time() - start) * 1000
            return 0, {"_error": str(exc)}, elapsed

    def _request_raw(
        self,
        method: str,
        path: str,
        params: Any = None,
        timeout: int = 60,
    ) -> tuple[int, bytes, str, float]:
        """Execute a request and return (status, raw_bytes, content_type, elapsed_ms)."""
        start = time.time()
        try:
            r = self.session.request(
                method,
                self._url(path),
                params=params,
                timeout=timeout,
            )
            elapsed = (time.time() - start) * 1000
            content_type = r.headers.get("Content-Type", "")
            return r.status_code, r.content, content_type, elapsed
        except requests.RequestException as exc:
            elapsed = (time.time() - start) * 1000
            return 0, str(exc).encode(), "", elapsed

    # -- Health ---------------------------------------------------------------

    def health(self) -> tuple[int, dict, float]:
        return self._request("GET", "/api/v1/health")

    # -- Cases ----------------------------------------------------------------

    def create_case(
        self, title: str, description: str = "", metadata: Optional[dict] = None
    ) -> tuple[int, dict, float]:
        payload = {"title": title, "description": description}
        if metadata:
            payload["metadata"] = metadata
        return self._request("POST", "/api/v1/cases", json=payload)

    def list_cases(self, skip: int = 0, limit: int = 20) -> tuple[int, dict, float]:
        return self._request("GET", "/api/v1/cases", params={"skip": skip, "limit": limit})

    def get_case(self, case_id: str) -> tuple[int, dict, float]:
        return self._request("GET", f"/api/v1/cases/{case_id}")

    def update_case(self, case_id: str, updates: dict) -> tuple[int, dict, float]:
        return self._request("PATCH", f"/api/v1/cases/{case_id}", json=updates)

    def delete_case(self, case_id: str) -> tuple[int, dict, float]:
        return self._request("DELETE", f"/api/v1/cases/{case_id}")

    # -- Files ----------------------------------------------------------------

    def upload_file(self, filename: str, content: bytes, mime_type: str) -> tuple[int, dict, float]:
        files = {"file": (filename, content, mime_type)}
        return self._request("POST", "/api/v1/files/upload", files=files)

    def get_file(self, file_id: str, download: bool = False) -> tuple[int, bytes, str, float]:
        params = {"download": 1} if download else None
        return self._request_raw("GET", f"/api/v1/files/{file_id}", params=params)

    def delete_file(self, file_id: str) -> tuple[int, dict, float]:
        return self._request("DELETE", f"/api/v1/files/{file_id}")

    # -- Documents ------------------------------------------------------------

    def ingest_documents(
        self,
        case_id: str,
        file_ids: list[str] | None = None,
        groups: list[list[str]] | None = None,
    ) -> tuple[int, dict, float]:
        if groups is not None:
            payload = {"groups": [{"file_ids": g} for g in groups]}
        else:
            payload = {"file_ids": file_ids or []}
        return self._request(
            "POST",
            f"/api/v1/cases/{case_id}/documents",
            json=payload,
            timeout=700,
        )

    def list_documents(self, case_id: str) -> tuple[int, dict, float]:
        return self._request("GET", f"/api/v1/cases/{case_id}/documents")

    def get_document(self, case_id: str, doc_id: str) -> tuple[int, dict, float]:
        return self._request("GET", f"/api/v1/cases/{case_id}/documents/{doc_id}")

    def get_document_ocr(self, case_id: str, doc_id: str) -> tuple[int, dict, float]:
        return self._request("GET", f"/api/v1/cases/{case_id}/documents/{doc_id}/ocr")

    def correct_document_ocr(
        self, case_id: str, doc_id: str, text: str, corrected_by: str = ""
    ) -> tuple[int, dict, float]:
        payload: dict = {"text": text}
        if corrected_by:
            payload["corrected_by"] = corrected_by
        return self._request("PATCH", f"/api/v1/cases/{case_id}/documents/{doc_id}/ocr", json=payload)

    def bulk_correct_document_ocr(
        self,
        case_id: str,
        corrections: list[dict],
        corrected_by: Optional[str] = None,
    ) -> tuple[int, dict, float]:
        payload: dict = {"corrections": corrections}
        if corrected_by:
            payload["corrected_by"] = corrected_by
        return self._request(
            "POST",
            f"/api/v1/cases/{case_id}/documents/ocr/bulk",
            json=payload,
        )

    def delete_document(self, case_id: str, doc_id: str) -> tuple[int, dict, float]:
        return self._request("DELETE", f"/api/v1/cases/{case_id}/documents/{doc_id}")

    # -- Query ----------------------------------------------------------------

    def query(
        self,
        query_text: str,
        case_id: str,
        conversation_id: Optional[str] = None,
    ) -> tuple[int, list[dict], float]:
        """Send a query and parse SSE events. Returns (status, events_list, elapsed_ms)."""
        payload: dict[str, Any] = {"query": query_text, "case_id": case_id}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        start = time.time()
        try:
            r = self.session.post(
                self._url("/api/v1/query"),
                json=payload,
                stream=True,
                timeout=500,
            )
            events = []
            current: dict[str, Any] = {}
            for line in r.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                if line.startswith("event:"):
                    current["event"] = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    raw = line.removeprefix("data:").strip()
                    try:
                        import json
                        current["data"] = json.loads(raw)
                    except Exception:
                        current["data"] = raw
                elif line == "" and current:
                    events.append(current)
                    current = {}
            if current:
                events.append(current)
            elapsed = (time.time() - start) * 1000
            return r.status_code, events, elapsed
        except requests.RequestException as exc:
            elapsed = (time.time() - start) * 1000
            return 0, [{"event": "error", "data": {"_error": str(exc)}}], elapsed

    # -- Conversations --------------------------------------------------------

    def list_conversations(self, case_id: str, skip: int = 0, limit: int = 20) -> tuple[int, dict, float]:
        return self._request(
            "GET",
            f"/api/v1/cases/{case_id}/conversations",
            params={"skip": skip, "limit": limit},
        )

    def get_conversation(self, conversation_id: str) -> tuple[int, dict, float]:
        return self._request("GET", f"/api/v1/conversations/{conversation_id}")

    def delete_conversation(self, conversation_id: str) -> tuple[int, dict, float]:
        return self._request("DELETE", f"/api/v1/conversations/{conversation_id}")

    # -- Summaries ------------------------------------------------------------

    def get_summary(self, case_id: str) -> tuple[int, dict, float]:
        return self._request("GET", f"/api/v1/cases/{case_id}/summary")

    def generate_summary(self, case_id: str) -> tuple[int, dict, float]:
        return self._request("POST", f"/api/v1/cases/{case_id}/summary/generate", timeout=800)

    def get_case_brief(self, case_id: str) -> tuple[int, dict, float]:
        return self._request("GET", f"/api/v1/cases/{case_id}/case-brief")

    # -- Case Reasoning -------------------------------------------------------

    def get_case_reasoning(self, case_id: str) -> tuple[int, dict, float]:
        return self._request("GET", f"/api/v1/cases/{case_id}/case-reasoning")

    # -- Reports --------------------------------------------------------------

    def generate_report(self, case_id: str) -> tuple[int, dict, float]:
        return self._request("POST", f"/api/v1/cases/{case_id}/reports/generate", json={})

    def get_report(self, case_id: str, report_id: str) -> tuple[int, dict, float]:
        return self._request("GET", f"/api/v1/cases/{case_id}/reports/{report_id}")

    def list_reports(self, case_id: str, skip: int = 0, limit: int = 20) -> tuple[int, dict, float]:
        return self._request(
            "GET", f"/api/v1/cases/{case_id}/reports", params={"skip": skip, "limit": limit}
        )

    # -- Legal Search ---------------------------------------------------------

    def legal_search(self, q: str, corpus: str = "civil") -> tuple[int, dict, float]:
        """Free-text RAG search against a legal corpus."""
        return self._request("GET", "/api/v1/legal/search", params={"q": q, "corpus": corpus})

    def legal_article_lookup(
        self,
        corpus: str = "civil",
        article_no: int | None = None,
        chapter: str | None = None,
        section: str | None = None,
    ) -> tuple[int, dict, float]:
        """Fetch articles directly by number, chapter, or section (no LLM)."""
        params: dict = {"corpus": corpus}
        if article_no is not None:
            params["article_no"] = article_no
        if chapter is not None:
            params["chapter"] = chapter
        if section is not None:
            params["section"] = section
        return self._request("GET", "/api/v1/legal/article", params=params)

    def legal_corpus_tree(self, corpus: str = "civil") -> tuple[int, dict, float]:
        """Fetch the full legal corpus as a book→part→chapter→section→article tree."""
        return self._request("GET", "/api/v1/legal/corpus/tree", params={"corpus": corpus})