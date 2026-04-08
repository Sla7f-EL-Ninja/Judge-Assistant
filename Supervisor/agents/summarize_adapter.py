"""
summarize_adapter.py

Adapter for the Summarization pipeline (Summerize/graph.py).

Wraps ``create_pipeline(llm)`` and returns an AgentResult with the
rendered Arabic case brief.

Performance fix
---------------
The original implementation added the Summerize directory to sys.path and
re-imported ``get_llm`` / ``create_pipeline`` on every call.  The path
guard (``if summerize_dir not in sys.path``) prevented duplicate path
entries, but the import statements still ran every time.  The imported
callables are now cached at the class level so the module-loading overhead
happens only once per process.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MongoDB helper (unchanged from original)
# ---------------------------------------------------------------------------

def _fetch_documents_from_mongo(case_id: str) -> List[Dict[str, str]]:
    """Fetch raw document texts from MongoDB for the given case_id.

    Returns a list of {raw_text, doc_id} dicts, or an empty list on failure.
    """
    try:
        from pymongo import MongoClient
        from config.supervisor import MONGO_URI, MONGO_DB, MONGO_COLLECTION

        client = MongoClient(MONGO_URI)
        collection = client[MONGO_DB][MONGO_COLLECTION]

        cursor = collection.find({"case_id": case_id})
        documents = []
        for doc in cursor:
            raw_text = doc.get("text", "")
            doc_id = str(
                doc.get("title") or doc.get("source_file") or doc.get("_id", "unknown")
            )
            if raw_text:
                documents.append({"raw_text": raw_text, "doc_id": doc_id})

        client.close()
        logger.info(
            "Fetched %d document(s) from MongoDB for case_id='%s'",
            len(documents),
            case_id,
        )
        return documents

    except Exception as e:
        logger.error("MongoDB fetch failed for case_id='%s': %s", case_id, e)
        return []


# ---------------------------------------------------------------------------
# One-time import loader
# ---------------------------------------------------------------------------

def _load_summarize_callables():
    """Import and return (get_llm, create_pipeline) from the Summerize package.

    Called once; result cached on the class.
    """
    summerize_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "Summerize"
    ))
    if summerize_dir not in sys.path:
        sys.path.insert(0, summerize_dir)

    from dotenv import load_dotenv
    load_dotenv()

    from config import get_llm          # noqa: E402  resolved from summerize_dir
    from graph import create_pipeline   # noqa: E402  resolved from summerize_dir

    return get_llm, create_pipeline


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SummarizeAdapter(AgentAdapter):
    """Thin wrapper around the Summarization LangGraph pipeline."""

    _get_llm = None
    _create_pipeline = None

    @classmethod
    def _get_callables(cls):
        if cls._get_llm is None:
            logger.info("Loading Summarize pipeline (first call)...")
            cls._get_llm, cls._create_pipeline = _load_summarize_callables()
            logger.info("Summarize pipeline loaded and cached.")
        return cls._get_llm, cls._create_pipeline

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """Run the summarisation pipeline on the provided documents.

        Document resolution priority:
        1. ``context["documents"]``            -- pre-built {raw_text, doc_id} dicts.
        2. ``context["agent_results"]["ocr"]`` -- OCR ran earlier this turn.
        3. ``context["uploaded_files"]``       -- file paths; read from disk.
        4. MongoDB                             -- fetch by context["case_id"].
        """
        try:
            get_llm, create_pipeline = self._get_callables()

            # --- 1. Explicit documents list ---
            documents = context.get("documents")

            # --- 2. OCR output from an earlier agent in the same turn ---
            if not documents:
                ocr_result = (context.get("agent_results") or {}).get("ocr")
                if ocr_result and isinstance(ocr_result, dict):
                    raw_texts = ocr_result.get("raw_texts", [])
                    documents = [
                        {"raw_text": t, "doc_id": f"doc_{i}"}
                        for i, t in enumerate(raw_texts)
                    ]

            # --- 3. uploaded_files paths -- read from disk ---
            if not documents:
                uploaded_files = context.get("uploaded_files") or []
                if uploaded_files:
                    documents = []
                    for file_path in uploaded_files:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                raw_text = f.read()
                            documents.append({
                                "raw_text": raw_text,
                                "doc_id": os.path.basename(file_path),
                            })
                            logger.info(
                                "Loaded file for summarisation: %s", file_path
                            )
                        except Exception as read_err:
                            logger.warning(
                                "Could not read uploaded file '%s': %s",
                                file_path,
                                read_err,
                            )

            # --- 4. MongoDB fallback ---
            if not documents:
                case_id = context.get("case_id", "")
                if case_id:
                    logger.info(
                        "No in-memory documents; fetching from MongoDB for case_id='%s'",
                        case_id,
                    )
                    documents = _fetch_documents_from_mongo(case_id)
                else:
                    logger.warning(
                        "No case_id in context; cannot fetch from MongoDB."
                    )

            if not documents:
                return AgentResult(
                    response="",
                    error=(
                        "No documents found for summarisation. "
                        "Please upload files or ingest them first using --ingest."
                    ),
                )

            logger.info("Summarising %d document(s).", len(documents))
            llm = get_llm("high")
            pipeline = create_pipeline(llm)
            result = pipeline.invoke({"documents": documents})

            rendered_brief = result.get("rendered_brief", "")
            all_sources = result.get("all_sources", [])

            return AgentResult(
                response=rendered_brief,
                sources=all_sources,
                raw_output={
                    "rendered_brief": rendered_brief,
                    "case_brief": result.get("case_brief", {}),
                },
            )

        except Exception as exc:
            error_msg = f"Summarize adapter error: {exc}"
            logger.exception(error_msg)
            # Reset so the next call retries the import
            SummarizeAdapter._get_llm = None
            SummarizeAdapter._create_pipeline = None
            return AgentResult(response="", error=error_msg)