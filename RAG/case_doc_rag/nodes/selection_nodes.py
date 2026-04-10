"""case_doc_rag.nodes.selection_nodes -- Document selection and finalization nodes.

Nodes: documentSelector, DocumentFinalizer
Utilities: get_available_doc_titles, fuzzy_match_doc_title
"""

import json
import logging
import threading
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from RAG.case_doc_rag.infrastructure import get_llm, get_mongo_collection
from RAG.case_doc_rag.models import DocSelection
from RAG.case_doc_rag.prompts import DOC_SELECTOR_SYSTEM_PROMPT_TEMPLATE
from RAG.case_doc_rag.state import AgentState

logger = logging.getLogger("case_doc_rag.selection_nodes")

# ---------------------------------------------------------------------------
# TTL cache for document titles
# ---------------------------------------------------------------------------

_titles_cache: Dict[str, List[str]] = {}
_titles_cache_ts: Dict[str, float] = {}
_titles_cache_lock = threading.Lock()
_CACHE_TTL = 60.0  # seconds


def get_available_doc_titles(case_id: str) -> List[str]:
    """Fetch document titles from MongoDB with a per-case_id TTL cache.

    Thread-safe. Uses double-check locking so the lock is NOT held during
    MongoDB I/O -- only for the brief cache read/write. This prevents a
    slow query for one case_id from blocking fetchers for other case_ids.
    """
    # Fast path: check cache under lock, return immediately on hit.
    with _titles_cache_lock:
        if (
            case_id in _titles_cache
            and time.time() - _titles_cache_ts.get(case_id, 0) < _CACHE_TTL
        ):
            logger.debug("Title cache hit for case_id=%s", case_id)
            return _titles_cache[case_id]

    # Slow path: perform MongoDB I/O outside the lock.
    try:
        collection = get_mongo_collection()
        query_filter = {"case_id": case_id} if case_id else {}
        docs = list(collection.find(query_filter, {"title": 1, "_id": 0}))
        titles = [
            doc["title"] for doc in docs
            if doc.get("title") and isinstance(doc["title"], str)
        ]
    except Exception:
        logger.exception(
            "MongoDB error fetching titles for case_id=%s", case_id
        )
        return []

    # Re-acquire lock and double-check before writing. Another thread may
    # have populated the cache while we were querying.
    with _titles_cache_lock:
        if (
            case_id in _titles_cache
            and time.time() - _titles_cache_ts.get(case_id, 0) < _CACHE_TTL
        ):
            # Another thread filled the cache while we queried; use its result.
            return _titles_cache[case_id]

        _titles_cache[case_id] = titles
        _titles_cache_ts[case_id] = time.time()

    logger.info(
        "Fetched %d doc titles for case_id=%s from MongoDB",
        len(titles), case_id,
    )
    return titles


def fuzzy_match_doc_title(
    candidate: str,
    available_titles: List[str],
    threshold: float = 0.5,
) -> Optional[str]:
    """Return the best-matching title if similarity meets the threshold."""
    if not candidate or not available_titles:
        return None
    best_match = None
    best_score = 0.0
    for title in available_titles:
        if not title:
            continue
        score = SequenceMatcher(None, candidate, title).ratio()
        if score > best_score:
            best_score = score
            best_match = title
    if best_score >= threshold:
        return best_match
    return None


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def documentSelector(state: AgentState) -> Dict[str, Any]:
    """Classify the query's document intent and select the target document.

    Fixes: Bug 9 (.content crash on query), Perf 2 (MongoDB caching).
    """
    request_id = state.get("request_id", "")
    query = state.get("query", "")

    if not query or not query.strip():
        return {
            "doc_selection_mode": "no_doc_specified",
            "selected_doc_id": None,
            "doc_titles": [],
        }

    case_id = state.get("case_id", "")
    available_titles = get_available_doc_titles(case_id)

    if available_titles:
        docs_list = "\n".join(f"- {title}" for title in available_titles)
    else:
        docs_list = "(No documents available in the case file)"

    system_prompt = DOC_SELECTOR_SYSTEM_PROMPT_TEMPLATE.format(
        available_docs=docs_list
    )

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}"),
        ])
        chain = prompt | get_llm("high").with_structured_output(DocSelection)
        result = chain.invoke({"query": query})
        mode = result.mode
        doc_id = result.doc_id
    except Exception:
        logger.exception("[%s] LLM error in documentSelector", request_id)
        return {
            "doc_selection_mode": "no_doc_specified",
            "selected_doc_id": None,
            "doc_titles": available_titles,
        }

    # Validate doc_id against available titles
    if doc_id is not None and available_titles:
        if doc_id not in available_titles:
            matched = fuzzy_match_doc_title(doc_id, available_titles)
            if matched:
                logger.info(
                    "[%s] documentSelector: fuzzy matched '%s' -> '%s'",
                    request_id, doc_id, matched,
                )
                doc_id = matched
            else:
                logger.warning(
                    "[%s] documentSelector: doc_id '%s' not found, falling back",
                    request_id, doc_id,
                )
                doc_id = None
                mode = "no_doc_specified"

    logger.info(
        "[%s] documentSelector: mode=%s doc_id=%s",
        request_id, mode, doc_id,
    )
    return {
        "doc_selection_mode": mode,
        "selected_doc_id": doc_id,
        "doc_titles": available_titles,
    }


def DocumentFinalizer(state: AgentState) -> Dict[str, Any]:
    """Retrieve and return a specific document directly from MongoDB.

    Fixes: Bug 5 (retrieved_docs set to raw MongoDB dict instead of Document).
    Goes directly to END -- must write sub_answers and final_answer.
    """
    request_id = state.get("request_id", "")
    doc_id = state.get("selected_doc_id")

    if doc_id is None:
        logger.warning("[%s] DocumentFinalizer: no doc_id specified", request_id)
        return {
            "final_answer": "لم يتم تحديد مستند للاسترجاع.",
            "error": "DocumentFinalizer reached with no doc_id",
        }

    try:
        collection = get_mongo_collection()
        doc = collection.find_one({"title": doc_id})
    except Exception:
        logger.exception(
            "[%s] DocumentFinalizer: MongoDB error for doc_id='%s'",
            request_id, doc_id,
        )
        return {
            "error": f"Failed to retrieve document '{doc_id}' from MongoDB",
            "final_answer": "حدث خطأ أثناء استرجاع المستند من قاعدة البيانات.",
        }

    if doc is None:
        logger.warning(
            "[%s] DocumentFinalizer: doc '%s' not found in MongoDB",
            request_id, doc_id,
        )
        return {
            "error": f"Document '{doc_id}' not found in MongoDB",
            "final_answer": f"لم يتم العثور على المستند '{doc_id}' في قاعدة البيانات.",
        }

    # Extract text content -- try field names in priority order
    extracted_text = ""
    for field_name in ("content", "text", "body"):
        val = doc.get(field_name)
        if val and isinstance(val, str) and val.strip():
            extracted_text = val
            break

    if not extracted_text:
        # Last resort: convert entire dict to JSON (excluding _id)
        doc_copy = {k: v for k, v in doc.items() if k != "_id"}
        extracted_text = json.dumps(doc_copy, ensure_ascii=False, default=str)

    source_file = doc.get("source_file", "unknown")
    chunk_index = doc.get("chunk_index", 0)

    answer_entry = {
        "question": state.get("query", ""),
        "answer": extracted_text,
        "sources": [f"{source_file}:chunk_{chunk_index}"],
        "found": True,
    }

    logger.info(
        "[%s] DocumentFinalizer: returning doc '%s' (%d chars)",
        request_id, doc_id, len(extracted_text),
    )
    return {
        "sub_answers": [answer_entry],
        "final_answer": extracted_text,
    }
