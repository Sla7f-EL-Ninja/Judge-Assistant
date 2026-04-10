"""case_doc_rag.nodes.selection_nodes -- Document selection and finalization nodes.

Public nodes
------------
documentSelector        -- Main-graph title-fetch step (LLM removed, Phase 2).
branchDocSelector       -- Per-branch doc classification node (Phase 3).
BranchDocumentFinalizer -- Per-branch short-circuit finalizer (Phase 4).
DocumentFinalizer       -- DEPRECATED: kept for reference / test compatibility.
                           No longer wired into any graph. Will be removed in a
                           future cleanup pass once all tests are updated.

Utilities
---------
get_available_doc_titles(case_id) -- MongoDB fetch with TTL cache.
fuzzy_match_doc_title(candidate, available_titles, threshold) -- SequenceMatcher helper.
"""

import json
import logging
import threading
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from RAG.case_doc_rag.infrastructure import get_llm, get_mongo_collection
from RAG.case_doc_rag.models import DocSelection
from RAG.case_doc_rag.prompts import DOC_SELECTOR_SYSTEM_PROMPT_TEMPLATE
from RAG.case_doc_rag.state import AgentState, SubQuestionState

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
# Main-graph node (Phase 2 — title fetch only, LLM removed)
# ---------------------------------------------------------------------------


def documentSelector(state: AgentState) -> Dict[str, Any]:
    """Fetch available document titles for the case and write them to AgentState.

    Phase 2 refactor: the LLM classification call has been removed entirely.
    This node is now a lightweight MongoDB prefetch step. Its only job is to
    populate AgentState.doc_titles before fan-out so that every branch's
    branchDocSelector can classify without hitting MongoDB again.

    The per-sub-question doc-selection classification (mode + doc_id) is now
    handled by branchDocSelector inside the branch sub-graph.

    Returns
    -------
    {"doc_titles": List[str]}
        Only doc_titles is written. doc_selection_mode and selected_doc_id are
        intentionally NOT written here -- they are branch-local after the refactor.
    """
    request_id = state.get("request_id", "")
    case_id = state.get("case_id", "")

    titles = get_available_doc_titles(case_id)

    logger.info(
        "[%s] documentSelector (title-fetch): %d titles for case_id=%s",
        request_id, len(titles), case_id,
    )
    return {"doc_titles": titles}


# ---------------------------------------------------------------------------
# Branch node (Phase 3 — per-sub-question LLM classifier)
# ---------------------------------------------------------------------------


def branchDocSelector(state: SubQuestionState) -> Dict[str, Any]:
    """Classify one sub-question's document intent inside the branch.

    Runs as the first node of every parallel branch. Reuses the existing
    DOC_SELECTOR_SYSTEM_PROMPT_TEMPLATE, DocSelection model, and
    fuzzy_match_doc_title helper -- no new logic, just relocated from
    documentSelector.

    Reads doc_titles from SubQuestionState (injected at dispatch time from
    AgentState.doc_titles). This means zero extra MongoDB calls regardless of
    fan-out width.

    NOTE: Each parallel branch invocation incurs one extra LLM call here. For
    a 3-sub-question fan-out that is 3 additional calls. This is the explicit
    tradeoff of per-branch accuracy vs. main-graph classification latency.

    On any exception, falls back to mode="no_doc_specified", doc_id=None so the
    branch continues to retrieval rather than crashing.

    Returns
    -------
    {"doc_selection_mode": str, "selected_doc_id": Optional[str]}
    """
    request_id = state.get("request_id", "")
    sub_question = state.get("sub_question", "")
    doc_titles = state.get("doc_titles", [])

    # Graceful no-op for empty sub_question
    if not sub_question or not sub_question.strip():
        logger.warning(
            "[%s] branchDocSelector: empty sub_question, defaulting to no_doc_specified",
            request_id,
        )
        return {"doc_selection_mode": "no_doc_specified", "selected_doc_id": None}

    if doc_titles:
        docs_list = "\n".join(f"- {title}" for title in doc_titles)
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
        result = chain.invoke({"query": sub_question})
        mode = result.mode
        doc_id = result.doc_id
    except Exception:
        logger.exception(
            "[%s] branchDocSelector: LLM error, falling back to no_doc_specified",
            request_id,
        )
        return {"doc_selection_mode": "no_doc_specified", "selected_doc_id": None}

    # Validate doc_id against available titles using fuzzy match
    if doc_id is not None and doc_titles:
        if doc_id not in doc_titles:
            matched = fuzzy_match_doc_title(doc_id, doc_titles)
            if matched:
                logger.info(
                    "[%s] branchDocSelector: fuzzy matched '%s' -> '%s'",
                    request_id, doc_id, matched,
                )
                doc_id = matched
            else:
                logger.warning(
                    "[%s] branchDocSelector: doc_id '%s' not found in titles, "
                    "falling back to no_doc_specified",
                    request_id, doc_id,
                )
                doc_id = None
                mode = "no_doc_specified"

    logger.info(
        "[%s] branchDocSelector: sub_question=%r mode=%s doc_id=%s",
        request_id, sub_question[:80], mode, doc_id,
    )
    return {"doc_selection_mode": mode, "selected_doc_id": doc_id}


# ---------------------------------------------------------------------------
# Branch finalizer node (Phase 4 — operates on SubQuestionState)
# ---------------------------------------------------------------------------


def BranchDocumentFinalizer(state: SubQuestionState) -> Dict[str, Any]:
    """Retrieve and return a specific document directly from MongoDB.

    Branch-adapted version of the former main-graph DocumentFinalizer.
    Operates on SubQuestionState instead of AgentState:
      - reads  sub_question  (instead of query)
      - writes sub_answer / sources / found / sub_answers

    This node is a terminal branch node -- it routes to END of the branch
    sub-graph when branchDocSelectorRouter returns "BranchDocumentFinalizer".

    On any failure, returns a graceful Arabic error string in sub_answers so
    that mergeAnswers / the Supervisor can surface it to the judge without
    crashing the pipeline.
    """
    request_id = state.get("request_id", "")
    doc_id = state.get("selected_doc_id")
    sub_question = state.get("sub_question", "")

    if doc_id is None:
        logger.warning(
            "[%s] BranchDocumentFinalizer: no doc_id specified", request_id
        )
        message = "لم يتم تحديد مستند للاسترجاع."
        entry = {
            "question": sub_question,
            "answer": message,
            "sources": [],
            "found": False,
        }
        return {
            "sub_answer": message,
            "sources": [],
            "found": False,
            "sub_answers": [entry],
        }

    try:
        collection = get_mongo_collection()
        doc = collection.find_one({"title": doc_id})
    except Exception:
        logger.exception(
            "[%s] BranchDocumentFinalizer: MongoDB error for doc_id='%s'",
            request_id, doc_id,
        )
        message = "حدث خطأ أثناء استرجاع المستند من قاعدة البيانات."
        entry = {
            "question": sub_question,
            "answer": message,
            "sources": [],
            "found": False,
        }
        return {
            "sub_answer": message,
            "sources": [],
            "found": False,
            "sub_answers": [entry],
        }

    if doc is None:
        logger.warning(
            "[%s] BranchDocumentFinalizer: doc '%s' not found in MongoDB",
            request_id, doc_id,
        )
        message = f"لم يتم العثور على المستند '{doc_id}' في قاعدة البيانات."
        entry = {
            "question": sub_question,
            "answer": message,
            "sources": [],
            "found": False,
        }
        return {
            "sub_answer": message,
            "sources": [],
            "found": False,
            "sub_answers": [entry],
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
    source_ref = f"{source_file}:chunk_{chunk_index}"

    answer_entry = {
        "question": sub_question,
        "answer": extracted_text,
        "sources": [source_ref],
        "found": True,
    }

    logger.info(
        "[%s] BranchDocumentFinalizer: returning doc '%s' (%d chars)",
        request_id, doc_id, len(extracted_text),
    )
    return {
        "sub_answer": extracted_text,
        "sources": [source_ref],
        "found": True,
        "sub_answers": [answer_entry],
    }


# ---------------------------------------------------------------------------
# DEPRECATED: main-graph DocumentFinalizer (Phase 9)
# ---------------------------------------------------------------------------


def DocumentFinalizer(state: AgentState) -> Dict[str, Any]:
    """DEPRECATED -- no longer wired into any graph.

    The per-branch equivalent is BranchDocumentFinalizer (above).
    Kept temporarily so that existing unit tests that import this symbol do not
    raise ImportError. Remove this function once all tests are updated to target
    BranchDocumentFinalizer via the branch sub-graph (Phase 10).
    """
    logger.warning(
        "DocumentFinalizer (main-graph) was called -- this node is deprecated "
        "and should not be reachable in the current graph. Use "
        "BranchDocumentFinalizer instead."
    )
    request_id = state.get("request_id", "")
    doc_id = state.get("selected_doc_id")

    if doc_id is None:
        return {
            "final_answer": "لم يتم تحديد مستند للاسترجاع.",
            "error": "DocumentFinalizer (deprecated) reached with no doc_id",
        }

    try:
        collection = get_mongo_collection()
        doc = collection.find_one({"title": doc_id})
    except Exception:
        logger.exception(
            "[%s] DocumentFinalizer (deprecated): MongoDB error for doc_id='%s'",
            request_id, doc_id,
        )
        return {
            "error": f"Failed to retrieve document '{doc_id}' from MongoDB",
            "final_answer": "حدث خطأ أثناء استرجاع المستند من قاعدة البيانات.",
        }

    if doc is None:
        return {
            "error": f"Document '{doc_id}' not found in MongoDB",
            "final_answer": f"لم يتم العثور على المستند '{doc_id}' في قاعدة البيانات.",
        }

    extracted_text = ""
    for field_name in ("content", "text", "body"):
        val = doc.get(field_name)
        if val and isinstance(val, str) and val.strip():
            extracted_text = val
            break

    if not extracted_text:
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

    return {
        "sub_answers": [answer_entry],
        "final_answer": extracted_text,
    }
