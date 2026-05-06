"""
scope_classifier.py
-------------------
Two-stage LangGraph node that narrows retrieval scope before hitting Qdrant.

Stage 1 — Chapter classification:
    Query + enumerated chapter list → {chapter_id, confidence}

Stage 2 — Section classification (only when chapter confidence ≥ threshold):
    Query + sections of that chapter → {section_id, confidence}

Fallback rules:
    - section confidence < threshold  → filter by chapter only
    - chapter confidence < threshold  → no metadata filter (global search)

State keys written:
    current_chapter, current_section, scope_confidence, scope_filter
"""

from __future__ import annotations

import json
import re
from typing import Optional

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from config.legal_rag import get_llm, MAX_LLM_CALLS, LLM_TIMEOUT, cfg
from RAG.legal_rag.indexing.toc import load_toc
from RAG.legal_rag.prompts import CHAPTER_CLASSIFIER_PROMPT, SECTION_CLASSIFIER_PROMPT
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm("medium")
    return _llm


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _chapter_threshold() -> float:
    return float(
        cfg.get("rag", {}).get("legal", {}).get("scope_chapter_threshold", 0.5)
    )


def _section_threshold() -> float:
    return float(
        cfg.get("rag", {}).get("legal", {}).get("scope_section_threshold", 0.5)
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_chapters(chapters: list) -> str:
    lines = []
    for ch in chapters:
        book  = ch.get("book") or ""
        part  = ch.get("part") or ""
        ctx   = " > ".join(filter(None, [book, part]))
        label = f"[{ch['id']}] {ch['title']}"
        if ctx:
            label += f"  ({ctx})"
        lines.append(label)
    return "\n".join(lines)


def _format_sections(sections: list) -> str:
    if not sections:
        return "(لا توجد أقسام محددة في هذا الفصل)"
    return "\n".join(f"[{s['id']}] {s['title']}" for s in sections)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _classify_chapter(query: str, law_name: str, chapters: list, state: dict) -> Optional[dict]:
    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        log_event(logger, "scope_chapter_skipped", reason="llm_budget_exhausted")
        return None
    prompt = CHAPTER_CLASSIFIER_PROMPT.format(
        law_name=law_name,
        query=query,
        chapters=_format_chapters(chapters),
    )
    try:
        resp = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
        state["llm_call_count"] = state.get("llm_call_count", 0) + 1
        return json.loads(_strip_fences(resp.content.strip()))
    except Exception as exc:
        log_event(logger, "scope_chapter_error", error=str(exc))
        return None


def _classify_section(
    query: str,
    law_name: str,
    chapter: dict,
    state: dict,
) -> Optional[dict]:
    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        log_event(logger, "scope_section_skipped", reason="llm_budget_exhausted")
        return None
    sections = chapter.get("sections", [])
    if not sections:
        return None
    prompt = SECTION_CLASSIFIER_PROMPT.format(
        law_name=law_name,
        query=query,
        chapter_title=chapter["title"],
        sections=_format_sections(sections),
    )
    try:
        resp = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
        state["llm_call_count"] = state.get("llm_call_count", 0) + 1
        return json.loads(_strip_fences(resp.content.strip()))
    except Exception as exc:
        log_event(logger, "scope_section_error", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@traceable(name="Scope Classifier Node")
def scope_classifier_node(state: dict) -> dict:
    """Classify query scope to chapter → section; write scope_filter to state."""
    query         = (
        state.get("refined_query")
        or state.get("rewritten_question")
        or state.get("last_query", "")
    )
    corpus_config = state.get("corpus_config")
    law_name      = corpus_config.law_display_name if corpus_config else "القانون"
    docs_path     = corpus_config.docs_path if corpus_config else ""

    chapters  = load_toc(docs_path)
    ch_thresh = _chapter_threshold()
    sc_thresh = _section_threshold()

    # ── Stage 1: chapter ─────────────────────────────────────────────────────
    ch_result = _classify_chapter(query, law_name, chapters, state)

    if not ch_result or ch_result.get("confidence", 0) < ch_thresh:
        log_event(logger, "scope_no_filter", query=query,
                  chapter_confidence=ch_result.get("confidence") if ch_result else None)
        state["scope_filter"]     = {}
        state["scope_confidence"] = ch_result.get("confidence", 0.0) if ch_result else 0.0
        return state

    ch_id   = str(ch_result["chapter_id"])
    ch_conf = float(ch_result.get("confidence", 0))
    chapter = next((c for c in chapters if c["id"] == ch_id), None)

    if chapter is None:
        log_event(logger, "scope_chapter_not_found", chapter_id=ch_id)
        state["scope_filter"]     = {}
        state["scope_confidence"] = 0.0
        return state

    state["current_chapter"] = chapter["title"]

    # ── Stage 2: section ─────────────────────────────────────────────────────
    sc_result = _classify_section(query, law_name, chapter, state)

    if sc_result and sc_result.get("confidence", 0) >= sc_thresh:
        sc_id   = str(sc_result["section_id"])
        section = next((s for s in chapter["sections"] if s["id"] == sc_id), None)

        if section:
            state["current_section"]  = section["title"]
            state["scope_filter"]     = {
                "chapter": chapter["title"],
                "section": section["title"],
            }
            state["scope_confidence"] = float(sc_result.get("confidence", ch_conf))
            log_event(logger, "scope_chapter_and_section",
                      chapter=chapter["title"], section=section["title"],
                      ch_confidence=ch_conf, sc_confidence=state["scope_confidence"])
            return state

    # Chapter found, section uncertain → filter by chapter only
    state["current_section"]  = None
    state["scope_filter"]     = {"chapter": chapter["title"]}
    state["scope_confidence"] = ch_conf
    log_event(logger, "scope_chapter_only",
              chapter=chapter["title"], ch_confidence=ch_conf,
              sc_confidence=sc_result.get("confidence") if sc_result else None)
    return state
