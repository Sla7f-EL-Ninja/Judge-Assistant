"""
generate.py
-----------
Generate Answer node: produces the final legal analysis.

Includes a citation-faithfulness check: article numbers cited in the
answer are verified against retrieved article indices. Invented
citations are stripped from the answer and logged.
"""

from __future__ import annotations

import logging
import re
from typing import List, Set
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langsmith import traceable

from config.legal_rag import get_llm, MAX_LLM_CALLS, LLM_TIMEOUT
from RAG.legal_rag.prompts import ANALYTICAL_PROMPT
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm("high")
    return _llm


def _retrieved_article_indices(docs: List[Document]) -> Set[int]:
    indices: Set[int] = set()
    for doc in docs:
        idx = doc.metadata.get("index")
        if isinstance(idx, int):
            indices.add(idx)
    return indices


def _verify_citations(answer: str, retrieved_indices: Set[int]) -> tuple[str, str]:
    """Strip any cited article numbers not in *retrieved_indices*.

    Returns:
        (cleaned_answer, citation_integrity)  — "full" | "partial" | "none"
    """
    cited = {int(m) for m in re.findall(r"المادة\s+(\d+)", answer)}
    if not cited:
        return answer, "none"

    invalid = cited - retrieved_indices
    if not invalid:
        return answer, "full"

    def _replacer(match: re.Match) -> str:
        return "" if int(match.group(1)) in invalid else match.group(0)

    cleaned = re.sub(r"المادة\s+(\d+)", _replacer, answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    log_event(logger, "citation_integrity",
              cited=sorted(cited),
              retrieved=sorted(retrieved_indices),
              invalid=sorted(invalid),
              level=logging.WARNING)
    return cleaned, "partial"


@traceable(name="Generate Answer Node")
def generate_answer_node(state: dict) -> dict:
    """Generate final legal analysis grounded in retrieved articles."""
    docs          = state.get("last_results", [])
    corpus_config = state.get("corpus_config")
    law_name      = corpus_config.law_display_name if corpus_config else "القانون"

    if not docs:
        state["final_answer"] = "لم يتم العثور على مواد قانونية ذات صلة للإجابة على السؤال."
        return state

    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        state["final_answer"] = "تعذر توليد الإجابة: تجاوز ميزانية استدعاءات النموذج."
        return state

    query = (
        state.get("refined_query")
        or state.get("rewritten_question")
        or state.get("last_query", "")
    )

    context_text = "\n\n".join(
        f"(المادة {d.metadata.get('index', '?')})\n{d.page_content}"
        for d in docs
    )

    prompt   = ANALYTICAL_PROMPT.format(
        law_name=law_name,
        context_text=context_text,
        query=query,
    )
    response = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
    state["llm_call_count"] = state.get("llm_call_count", 0) + 1

    answer = response.content.strip()

    retrieved_indices          = _retrieved_article_indices(docs)
    answer, citation_integrity = _verify_citations(answer, retrieved_indices)

    state["final_answer"]       = answer
    state["citation_integrity"] = citation_integrity

    log_event(logger, "generate",
              corpus=corpus_config.name if corpus_config else "unknown",
              docs=len(docs),
              answer_len=len(answer),
              citation_integrity=citation_integrity)
    return state
