"""
generate.py
-----------
Generate Answer node: produces the final legal analysis.

Includes a citation-faithfulness check: any article numbers cited in the
answer are verified against the retrieved article indices.  Invented
citations (hallucinated article numbers) are stripped from the answer
and logged.
"""

from __future__ import annotations

import logging
import re
from typing import List, Set
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_core.documents import Document
from langsmith import traceable

from config import get_llm

MAX_LLM_CALLS: int = 5
LLM_TIMEOUT: int = 30
from RAG.civil_law_rag.prompts import ANALYTICAL_PROMPT
from RAG.civil_law_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm("high")
    return _llm


def _retrieved_article_indices(docs: List[Document]) -> Set[int]:
    """Return the set of article index numbers present in retrieved docs."""
    indices: Set[int] = set()
    for doc in docs:
        idx = doc.metadata.get("index")
        if isinstance(idx, int):
            indices.add(idx)
    return indices


def _verify_citations(
    answer: str,
    retrieved_indices: Set[int],
) -> tuple[str, str]:
    """Strip any cited article numbers not in *retrieved_indices*.

    Returns:
        (cleaned_answer, citation_integrity)
        citation_integrity: "full" | "partial" | "none"
    """
    cited = {int(m) for m in re.findall(r"المادة\s+(\d+)", answer)}
    if not cited:
        return answer, "none"

    invalid = cited - retrieved_indices
    if not invalid:
        return answer, "full"

    # Strip invented citations — replace "المادة X" where X not in retrieved
    def _replacer(match: re.Match) -> str:
        num = int(match.group(1))
        if num in invalid:
            return ""  # remove the citation
        return match.group(0)

    # cleaned = re.sub(r"المادة\s+\d+", _replacer, answer)
    cleaned = re.sub(r"المادة\s+(\d+)", _replacer, answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    log_event(
        logger, "citation_integrity",
        cited=sorted(cited),
        retrieved=sorted(retrieved_indices),
        invalid=sorted(invalid),
        level=logging.WARNING,
    )
    return cleaned, "partial"


@traceable(name="Generate Answer Node")
def generate_answer_node(state: dict) -> dict:
    """Generate final legal analysis grounded in retrieved articles."""
    docs = state.get("last_results", [])

    if not docs:
        state["final_answer"] = (
            "لم يتم العثور على مواد قانونية ذات صلة للإجابة على السؤال."
        )
        return state

    # Budget guard
    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        state["final_answer"] = (
            "تعذر توليد الإجابة: تجاوز ميزانية استدعاءات النموذج."
        )
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

    prompt = ANALYTICAL_PROMPT.format(context_text=context_text, query=query)
    response = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
    state["llm_call_count"] = state.get("llm_call_count", 0) + 1

    answer = response.content.strip()

    # Citation verification
    retrieved_indices = _retrieved_article_indices(docs)
    answer, citation_integrity = _verify_citations(answer, retrieved_indices)

    state["final_answer"] = answer
    if "citation_integrity" not in state:
        state["citation_integrity"] = citation_integrity

    log_event(logger, "generate",
              docs=len(docs),
              answer_len=len(answer),
              citation_integrity=citation_integrity)
    return state