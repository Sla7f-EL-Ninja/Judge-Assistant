"""Query Generation Node — produces per-element law and fact queries before retrieval."""
import logging
from typing import Any, Dict, List

from config import get_llm
from ..prompts import get_prompt

logger = logging.getLogger(__name__)


def generate_retrieval_queries_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from ..schemas import RetrievalQueries
    _QUERY_GENERATION_SYSTEM, _QUERY_GENERATION_USER = get_prompt("query_generation")

    issue_title: str = state.get("issue_title", "")
    legal_domain: str = state.get("legal_domain", "")
    source_text: str = state.get("source_text", "")
    required_elements: List[Dict] = state.get("required_elements") or []

    if not required_elements:
        return {
            "law_queries": [],
            "fact_queries": [],
            "intermediate_steps": [f"توليد الاستعلامات '{issue_title}': لا عناصر"],
        }

    elements_text = "\n".join(
        f"- [{el['element_id']}] ({el['element_type']}) {el['description']}"
        for el in required_elements
    )
    prompt = (
        f"{_QUERY_GENERATION_SYSTEM}\n\n"
        f"{_QUERY_GENERATION_USER.format(issue_title=issue_title, legal_domain=legal_domain, source_text=source_text, elements_text=elements_text)}"
    )

    llm = get_llm("medium")
    structured_llm = llm.with_structured_output(RetrievalQueries)

    try:
        result: RetrievalQueries = structured_llm.invoke(prompt)
        law_queries = [{"element_id": q.element_id, "query": q.law_query} for q in result.queries]
        fact_queries = [{"element_id": q.element_id, "query": q.fact_query} for q in result.queries]
        logger.info("Query generation for '%s': %d law queries, %d fact queries",
                    issue_title, len(law_queries), len(fact_queries))
        return {
            "law_queries": law_queries,
            "fact_queries": fact_queries,
            "intermediate_steps": [f"توليد الاستعلامات '{issue_title}': {len(law_queries)} استعلام قانوني، {len(fact_queries)} استعلام وقائعي"],
        }
    except Exception as exc:
        logger.warning("generate_retrieval_queries_node failed for '%s': %s — using fallback", issue_title, exc)
        law_queries = [
            {"element_id": el["element_id"], "query": f"{legal_domain}: {el['description']} — {issue_title}"}
            for el in required_elements
        ]
        fact_queries = [
            {"element_id": el["element_id"], "query": f"وقائع تتعلق بـ: {el['description']} في {issue_title}"}
            for el in required_elements
        ]
        return {
            "law_queries": law_queries,
            "fact_queries": fact_queries,
            "error_log": [f"generate_retrieval_queries_node '{issue_title}': {exc} — استعلامات احتياطية"],
            "intermediate_steps": [f"توليد الاستعلامات '{issue_title}': فشل، استعلامات احتياطية"],
        }