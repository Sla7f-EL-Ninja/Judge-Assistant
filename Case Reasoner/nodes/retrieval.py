"""Law and Fact Retrieval Nodes — tool calls, no LLM."""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Civil law RAG max query length (matches service.py MAX_QUERY_LENGTH)
_MAX_QUERY_LEN = 500


def _parse_articles(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize CivilLawResult.sources into a flat list with article_number as int."""
    articles = []
    for src in sources:
        raw_num = src.get("article") or src.get("article_number")
        if raw_num is None:
            continue
        try:
            article_number = int(str(raw_num).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
        except (ValueError, TypeError):
            continue
        articles.append({
            "article_number": article_number,
            "article_text": src.get("content", src.get("article_text", "")),
            "title": src.get("title", ""),
            "book": src.get("book", ""),
            "part": src.get("part", ""),
            "chapter": src.get("chapter", ""),
        })
    return articles


def retrieve_law_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from tools import civil_law_rag_tool

    issue_title = state.get("issue_title", "")
    source_text = state.get("source_text", "")
    query = f"{issue_title} {source_text}"[:_MAX_QUERY_LEN]

    try:
        result = civil_law_rag_tool(query)
        retrieved_articles = _parse_articles(result.get("sources") or [])
        logger.info("Law retrieval for '%s': %d articles", issue_title, len(retrieved_articles))
        return {
            "law_retrieval_result": result,
            "retrieved_articles": retrieved_articles,
            "intermediate_steps": [f"استرداد القانون '{issue_title}': {len(retrieved_articles)} مادة"],
        }
    except Exception as exc:
        logger.error("retrieve_law_node failed for '%s': %s", issue_title, exc)
        return {
            "law_retrieval_result": {"answer": "", "sources": [], "error": str(exc)},
            "retrieved_articles": [],
            "error_log": [f"retrieve_law_node '{issue_title}': {exc}"],
            "intermediate_steps": [f"استرداد القانون '{issue_title}': فشل"],
        }


def retrieve_facts_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from tools import case_documents_rag_tool

    issue_title = state.get("issue_title", "")
    source_text = state.get("source_text", "")
    case_id = state.get("case_id", "")
    query = f"{issue_title} {source_text}"

    try:
        result = case_documents_rag_tool(query, case_id)
        retrieved_facts = result.get("final_answer") or ""
        if result.get("error"):
            logger.warning("retrieve_facts_node partial error for '%s': %s", issue_title, result["error"])
        logger.info("Fact retrieval for '%s': %d chars", issue_title, len(retrieved_facts))
        return {
            "fact_retrieval_result": result,
            "retrieved_facts": retrieved_facts,
            "intermediate_steps": [f"استرداد الوقائع '{issue_title}': {len(retrieved_facts)} حرف"],
        }
    except Exception as exc:
        logger.error("retrieve_facts_node failed for '%s': %s", issue_title, exc)
        return {
            "fact_retrieval_result": {"final_answer": "", "sub_answers": [], "error": str(exc)},
            "retrieved_facts": "",
            "error_log": [f"retrieve_facts_node '{issue_title}': {exc}"],
            "intermediate_steps": [f"استرداد الوقائع '{issue_title}': فشل"],
        }
