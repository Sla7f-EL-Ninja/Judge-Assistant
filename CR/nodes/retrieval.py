"""Law and Fact Retrieval Nodes — per-element tool calls, no LLM."""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MAX_QUERY_LEN = 500


def _parse_articles(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for art in articles:
        num = art.get("article_number")
        if num not in seen:
            seen.add(num)
            result.append(art)
    return result


def retrieve_law_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from CR.tools import civil_law_rag_tool

    issue_title: str = state.get("issue_title", "")
    law_queries: List[Dict] = state.get("law_queries") or []

    if not law_queries:
        source_text = state.get("source_text", "")
        fallback_query = f"{issue_title} {source_text}"[:_MAX_QUERY_LEN]
        law_queries = [{"element_id": "E0", "query": fallback_query}]
        logger.warning("retrieve_law_node: no law_queries — using fallback query for '%s'", issue_title)

    all_articles: List[Dict] = []
    last_result: Dict = {}
    errors: List[str] = []

    for item in law_queries:
        element_id = item.get("element_id", "")
        query = item.get("query", "")[:_MAX_QUERY_LEN]
        try:
            result = civil_law_rag_tool(query)
            parsed = _parse_articles(result.get("sources") or [])
            all_articles.extend(parsed)
            last_result = result
            logger.debug("Law retrieval [%s] '%s': %d articles", element_id, issue_title, len(parsed))
        except Exception as exc:
            logger.error("retrieve_law_node [%s] failed for '%s': %s", element_id, issue_title, exc)
            errors.append(f"retrieve_law_node [{element_id}] '{issue_title}': {exc}")

    deduplicated = _deduplicate_articles(all_articles)
    logger.info("Law retrieval for '%s': %d total articles (%d unique) across %d queries",
                issue_title, len(all_articles), len(deduplicated), len(law_queries))

    output = {
        "law_retrieval_result": last_result or {"answer": "", "sources": [], "error": errors[0] if errors else None},
        "retrieved_articles": deduplicated,
        "intermediate_steps": [f"استرداد القانون '{issue_title}': {len(deduplicated)} مادة فريدة من {len(law_queries)} استعلام"],
    }
    if errors:
        output["error_log"] = errors
    return output


def retrieve_facts_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from CR.tools import case_documents_rag_tool

    issue_title: str = state.get("issue_title", "")
    case_id: str = state.get("case_id", "")
    fact_queries: List[Dict] = state.get("fact_queries") or []

    if not fact_queries:
        source_text = state.get("source_text", "")
        fallback_query = f"{issue_title} {source_text}"
        fact_queries = [{"element_id": "E0", "query": fallback_query}]
        logger.warning("retrieve_facts_node: no fact_queries — using fallback query for '%s'", issue_title)

    per_element_facts: List[str] = []
    errors: List[str] = []

    for item in fact_queries:
        element_id = item.get("element_id", "")
        query = item.get("query", "")
        try:
            result = case_documents_rag_tool(query, case_id)
            answer = result.get("final_answer") or ""
            if result.get("error"):
                logger.warning("retrieve_facts_node [%s] partial error for '%s': %s",
                               element_id, issue_title, result["error"])
            if answer.strip():
                per_element_facts.append(f"[{element_id}]:\n{answer.strip()}")
            logger.debug("Fact retrieval [%s] '%s': %d chars", element_id, issue_title, len(answer))
        except Exception as exc:
            logger.error("retrieve_facts_node [%s] failed for '%s': %s", element_id, issue_title, exc)
            errors.append(f"retrieve_facts_node [{element_id}] '{issue_title}': {exc}")

    retrieved_facts = "\n\n".join(per_element_facts) if per_element_facts else ""
    logger.info("Fact retrieval for '%s': %d elements answered, %d chars total",
                issue_title, len(per_element_facts), len(retrieved_facts))

    output = {
        "fact_retrieval_result": {"final_answer": retrieved_facts, "sub_answers": per_element_facts},
        "retrieved_facts": retrieved_facts,
        "intermediate_steps": [f"استرداد الوقائع '{issue_title}': {len(per_element_facts)} عنصر، {len(retrieved_facts)} حرف"],
    }
    if errors:
        output["error_log"] = errors
    return output