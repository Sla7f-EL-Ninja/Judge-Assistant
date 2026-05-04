"""
legal_search_service.py

Direct legal corpus search via MCP legal_rag server.
"""

import json
import logging
from mcp_servers.errors import MCPUnavailable, ToolError, ErrorCode
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger("hakim.api.legal_search")

# Map user-facing corpus names to MCP corpus identifiers
_CORPUS_MAP = {
    "civil": "civil_law",
}

VALID_CORPORA = list(_CORPUS_MAP.keys())

async def search_articles(query: str, corpus: str) -> dict:
    mcp_corpus = _CORPUS_MAP.get(corpus)
    if mcp_corpus is None:
        raise ValueError(f"Invalid corpus '{corpus}'. Valid: {VALID_CORPORA}")

    try:
        from mcp_servers.lifecycle import get_client
        client = get_client("legal_rag")
    except (RuntimeError, KeyError) as exc:
        raise RuntimeError(f"MCP_UNAVAILABLE: {exc}") from exc

    try:
        data = client.call("search_legal_corpus", query=query, corpus=mcp_corpus)
    except MCPUnavailable as exc:
        raise RuntimeError(f"MCP_UNAVAILABLE: {exc}") from exc
    except ToolError as exc:
        raise ValueError(f"{exc.message}") from exc  # drop the [ErrorCode.X] prefix

    raw_sources = data.get("sources", [])
    sources = [
        f"المادة {s['article']} — {s.get('title', '')}" if isinstance(s, dict) else str(s)
        for s in raw_sources
    ]

    return {
        "query": query,
        "corpus": corpus,
        "answer": data.get("answer", ""),
        "sources": sources,
        "retrieval_confidence": data.get("retrieval_confidence"),
        "from_cache": data.get("from_cache", False),
    }


_COLLECTION_MAP = {
    "civil": "civil_law_docs",
}


async def lookup_articles(
    corpus: str,
    article_no: int | None = None,
    chapter: str | None = None,
    section: str | None = None,
) -> dict:
    collection = _COLLECTION_MAP.get(corpus)
    if collection is None:
        raise ValueError(f"Invalid corpus '{corpus}'. Valid: {list(_COLLECTION_MAP)}")

    from api.db.qdrant import get_qdrant_client
    client = get_qdrant_client()

    conditions = []
    if article_no is not None:
        conditions.append(FieldCondition(key="metadata.index", match=MatchValue(value=article_no)))
    if chapter is not None:
        conditions.append(FieldCondition(key="metadata.chapter", match=MatchValue(value=chapter)))
    if section is not None:
        conditions.append(FieldCondition(key="metadata.section", match=MatchValue(value=section)))

    if not conditions:
        raise ValueError("At least one filter (article_no, chapter, section) is required")

    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=conditions),
        limit=50,
        with_payload=True,
        with_vectors=False,
    )

    articles = [
        {
            "index": p.payload.get("metadata", {}).get("index"),
            "chapter": p.payload.get("metadata", {}).get("chapter"),
            "section": p.payload.get("metadata", {}).get("section"),
            "text": p.payload.get("page_content") or p.payload.get("text", ""),
        }
        for p in results
    ]
    articles.sort(key=lambda a: a["index"] or 0)

    return {
        "corpus": corpus,
        "filters": {
            "article_no": article_no,
            "chapter": chapter,
            "section": section,
        },
        "count": len(articles),
        "articles": articles,
    }