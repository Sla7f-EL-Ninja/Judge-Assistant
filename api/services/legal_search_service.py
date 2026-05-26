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
    "civil":      "civil_law",
    "evidence":   "evidence_law",
    "procedural": "procedural_law",
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
    "civil":      "civil_law_docs",
    "evidence":   "evidence_law_docs",
    "procedural": "procedures_law_docs",   # ← matches your Qdrant exactly
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
    articles.sort(key=lambda a: a["index"] if a["index"] is not None else float("inf"))

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


async def get_corpus_tree(corpus: str) -> dict:
    collection = _COLLECTION_MAP.get(corpus)
    if collection is None:
        raise ValueError(f"Invalid corpus '{corpus}'. Valid: {list(_COLLECTION_MAP)}")

    from api.db.qdrant import get_qdrant_client
    client = get_qdrant_client()

    all_points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    # ---------------------------------------------------------------------------
    # Build intermediate dicts using None-aware routing so articles land at the
    # correct structural level instead of being forced into "غير محدد" nodes.
    #
    # Routing rules (checked in order):
    #   part is None   → book.direct_articles
    #   chapter is None → part.direct_articles
    #   section is None → chapter.direct_articles
    #   else            → chapter.sections[section]
    # ---------------------------------------------------------------------------

    books: dict = {}  # book_name -> {"direct": [], "parts": {}}

    def _book(name):
        if name not in books:
            books[name] = {"direct": [], "parts": {}}
        return books[name]

    def _part(book_node, name):
        if name not in book_node["parts"]:
            book_node["parts"][name] = {"direct": [], "chapters": {}}
        return book_node["parts"][name]

    def _chapter(part_node, name):
        if name not in part_node["chapters"]:
            part_node["chapters"][name] = {"direct": [], "sections": {}}
        return part_node["chapters"][name]

    def _section(chapter_node, name):
        if name not in chapter_node["sections"]:
            chapter_node["sections"][name] = []
        return chapter_node["sections"][name]

    for p in all_points:
        m       = p.payload.get("metadata", {})
        book    = m.get("book")    or "غير محدد"
        part    = m.get("part")    or None
        chapter = m.get("chapter") or None
        section = m.get("section") or None

        article = {
            "index": m.get("index"),
            "title": m.get("title"),
            "text":  p.payload.get("page_content") or p.payload.get("text", ""),
        }

        book_node = _book(book)

        if part is None:
            book_node["direct"].append(article)
        elif chapter is None:
            _part(book_node, part)["direct"].append(article)
        elif section is None:
            _chapter(_part(book_node, part), chapter)["direct"].append(article)
        else:
            _section(_chapter(_part(book_node, part), chapter), section).append(article)

    # ---------------------------------------------------------------------------
    # Serialize to list structure — all levels sorted by minimum article index
    # so the output order matches the legislative sequence, not Arabic
    # lexicographic order.
    # ---------------------------------------------------------------------------

    def _sort(lst):
        """Sort article dicts by index; None indices go last."""
        return sorted(lst, key=lambda a: a["index"] if a["index"] is not None else float("inf"))

    # ── min-index helpers ────────────────────────────────────────────────────

    def _min_idx(articles):
        idxs = [a["index"] for a in articles if a.get("index") is not None]
        return min(idxs) if idxs else float("inf")

    def _chapter_min(ch_data):
        arts = list(ch_data["direct"])
        for sec_arts in ch_data["sections"].values():
            arts.extend(sec_arts)
        return _min_idx(arts)

    def _part_min(part_data):
        arts = list(part_data["direct"])
        for ch_data in part_data["chapters"].values():
            arts.extend(ch_data["direct"])
            for sec_arts in ch_data["sections"].values():
                arts.extend(sec_arts)
        return _min_idx(arts)

    def _book_min(book_data):
        arts = list(book_data["direct"])
        for part_data in book_data["parts"].values():
            arts.extend(part_data["direct"])
            for ch_data in part_data["chapters"].values():
                arts.extend(ch_data["direct"])
                for sec_arts in ch_data["sections"].values():
                    arts.extend(sec_arts)
        return _min_idx(arts)

    result_tree = []
    for book_name, book_data in sorted(books.items(), key=lambda kv: _book_min(kv[1])):
        book_node = {
            "book": book_name,
            "direct_articles": _sort(book_data["direct"]),
            "parts": [],
        }
        for part_name, part_data in sorted(book_data["parts"].items(), key=lambda kv: _part_min(kv[1])):
            part_node = {
                "part": part_name,
                "direct_articles": _sort(part_data["direct"]),
                "chapters": [],
            }
            for chapter_name, chapter_data in sorted(part_data["chapters"].items(), key=lambda kv: _chapter_min(kv[1])):
                chapter_node = {
                    "chapter": chapter_name,
                    "direct_articles": _sort(chapter_data["direct"]),
                    "sections": [],
                }
                for section_name, articles in sorted(
                    chapter_data["sections"].items(), key=lambda kv: _min_idx(kv[1])
                ):
                    chapter_node["sections"].append({
                        "section": section_name,
                        "articles": _sort(articles),
                    })
                part_node["chapters"].append(chapter_node)
            book_node["parts"].append(part_node)
        result_tree.append(book_node)

    return {
        "corpus": corpus,
        "total_articles": len(all_points),
        "tree": result_tree,
    }