# """
# toc.py
# ------
# Table-of-contents extractor for the Egyptian Civil Law tagged source.

# Parses BOOK / PART / CHAPTER / SECTION heading blocks and emits a
# structured hierarchy used by the scope classifier to map a query to
# its nearest chapter → section before retrieval.

# Cache: a JSON file (.toc_cache.json) keyed by source mtime+size so the
# parse only happens once after an index rebuild.
# """

# from __future__ import annotations

# import json
# import re
# from pathlib import Path
# from typing import List, Optional

# from RAG.civil_law_rag.config import DOCS_PATH
# from RAG.civil_law_rag.telemetry import get_logger, log_event

# logger = get_logger(__name__)

# _CACHE_FILE = Path(DOCS_PATH).parent / ".toc_cache.json"

# # ---------------------------------------------------------------------------
# # Data structures
# # ---------------------------------------------------------------------------

# def _make_section(id: str, title: str) -> dict:
#     return {"id": id, "title": title}


# def _make_chapter(id: str, title: str, book: Optional[str], part: Optional[str]) -> dict:
#     return {"id": id, "title": title, "book": book, "part": part, "sections": []}


# # ---------------------------------------------------------------------------
# # Parsing helpers
# # ---------------------------------------------------------------------------

# _TAG_PAT = re.compile(
#     r"\[(?P<close>/)?(?P<tag>BOOK|PART|CHAPTER|SECTION)(?P<attrs>[^\]]*)\]",
#     re.IGNORECASE,
# )


# def _chapter_title(content: str) -> str:
#     """Build a human-readable chapter title from the chapter block content.

#     Chapter blocks typically have two non-empty lines:
#         Line 1: "(الفصل الأول)"   — the chapter number label
#         Line 2: "القانون وتطبيقه" — the topical title

#     We join both so the classifier sees the full context.
#     """
#     lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
#     if not lines:
#         return content.strip()[:80]
#     if len(lines) == 1:
#         return lines[0]
#     return f"{lines[0]} - {lines[1]}"


# def _section_title(content: str) -> str:
#     lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
#     return lines[0] if lines else content.strip()[:80]


# def _first_line(content: str) -> str:
#     lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
#     return lines[0] if lines else content.strip()[:80]


# # ---------------------------------------------------------------------------
# # Main parser
# # ---------------------------------------------------------------------------

# def _parse_toc(text: str) -> List[dict]:
#     """Return a list of chapter dicts with nested section lists."""
#     chapters: List[dict] = []

#     current_book:    Optional[str] = None
#     current_part:    Optional[str] = None
#     chapter_counter: int = 0
#     section_counter: int = 0

#     tag_stack: list[tuple[str, int]] = []  # (tag, content_start)

#     for m in _TAG_PAT.finditer(text):
#         tag      = m.group("tag").upper()
#         is_close = bool(m.group("close"))

#         if not is_close:
#             tag_stack.append((tag, m.end()))
#         else:
#             for i in range(len(tag_stack) - 1, -1, -1):
#                 if tag_stack[i][0] == tag:
#                     _, content_start = tag_stack.pop(i)
#                     content = text[content_start:m.start()].strip()

#                     if tag == "BOOK":
#                         current_book = _first_line(content)
#                         current_part = None

#                     elif tag == "PART":
#                         current_part = _first_line(content)

#                     elif tag == "CHAPTER":
#                         chapter_counter += 1
#                         section_counter  = 0
#                         title = _chapter_title(content)
#                         chapters.append(
#                             _make_chapter(
#                                 id=str(chapter_counter),
#                                 title=title,
#                                 book=current_book,
#                                 part=current_part,
#                             )
#                         )

#                     elif tag == "SECTION":
#                         section_counter += 1
#                         title = _section_title(content)
#                         if chapters:
#                             chapters[-1]["sections"].append(
#                                 _make_section(
#                                     id=f"{chapter_counter}.{section_counter}",
#                                     title=title,
#                                 )
#                             )
#                     break

#     return chapters


# # ---------------------------------------------------------------------------
# # Cache helpers
# # ---------------------------------------------------------------------------

# def _source_fingerprint() -> str:
#     p = Path(DOCS_PATH)
#     stat = p.stat()
#     return f"{stat.st_mtime:.0f}:{stat.st_size}"


# def _load_cache() -> Optional[list]:
#     if not _CACHE_FILE.exists():
#         return None
#     try:
#         data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
#         if data.get("fingerprint") == _source_fingerprint():
#             return data["chapters"]
#     except Exception:
#         pass
#     return None


# def _save_cache(chapters: list) -> None:
#     try:
#         _CACHE_FILE.write_text(
#             json.dumps({"fingerprint": _source_fingerprint(), "chapters": chapters},
#                        ensure_ascii=False, indent=2),
#             encoding="utf-8",
#         )
#     except Exception as exc:
#         log_event(logger, "toc_cache_write_error", error=str(exc), level=30)


# # ---------------------------------------------------------------------------
# # Public API
# # ---------------------------------------------------------------------------

# _toc_cache: Optional[List[dict]] = None


# def load_toc() -> List[dict]:
#     """Return the parsed TOC, building and caching it on first call.

#     Returns list of chapter dicts:
#         [{"id": "1", "title": "...", "book": "...", "part": "...",
#           "sections": [{"id": "1.1", "title": "..."}, ...]}, ...]
#     """
#     global _toc_cache
#     if _toc_cache is not None:
#         return _toc_cache

#     cached = _load_cache()
#     if cached is not None:
#         _toc_cache = cached
#         log_event(logger, "toc_loaded_from_cache", chapters=len(cached))
#         return _toc_cache

#     text = Path(DOCS_PATH).read_text(encoding="utf-8")
#     chapters = _parse_toc(text)
#     _save_cache(chapters)
#     _toc_cache = chapters
#     log_event(logger, "toc_built", chapters=len(chapters))
#     return _toc_cache


# def get_toc_hash() -> str:
#     """Fingerprint of the TOC source — used as a cache key component."""
#     return _source_fingerprint()


"""
toc.py
------
Table-of-contents extractor for the Egyptian Civil Law tagged source.

Parses BOOK / PART / CHAPTER / SECTION heading blocks and emits a
structured hierarchy used by the scope classifier to map a query to
its nearest chapter → section before retrieval.

Cache: a JSON file (.toc_cache.json) keyed by source mtime+size so the
parse only happens once after an index rebuild.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from RAG.civil_law_rag.config import DOCS_PATH
from RAG.civil_law_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_CACHE_FILE = Path(DOCS_PATH).parent / ".toc_cache.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def _make_section(id: str, title: str) -> dict:
    return {"id": id, "title": title}


def _make_chapter(id: str, title: str, book: Optional[str], part: Optional[str]) -> dict:
    return {"id": id, "title": title, "book": book, "part": part, "sections": []}


# ---------------------------------------------------------------------------
# Parsing helpers — must match splitter.py exactly
# ---------------------------------------------------------------------------

_TAG_PAT = re.compile(
    r"\[(?P<close>/)?(?P<tag>BOOK|PART|CHAPTER|SECTION)(?P<attrs>[^\]]*)\]",
    re.IGNORECASE,
)


def _flatten(content: str) -> str:
    """Collapse all whitespace/newlines into a single space."""
    return " ".join(content.split())


def _chapter_title(content: str) -> str:
    """Join first two non-empty lines with ' - '.

    Matches splitter.py _chapter_title() exactly so TOC titles
    align with Qdrant metadata.chapter values.

    Example:
        "(الفصل الأول)\\nالعقد" → "(الفصل الأول) - العقد"
    """
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    return f"{lines[0]} - {lines[1]}"


def _section_title(content: str) -> str:
    """Flatten section content to single line.

    Matches splitter.py _section_title() exactly.
    """
    return _flatten(content)


def _first_line(content: str) -> str:
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    return lines[0] if lines else content.strip()[:80]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def _parse_toc(text: str) -> List[dict]:
    """Return a list of chapter dicts with nested section lists."""
    chapters: List[dict] = []

    current_book:    Optional[str] = None
    current_part:    Optional[str] = None
    chapter_counter: int = 0
    section_counter: int = 0

    tag_stack: list[tuple[str, int]] = []  # (tag, content_start)

    for m in _TAG_PAT.finditer(text):
        tag      = m.group("tag").upper()
        is_close = bool(m.group("close"))

        if not is_close:
            tag_stack.append((tag, m.end()))
        else:
            for i in range(len(tag_stack) - 1, -1, -1):
                if tag_stack[i][0] == tag:
                    _, content_start = tag_stack.pop(i)
                    content = text[content_start:m.start()].strip()

                    if tag == "BOOK":
                        current_book = _flatten(content)
                        current_part = None

                    elif tag == "PART":
                        current_part = _flatten(content)

                    elif tag == "CHAPTER":
                        chapter_counter += 1
                        section_counter  = 0
                        chapters.append(
                            _make_chapter(
                                id=str(chapter_counter),
                                title=_chapter_title(content),
                                book=current_book,
                                part=current_part,
                            )
                        )

                    elif tag == "SECTION":
                        section_counter += 1
                        if chapters:
                            chapters[-1]["sections"].append(
                                _make_section(
                                    id=f"{chapter_counter}.{section_counter}",
                                    title=_section_title(content),
                                )
                            )
                    break

    return chapters


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _source_fingerprint() -> str:
    p = Path(DOCS_PATH)
    stat = p.stat()
    return f"{stat.st_mtime:.0f}:{stat.st_size}"


def _load_cache() -> Optional[list]:
    if not _CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if data.get("fingerprint") == _source_fingerprint():
            return data["chapters"]
    except Exception:
        pass
    return None


def _save_cache(chapters: list) -> None:
    try:
        _CACHE_FILE.write_text(
            json.dumps(
                {"fingerprint": _source_fingerprint(), "chapters": chapters},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        log_event(logger, "toc_cache_write_error", error=str(exc), level=30)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_toc_cache: Optional[List[dict]] = None


def load_toc() -> List[dict]:
    """Return the parsed TOC, building and caching it on first call."""
    global _toc_cache
    if _toc_cache is not None:
        return _toc_cache

    cached = _load_cache()
    if cached is not None:
        _toc_cache = cached
        log_event(logger, "toc_loaded_from_cache", chapters=len(cached))
        return _toc_cache

    text = Path(DOCS_PATH).read_text(encoding="utf-8")
    chapters = _parse_toc(text)
    _save_cache(chapters)
    _toc_cache = chapters
    log_event(logger, "toc_built", chapters=len(chapters))
    return _toc_cache


def get_toc_hash() -> str:
    """Fingerprint of the TOC source — used as a cache key component."""
    return _source_fingerprint()

