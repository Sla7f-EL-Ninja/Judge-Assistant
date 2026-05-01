"""
toc.py
------
Table-of-contents extractor for any tagged legal source file.

Parses BOOK / PART / CHAPTER / SECTION heading blocks and emits a
structured hierarchy used by the scope classifier to map a query to
its nearest chapter → section before retrieval.

API change from v1: load_toc(docs_path) now takes an explicit path
argument so the same module serves any corpus (civil law, evidence law…).
The in-memory + JSON cache is keyed by docs_path.

Cache: a JSON file (.toc_cache.json) next to docs_path, keyed by
source mtime+size, so the parse only happens once after an index rebuild.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache  {docs_path_str: List[dict]}
# ---------------------------------------------------------------------------
_toc_cache: Dict[str, List[dict]] = {}
_toc_lock  = threading.Lock()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def _make_section(id: str, title: str) -> dict:
    return {"id": id, "title": title}


def _make_chapter(
    id: str,
    title: str,
    book: Optional[str],
    part: Optional[str],
) -> dict:
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
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
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
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
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

def _source_fingerprint(docs_path: str) -> str:
    p = Path(docs_path)
    stat = p.stat()
    return f"{stat.st_mtime:.0f}:{stat.st_size}"


def _cache_file_path(docs_path: str) -> Path:
    return Path(docs_path).parent / ".toc_cache.json"


def _load_disk_cache(docs_path: str) -> Optional[list]:
    cache_file = _cache_file_path(docs_path)
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        # Cache may contain entries for multiple docs_paths; find the right one.
        entry = data.get(docs_path) or data  # backward compat with old single-entry format
        if isinstance(entry, dict) and entry.get("fingerprint") == _source_fingerprint(docs_path):
            return entry["chapters"]
    except Exception:
        pass
    return None


def _save_disk_cache(docs_path: str, chapters: list) -> None:
    cache_file = _cache_file_path(docs_path)
    # Load existing multi-entry cache if present
    existing: dict = {}
    if cache_file.exists():
        try:
            existing = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(existing, dict) or "chapters" in existing:
                # Old single-entry format — reset
                existing = {}
        except Exception:
            existing = {}

    existing[docs_path] = {
        "fingerprint": _source_fingerprint(docs_path),
        "chapters": chapters,
    }
    try:
        cache_file.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        log_event(logger, "toc_cache_write_error", error=str(exc), level=30)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_toc(docs_path: str) -> List[dict]:
    """Return the parsed TOC for *docs_path*, building and caching on first call.

    Thread-safe: concurrent calls for the same path block until the first
    completes; calls for different paths may run concurrently.

    Returns list of chapter dicts::

        [{"id": "1", "title": "...", "book": "...", "part": "...",
          "sections": [{"id": "1.1", "title": "..."}, ...]}, ...]
    """
    if docs_path in _toc_cache:
        return _toc_cache[docs_path]

    with _toc_lock:
        if docs_path in _toc_cache:
            return _toc_cache[docs_path]

        cached = _load_disk_cache(docs_path)
        if cached is not None:
            _toc_cache[docs_path] = cached
            log_event(logger, "toc_loaded_from_cache",
                      docs_path=docs_path, chapters=len(cached))
            return _toc_cache[docs_path]

        text     = Path(docs_path).read_text(encoding="utf-8")
        chapters = _parse_toc(text)
        _save_disk_cache(docs_path, chapters)
        _toc_cache[docs_path] = chapters
        log_event(logger, "toc_built", docs_path=docs_path, chapters=len(chapters))

    return _toc_cache[docs_path]


def get_toc_hash(docs_path: str) -> str:
    """Fingerprint of the TOC source — used as a cache key component."""
    return _source_fingerprint(docs_path)
