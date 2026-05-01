"""
splitter.py
-----------
Hierarchical text splitter for tagged Egyptian legal documents.

Parses the tagged format:
    [PREFACE]...[/PREFACE]
    [BOOK]...[/BOOK]
    [PART]...[/PART]
    [CHAPTER]...[/CHAPTER]
    [SECTION]...[/SECTION]
    [ARTICLE id=N]...[/ARTICLE]

Hierarchy: book → part → chapter → section → article

Each article Document includes metadata:
    {
        "type":       "article",
        "title":      "مادة (N)",
        "index":      int,
        "book":       str | None,
        "part":       str | None,
        "chapter":    str | None,
        "section":    str | None,
        "source":     <source_value>,   ← injected, not hardcoded
    }

API change from v1: split_legal_document(text, source_value) replaces
split_egyptian_civil_law(text).  The old name is kept as an alias for
backward compatibility.
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Arabic-Indic → Western digit conversion
# ---------------------------------------------------------------------------

def _to_western(s: str) -> str:
    return s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def _parse_article_id(id_str: str) -> Optional[int]:
    """Convert article id attribute to integer. Returns None for non-numeric ids."""
    cleaned = _to_western(id_str.strip())
    if cleaned.isdigit():
        return int(cleaned)
    return None


def _flatten(content: str) -> str:
    """Collapse all whitespace/newlines into a single space."""
    return " ".join(content.split())


def _chapter_title(content: str) -> str:
    """Join first two non-empty lines with ' - ' to form the full chapter title.

    Must match toc.py _chapter_title() exactly so metadata.chapter values
    written at index time align with TOC titles used by the scope classifier.

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
    """Return section as a single flattened line — no newlines.

    Must match toc.py _section_title() exactly.
    """
    return _flatten(content)


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------

def split_legal_document(text: str, source_value: str) -> List[Document]:
    """Parse a tagged legal text into one Document per article.

    Args:
        text:         Full tagged legal text (UTF-8 Arabic).
        source_value: Value written to metadata.source, e.g. "civil_law"
                      or "evidence_law".  Must match CorpusConfig.source_filter_value.

    Returns:
        List of Document objects — one per numeric article, ready for embedding.
    """
    docs: List[Document] = []

    current_book:    Optional[str] = None
    current_part:    Optional[str] = None
    current_chapter: Optional[str] = None
    current_section: Optional[str] = None

    tag_pat = re.compile(
        r"\[(?P<close>/)?(?P<tag>PREFACE|BOOK|PART|CHAPTER|SECTION|ARTICLE)"
        r"(?P<attrs>[^\]]*)\]",
        re.IGNORECASE,
    )

    tag_stack: list[tuple[str, str, int]] = []  # (tag, attrs, content_start)

    for m in tag_pat.finditer(text):
        tag      = m.group("tag").upper()
        is_close = bool(m.group("close"))
        attrs    = m.group("attrs").strip()

        if not is_close:
            tag_stack.append((tag, attrs, m.end()))
        else:
            for i in range(len(tag_stack) - 1, -1, -1):
                if tag_stack[i][0] == tag:
                    open_tag, open_attrs, content_start = tag_stack.pop(i)
                    content = text[content_start:m.start()].strip()

                    if open_tag == "BOOK":
                        current_book    = _flatten(content)
                        current_part    = None
                        current_chapter = None
                        current_section = None

                    elif open_tag == "PART":
                        current_part    = _flatten(content)
                        current_chapter = None
                        current_section = None

                    elif open_tag == "CHAPTER":
                        current_chapter = _chapter_title(content)
                        current_section = None

                    elif open_tag == "SECTION":
                        current_section = _section_title(content)

                    elif open_tag == "ARTICLE":
                        id_match = re.search(r'id\s*=\s*"?([^"\s\]]+)"?', open_attrs)
                        if not id_match:
                            break
                        index = _parse_article_id(id_match.group(1))
                        if index is None:
                            break  # skip issuance_1, issuance_2, etc.

                        docs.append(Document(
                            page_content=_flatten(content),
                            metadata={
                                "type":    "article",
                                "title":   f"مادة ({index})",
                                "index":   index,
                                "book":    current_book,
                                "part":    current_part,
                                "chapter": current_chapter,
                                "section": current_section,
                                "source":  source_value,
                            },
                        ))

                    break  # PREFACE — skip entirely

    return docs


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

def split_egyptian_civil_law(text: str) -> List[Document]:
    """Deprecated alias — use split_legal_document(text, 'civil_law') instead."""
    return split_legal_document(text, source_value="civil_law")
