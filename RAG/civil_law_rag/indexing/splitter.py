# """
# splitter.py
# -----------
# Egyptian Civil Law hierarchical text splitter.

# Parses the tagged civil law format:
#     [PREFACE]...[/PREFACE]
#     [BOOK]...[/BOOK]
#     [PART]...[/PART]
#     [CHAPTER]...[/CHAPTER]
#     [SECTION]...[/SECTION]
#     [ARTICLE id=N]...[/ARTICLE]

# Hierarchy: book → part → chapter → section → article

# Each article Document includes metadata:
#     {
#         "type":       "article",
#         "title":      "مادة (N)",
#         "index":      int,
#         "book":       str | None,
#         "part":       str | None,
#         "chapter":    str | None,
#         "section":    str | None,
#         "source":     "civil_law",
#     }
# """

# from __future__ import annotations

# import re
# from typing import List

# from langchain_core.documents import Document


# # ---------------------------------------------------------------------------
# # Arabic-Indic → Western digit conversion
# # ---------------------------------------------------------------------------

# def _to_western(s: str) -> str:
#     return s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


# def _parse_article_id(id_str: str) -> int | None:
#     """Convert article id attribute to integer index.
#     Handles: '1', '٧٣٩', 'issuance_1' (returns None for non-numeric).
#     """
#     cleaned = _to_western(id_str.strip())
#     if cleaned.isdigit():
#         return int(cleaned)
#     return None


# # ---------------------------------------------------------------------------
# # Tag extraction helpers
# # ---------------------------------------------------------------------------

# def _inner(tag: str, text: str) -> str:
#     """Return the inner text of the first occurrence of [TAG]...[/TAG]."""
#     m = re.search(rf"\[{tag}[^\]]*\](.*?)\[/{tag}\]", text, re.DOTALL)
#     return m.group(1).strip() if m else ""


# def _first_line(text: str) -> str:
#     lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
#     return lines[0] if lines else text.strip()


# # ---------------------------------------------------------------------------
# # Main splitter
# # ---------------------------------------------------------------------------

# def split_egyptian_civil_law(text: str) -> List[Document]:
#     """Parse tagged civil law text into structured Documents.

#     Args:
#         text: Full tagged civil law text (UTF-8 Arabic).

#     Returns:
#         List of Document objects — one per article, ready for embedding.
#     """
#     docs: List[Document] = []

#     current_book:    str | None = None
#     current_part:    str | None = None
#     current_chapter: str | None = None
#     current_section: str | None = None

#     # Tokenize: split into a flat list of (tag_type, attributes, content)
#     # We scan the text linearly using a single regex that matches any tag block
#     token_pat = re.compile(
#         r"\[(?P<close>/)?(?P<tag>PREFACE|BOOK|PART|CHAPTER|SECTION|ARTICLE)"
#         r"(?P<attrs>[^\]]*)\]",
#         re.IGNORECASE,
#     )

#     pos = 0
#     tag_stack: list[tuple[str, str, int]] = []  # (tag, attrs, start_pos)

#     for m in token_pat.finditer(text):
#         tag  = m.group("tag").upper()
#         is_close = bool(m.group("close"))
#         attrs = m.group("attrs").strip()

#         if not is_close:
#             tag_stack.append((tag, attrs, m.end()))
#         else:
#             # Find the matching open tag
#             for i in range(len(tag_stack) - 1, -1, -1):
#                 if tag_stack[i][0] == tag:
#                     open_tag, open_attrs, content_start = tag_stack.pop(i)
#                     content = text[content_start:m.start()].strip()
#                     _handle_block(
#                         tag=open_tag,
#                         attrs=open_attrs,
#                         content=content,
#                         docs=docs,
#                         current_book=current_book,
#                         current_part=current_part,
#                         current_chapter=current_chapter,
#                         current_section=current_section,
#                     )
#                     # Update context trackers
#                     if open_tag == "BOOK":
#                         current_book    = _first_line(content)
#                         current_part    = None
#                         current_chapter = None
#                         current_section = None
#                     elif open_tag == "PART":
#                         current_part    = _first_line(content)
#                         current_chapter = None
#                         current_section = None
#                     elif open_tag == "CHAPTER":
#                         current_chapter = _first_line(content)
#                         current_section = None
#                     elif open_tag == "SECTION":
#                         current_section = content.replace("\n", " ").strip()
#                     break

#     return docs


# # ---------------------------------------------------------------------------
# # Block handler
# # ---------------------------------------------------------------------------

# def _handle_block(
#     tag: str,
#     attrs: str,
#     content: str,
#     docs: List[Document],
#     current_book: str | None,
#     current_part: str | None,
#     current_chapter: str | None,
#     current_section: str | None,
# ) -> None:
#     """Process a parsed tag block and append to docs if applicable."""

#     if tag == "ARTICLE":
#         # Parse id= attribute
#         id_match = re.search(r'id\s*=\s*"?([^"\s\]]+)"?', attrs)
#         article_id = id_match.group(1) if id_match else ""
#         index = _parse_article_id(article_id)

#         if index is None:
#             # Non-numeric id (issuance articles etc.) — skip or store as preface
#             return

#         # Extract clean title from first line of content
#         first = _first_line(content)
#         # Normalize title to "مادة (N)" format
#         title = f"مادة ({index})"

#         meta = {
#             "type":    "article",
#             "title":   title,
#             "index":   index,
#             "book":    current_book,
#             "part":    current_part,
#             "chapter": current_chapter,
#             "section": current_section,
#             "source":  "civil_law",
#         }
#         docs.append(Document(page_content=content, metadata=meta))

#     # BOOK, PART, CHAPTER, SECTION are context-only — no documents created
#     # Their content updates the tracker variables in the caller

"""
splitter.py
-----------
Egyptian Civil Law hierarchical text splitter.

Parses the tagged civil law format:
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
        "source":     "civil_law",
    }
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
    """Join all non-empty lines with ' - ' to form the full chapter title.
    
    Example:
        "(الفصل الأول)\\nالعقد" → "(الفصل الأول) - العقد"
    
    Must match toc.py _chapter_title() exactly.
    """
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
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

def split_egyptian_civil_law(text: str) -> List[Document]:
    """Parse tagged civil law text into one Document per article."""
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
                            break  # skip issuance_1, issuance_2 etc.

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
                                "source":  "civil_law",
                            },
                        ))

                    break  # PREFACE — skip entirely

    return docs