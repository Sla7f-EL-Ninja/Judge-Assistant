"""
legal_search.py

Schemas for the direct legal corpus search endpoint.
"""

from typing import List, Optional, Any

from pydantic import BaseModel


class LegalArticleItem(BaseModel):
    index: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    text: str

class LegalArticleLookupResponse(BaseModel):
    corpus: str
    filters: dict[str, Any]
    count: int
    articles: List[LegalArticleItem]

class LegalSearchResponse(BaseModel):
    query: str
    corpus: str
    answer: str
    sources: List[Any]
    retrieval_confidence: Optional[float] = None
    from_cache: bool = False

class LegalArticleNode(BaseModel):
    index: Optional[int] = None
    title: Optional[str] = None
    text: str

class LegalSectionNode(BaseModel):
    section: str
    articles: List[LegalArticleNode]

class LegalChapterNode(BaseModel):
    chapter: str
    direct_articles: List[LegalArticleNode] = []  # articles with chapter but no section
    sections: List[LegalSectionNode]

class LegalPartNode(BaseModel):
    part: str
    direct_articles: List[LegalArticleNode] = []  # articles with part but no chapter/section
    chapters: List[LegalChapterNode]

class LegalBookNode(BaseModel):
    book: str
    direct_articles: List[LegalArticleNode] = []  # articles with no part/chapter/section
    parts: List[LegalPartNode]

class LegalCorpusTreeResponse(BaseModel):
    corpus: str
    total_articles: int
    tree: List[LegalBookNode]