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