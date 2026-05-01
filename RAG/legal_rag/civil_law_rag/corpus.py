"""
corpus.py
---------
CorpusConfig for the Egyptian Civil Law (القانون المدني المصري).

Bump corpus_version whenever civil_law.txt changes.
Bump prompts_version whenever any prompt in legal_rag/prompts.py changes.
Both trigger automatic semantic cache invalidation.
"""

from __future__ import annotations

from pathlib import Path

from RAG.legal_rag.corpus_config import CorpusConfig
from RAG.legal_rag.prompts import PROMPTS_VERSION

_DOCS_PATH = str(
    Path(__file__).resolve().parent / "docs" / "civil_law.txt"
)

CIVIL_LAW_CORPUS = CorpusConfig(
    name                = "civil_law",
    collection_name     = "civil_law_docs",
    source_filter_value = "civil_law",
    docs_path           = _DOCS_PATH,
    law_display_name    = "القانون المدني المصري",
    corpus_version      = "1.0.0",
    prompts_version     = PROMPTS_VERSION,
)
