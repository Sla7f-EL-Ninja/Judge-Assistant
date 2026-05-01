"""
corpus.py
---------
CorpusConfig for the Egyptian Law of Evidence in Civil and
Commercial Matters (قانون الإثبات في المواد المدنية والتجارية).

Bump corpus_version whenever evidence_law.txt changes.
Bump prompts_version whenever any prompt in legal_rag/prompts.py changes.
"""

from __future__ import annotations

from pathlib import Path

from RAG.legal_rag.corpus_config import CorpusConfig
from RAG.legal_rag.prompts import PROMPTS_VERSION

_DOCS_PATH = str(
    Path(__file__).resolve().parent / "docs" / "evidence_law.txt"
)

EVIDENCE_CORPUS = CorpusConfig(
    name                = "evidence_law",
    collection_name     = "evidence_law_docs",
    source_filter_value = "evidence_law",
    docs_path           = _DOCS_PATH,
    law_display_name    = "قانون الإثبات في المواد المدنية والتجارية",
    corpus_version      = "1.0.0",
    prompts_version     = PROMPTS_VERSION,
)
