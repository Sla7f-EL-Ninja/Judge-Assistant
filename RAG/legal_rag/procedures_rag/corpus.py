"""
corpus.py
---------
CorpusConfig for the Egyptian Law of Procedures in Civil and
Commercial Matters (قانون الإجراءات في المواد المدنية والتجارية).

Bump corpus_version whenever procedures_law.txt changes.
Bump prompts_version whenever any prompt in legal_rag/prompts.py changes.
"""

from __future__ import annotations

from pathlib import Path

from RAG.legal_rag.corpus_config import CorpusConfig
from RAG.legal_rag.prompts import PROMPTS_VERSION

_DOCS_PATH = str(
    Path(__file__).resolve().parent / "docs" / "procedures_law.txt"
)

PROCEDURES_CORPUS = CorpusConfig(
    name                = "procedures_law",
    collection_name     = "procedures_law_docs",
    source_filter_value = "procedures_law",
    docs_path           = _DOCS_PATH,
    law_display_name    = "قانون المرافعات المدنية والتجارية",
    corpus_version      = "1.0.0",
    prompts_version     = PROMPTS_VERSION,
)
