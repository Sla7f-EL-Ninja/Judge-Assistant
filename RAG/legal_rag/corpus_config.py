"""
corpus_config.py
----------------
CorpusConfig — single dataclass that makes the legal_rag engine
corpus-agnostic.

Every node that previously hardcoded "civil_law", collection names, or
law display names now reads from this object, which is carried in
state["corpus_config"] for the lifetime of a single graph invocation.

Usage::

    from RAG.legal_rag.corpus_config import CorpusConfig

    MY_CORPUS = CorpusConfig(
        name                = "civil_law",
        collection_name     = "civil_law_docs",
        source_filter_value = "civil_law",
        docs_path           = "/path/to/civil_law.txt",
        law_display_name    = "القانون المدني المصري",
        corpus_version      = "1.0.0",
        prompts_version     = "1.2.0",
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusConfig:
    """Immutable configuration for one legal corpus.

    Frozen so it is safe to share across threads and graph invocations
    without defensive copying.

    Attributes:
        name:                 Short machine identifier, e.g. "civil_law".
                              Used as a key in singleton caches.
        collection_name:      Qdrant collection to read/write.
        source_filter_value:  Value of metadata.source in Qdrant payloads.
                              Must match the value written by the splitter.
        docs_path:            Absolute path to the tagged source .txt file.
        law_display_name:     Human-readable name injected into LLM prompts,
                              e.g. "القانون المدني المصري".
        corpus_version:       Bumped whenever the source text changes.
                              Automatically invalidates the semantic cache.
        prompts_version:      Bumped whenever any prompt text changes.
                              Automatically invalidates the semantic cache.
    """

    name:                 str
    collection_name:      str
    source_filter_value:  str
    docs_path:            str
    law_display_name:     str
    corpus_version:       str
    prompts_version:      str
