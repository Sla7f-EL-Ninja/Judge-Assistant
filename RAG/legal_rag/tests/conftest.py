"""conftest.py for RAG/legal_rag/tests/ — shared fixtures."""
import pytest
from RAG.legal_rag.corpus_config import CorpusConfig


@pytest.fixture
def civil_corpus() -> CorpusConfig:
    return CorpusConfig(
        name="civil",
        collection_name="civil_law_docs",
        source_filter_value="civil_law",
        docs_path="/fake/civil_law.txt",
        law_display_name="القانون المدني المصري",
        corpus_version="1.0.0",
        prompts_version="1.0.0",
    )
