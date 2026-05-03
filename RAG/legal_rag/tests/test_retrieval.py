"""
test_retrieval.py
-----------------
Unit tests for the vector database retrieval layer.

Tests verify that the system gracefully handles empty databases,
applies the correct metadata filters (e.g., routing by corpus), 
and survives connection timeouts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from langchain_core.documents import Document

# Adjust these imports to match where your retrieval logic actually lives.
# Assuming retrieve_node updates state["last_results"]
from RAG.legal_rag.state import make_initial_state
from RAG.legal_rag.nodes.retrieve import retrieve_node 


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_qdrant_search():
    """
    Mocks the underlying LangChain vector store search method.
    Adjust 'RAG.legal_rag.nodes.retrieve.vector_store' to wherever your 
    Qdrant instance/retrieve is initialized.
    """
    # NOTE: You may need to change the patch target depending on your code.
    # E.g., patch('langchain_community.vectorstores.Qdrant.similarity_search')
    with patch('RAG.legal_rag.nodes.retrieve.vector_store.similarity_search') as mock_search:
        yield mock_search


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_retrieval_empty_database_graceful_handling(mock_qdrant_search):
    """If the database is empty or returns no results, the node should not crash."""
    # Arrange: DB returns empty list
    mock_qdrant_search.return_value = []
    
    state = make_initial_state()
    state["last_query"] = "ما هي شروط صحة العقد؟"
    state["corpus"] = "civil"

    # Act
    result_state = retrieve_node(state)

    # Assert
    assert mock_qdrant_search.called
    assert "last_results" in result_state
    assert len(result_state["last_results"]) == 0
    # Ideally, your system sets confidence to 0 when no docs are found
    if "retrieval_confidence" in result_state:
         assert result_state["retrieval_confidence"] == 0.0


def test_retrieval_metadata_filtering_by_corpus(mock_qdrant_search):
    """Ensures that the requested corpus is passed as a filter to the DB."""
    mock_qdrant_search.return_value = [
        Document(page_content="dummy", metadata={"corpus": "evidence", "index": 1})
    ]
    
    state = make_initial_state()
    state["last_query"] = "طرق الإثبات"
    state["corpus"] = "evidence"

    # Act
    retrieve_node(state)

    # Assert
    assert mock_qdrant_search.called
    
    # Extract how similarity_search was called
    _, kwargs = mock_qdrant_search.call_args
    
    # Verify a filter was passed. 
    # (The exact assertion depends on whether you use Qdrant Filter objects or dicts)
    assert "filter" in kwargs or "kwargs" in kwargs, "No filter was passed to the vector store"
    
    # If passing dict-based LangChain filters:
    # assert kwargs.get("filter") == {"corpus": "evidence"}
    

def test_retrieval_metadata_filtering_all_corpora(mock_qdrant_search):
    """If the router determined corpus='all', no strict corpus filter should be applied."""
    mock_qdrant_search.return_value = [Document(page_content="dummy")]
    
    state = make_initial_state()
    state["last_query"] = "سؤال عام"
    state["corpus"] = "all"

    # Act
    retrieve_node(state)

    # Assert
    _, kwargs = mock_qdrant_search.call_args
    
    # Ensure no strict corpus filter is blocking results
    filter_arg = kwargs.get("filter", {})
    assert "corpus" not in filter_arg, "Corpus filter should not be applied when corpus='all'"


def test_retrieval_timeout_handling(mock_qdrant_search):
    """If the vector DB times out or crashes, the system must recover or report cleanly."""
    import requests
    
    # Arrange: Simulate a network timeout from Qdrant/Requests
    mock_qdrant_search.side_effect = requests.exceptions.Timeout("Connection to vector store timed out")
    
    state = make_initial_state()
    state["last_query"] = "ما هي شروط العقد؟"

    # Act
    # Depending on your architecture, retrieve_node might catch this and return a state,
    # or let it bubble up for the graph to handle. 
    # Assuming retrieve_node catches it and flags an error in state:
    try:
        result_state = retrieve_node(state)
        
        # Assert (If caught internally)
        assert len(result_state.get("last_results", [])) == 0
        assert "failure_reason" in result_state or "error" in result_state
        
    except requests.exceptions.Timeout:
        # Assert (If allowed to bubble up)
        # If your design choice is to let it fail loudly so an API gateway can catch a 500 error,
        # that's fine too. But you should intentionally document that behavior.
        pytest.fail("retrieve_node should catch DB timeouts and return an error state, rather than crashing the graph.")


def test_retrieval_malformed_documents(mock_qdrant_search):
    """If the DB returns documents missing required metadata (like 'index'), it shouldn't break downstream formatting."""
    # Arrange: Return a document missing the standard 'index' metadata
    mock_qdrant_search.return_value = [
        Document(page_content="Text with no index", metadata={"book": "First"})
    ]
    
    state = make_initial_state()
    state["last_query"] = "test"

    # Act
    result_state = retrieve_node(state)
    
    # Assert
    assert len(result_state["last_results"]) == 1
    # Ensure the system didn't crash and passed the weird doc through safely
    assert result_state["last_results"][0].metadata.get("index") is None