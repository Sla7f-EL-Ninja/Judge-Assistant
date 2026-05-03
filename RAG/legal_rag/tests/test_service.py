# """
# test_service.py
# ---------------
# Unit tests for the service layer (validation, cache, source extraction).
# No network or LLM calls are made.
# """

# from __future__ import annotations

# import pytest

# from RAG.civil_law_rag.errors import QueryValidationError
# from RAG.civil_law_rag.service import validate_query, _extract_sources


# # ---------------------------------------------------------------------------
# # validate_query
# # ---------------------------------------------------------------------------

# def test_validate_empty_raises():
#     with pytest.raises(QueryValidationError):
#         validate_query("")


# def test_validate_none_raises():
#     with pytest.raises(QueryValidationError):
#         validate_query(None)  # type: ignore


# def test_validate_too_short_raises():
#     with pytest.raises(QueryValidationError):
#         validate_query("مم")  # 2 chars, below MIN_QUERY_LENGTH=3


# def test_validate_too_long_raises():
#     with pytest.raises(QueryValidationError):
#         validate_query("م" * 2001)


# def test_validate_no_arabic_raises():
#     with pytest.raises(QueryValidationError):
#         validate_query("hello world completely english query no arabic at all")


# def test_validate_valid_arabic():
#     q = validate_query("ما هي شروط صحة العقد؟")
#     assert q == "ما هي شروط صحة العقد؟"


# def test_validate_strips_whitespace():
#     q = validate_query("  ما هي شروط العقد؟  ")
#     assert q == "ما هي شروط العقد؟"


# # ---------------------------------------------------------------------------
# # _extract_sources
# # ---------------------------------------------------------------------------

# def test_extract_sources_populated():
#     from langchain_core.documents import Document
#     docs = [
#         Document(page_content="text", metadata={
#             "index": 89, "title": "المادة 89", "book": "الكتاب الأول",
#             "part": "الباب الأول", "chapter": None
#         }),
#         Document(page_content="text", metadata={
#             "index": 90, "title": "المادة 90", "book": None, "part": None, "chapter": None
#         }),
#     ]
#     result_state = {"last_results": docs}
#     sources = _extract_sources(result_state)
#     assert len(sources) == 2
#     assert sources[0]["article"] == 89
#     assert sources[1]["article"] == 90


# def test_extract_sources_no_index():
#     from langchain_core.documents import Document
#     docs = [
#         Document(page_content="text", metadata={"title": "preface"}),  # no index
#     ]
#     sources = _extract_sources({"last_results": docs})
#     assert sources == []


# def test_extract_sources_empty():
#     sources = _extract_sources({"last_results": []})
#     assert sources == []


"""
test_service.py
---------------
Unit tests for the service layer (validation, cache, source extraction).
"""

from __future__ import annotations

import pytest

# UPDATE: Pointing to legal_rag
from RAG.legal_rag.errors import QueryValidationError
from RAG.legal_rag.service import validate_query, _extract_sources


# ---------------------------------------------------------------------------
# validate_query
# ---------------------------------------------------------------------------

def test_validate_empty_raises():
    with pytest.raises(QueryValidationError):
        validate_query("")


def test_validate_too_short_raises():
    with pytest.raises(QueryValidationError):
        validate_query("مم") 


def test_validate_too_long_raises():
    with pytest.raises(QueryValidationError):
        validate_query("م" * 2001)


def test_validate_no_arabic_raises():
    with pytest.raises(QueryValidationError):
        validate_query("hello world completely english query no arabic at all")


def test_validate_valid_arabic():
    q = validate_query("ما هي شروط صحة العقد؟")
    assert q == "ما هي شروط صحة العقد؟"


# ---------------------------------------------------------------------------
# _extract_sources
# ---------------------------------------------------------------------------

def test_extract_sources_populated():
    from langchain_core.documents import Document
    docs = [
        Document(page_content="text", metadata={
            "index": 89, "title": "المادة 89", "book": "الكتاب الأول"
        }),
        Document(page_content="text", metadata={
            "index": 90, "title": "المادة 90", "book": None
        }),
    ]
    result_state = {"last_results": docs}
    sources = _extract_sources(result_state)
    assert len(sources) == 2
    assert sources[0]["article"] == 89
    assert sources[1]["article"] == 90


def test_extract_sources_no_index():
    from langchain_core.documents import Document
    docs = [
        Document(page_content="text", metadata={"title": "preface"}), 
    ]
    sources = _extract_sources({"last_results": docs})
    assert sources == []