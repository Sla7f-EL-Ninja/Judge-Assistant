# from typing import Any, Dict


# def civil_law_rag_tool(query: str) -> Dict[str, Any]:
#     from RAG.civil_law_rag.service import ask_question
#     result = ask_question(query)
#     return {
#         "answer": result.answer,
#         "sources": result.sources,          # [{article, title, book, part, chapter}]
#         "classification": result.classification,
#         "retrieval_confidence": result.retrieval_confidence,
#         "citation_integrity": result.citation_integrity,
#         "from_cache": result.from_cache,
#         "error": None,
#     }


# def case_documents_rag_tool(query: str, case_id: str) -> Dict[str, Any]:
#     from RAG.case_doc_rag import run
#     return run(query=query, case_id=case_id)


from typing import Any, Dict


def civil_law_rag_tool(query: str) -> Dict[str, Any]:
    """Query the unified legal RAG pipeline.

    Previously pointed at RAG.civil_law_rag.service (civil law only).
    Now uses the unified RAG.legal_rag.service so corpus routing is
    automatic — the query is routed to civil, evidence, or procedures
    law based on its content.

    The return shape is identical to the old civil-law-only version so
    all callers (retrieval.py, validation.py) need no changes.
    """
    from RAG.legal_rag.service import ask_question
    result = ask_question(query)
    return {
        "answer":               result.answer,
        "sources":              result.sources,
        "classification":       result.classification,
        "retrieval_confidence": result.retrieval_confidence,
        "citation_integrity":   result.citation_integrity,
        "from_cache":           result.from_cache,
        "corpus":               result.corpus,
        "error":                None,
    }


def case_documents_rag_tool(query: str, case_id: str) -> Dict[str, Any]:
    from RAG.case_doc_rag import run
    return run(query=query, case_id=case_id)