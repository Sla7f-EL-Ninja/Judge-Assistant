from typing import Any, Dict


def civil_law_rag_tool(query: str) -> Dict[str, Any]:
    from RAG.civil_law_rag.service import ask_question
    result = ask_question(query)
    return {
        "answer": result.answer,
        "sources": result.sources,          # [{article, title, book, part, chapter}]
        "classification": result.classification,
        "retrieval_confidence": result.retrieval_confidence,
        "citation_integrity": result.citation_integrity,
        "from_cache": result.from_cache,
        "error": None,
    }


def case_documents_rag_tool(query: str, case_id: str) -> Dict[str, Any]:
    from RAG.case_doc_rag import run
    return run(query=query, case_id=case_id)
