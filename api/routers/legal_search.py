"""
legal_search.py

GET /api/v1/legal/search — direct legal corpus search, no supervisor graph.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_current_user
from api.errors import MCP_UNAVAILABLE, INVALID_CORPUS, VALIDATION_ERROR
from api.schemas.common import ErrorEnvelope
from api.schemas.legal_search import LegalSearchResponse, LegalArticleLookupResponse, LegalCorpusTreeResponse
from api.services.legal_search_service import search_articles, VALID_CORPORA, lookup_articles, get_corpus_tree

logger = logging.getLogger("hakim.api.legal_search")

router = APIRouter(prefix="/api/v1/legal", tags=["LegalSearch"])


@router.get(
    "/search",
    response_model=LegalSearchResponse,
    summary="Search the legal corpus directly",
    description=(
        "Search the civil law corpus without invoking the supervisor graph. "
        f"Supported corpora: {VALID_CORPORA}. Only 'civil' is available today."
    ),
    responses={
        400: {"model": ErrorEnvelope, "description": "Invalid corpus or query"},
        401: {"model": ErrorEnvelope},
        503: {"model": ErrorEnvelope, "description": "MCP server unavailable"},
    },
)
async def legal_search(
    q: str = Query(..., min_length=1, description="Search query in Arabic or English"),
    corpus: str = Query("civil", description="Corpus to search: civil"),
    user_id: str = Depends(get_current_user),
) -> LegalSearchResponse:
    if corpus not in VALID_CORPORA:
        raise HTTPException(
            status_code=400,
            detail={
                "code": INVALID_CORPUS,
                "message": f"Invalid corpus '{corpus}'. Valid options: {VALID_CORPORA}",
            },
        )

    try:
        result = await search_articles(query=q, corpus=corpus)
    except RuntimeError as exc:
        msg = str(exc)
        if "MCP_UNAVAILABLE" in msg:
            raise HTTPException(
                status_code=503,
                detail={"code": MCP_UNAVAILABLE, "message": "Legal RAG MCP server unavailable"},
            )
        raise HTTPException(
            status_code=500,
            detail={"code": MCP_UNAVAILABLE, "message": msg},
        )
    except ValueError as exc:
        msg = str(exc)
        code = VALIDATION_ERROR if "ErrorCode" in msg or "QUERY" in msg else INVALID_CORPUS
        raise HTTPException(status_code=400, detail={"code": code, "message": msg})

    return LegalSearchResponse(**result)


@router.get(
    "/article",
    response_model=LegalArticleLookupResponse,
    summary="Fetch legal articles by number, chapter, or section",
    responses={
        400: {"model": ErrorEnvelope},
        401: {"model": ErrorEnvelope},
    },
)
async def legal_article_lookup(
    corpus: str = Query("civil", description="Corpus: civil"),
    article_no: int | None = Query(None, description="Article number, e.g. 190"),
    chapter: str | None = Query(None, description="Chapter name/number"),
    section: str | None = Query(None, description="Section name/number"),
    user_id: str = Depends(get_current_user),
) -> LegalArticleLookupResponse:
    if corpus not in VALID_CORPORA:
        raise HTTPException(
            status_code=400,
            detail={"code": INVALID_CORPUS, "message": f"Invalid corpus '{corpus}'"},
        )
    try:
        result = await lookup_articles(
            corpus=corpus,
            article_no=article_no,
            chapter=chapter,
            section=section,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": VALIDATION_ERROR, "message": str(exc)},
        )
    return LegalArticleLookupResponse(**result)


@router.get(
    "/corpus/tree",
    response_model=LegalCorpusTreeResponse,
    summary="Return the full legal corpus as a structured tree",
    description=(
        "Returns all articles nested as book → part → chapter → section → article. "
        "Intended for building a legal dictionary / table of contents UI."
    ),
    responses={
        400: {"model": ErrorEnvelope},
        401: {"model": ErrorEnvelope},
    },
)
async def legal_corpus_tree(
    corpus: str = Query("civil", description="Corpus: civil"),
    user_id: str = Depends(get_current_user),
) -> LegalCorpusTreeResponse:
    if corpus not in VALID_CORPORA:
        raise HTTPException(
            status_code=400,
            detail={"code": INVALID_CORPUS, "message": f"Invalid corpus '{corpus}'"},
        )
    result = await get_corpus_tree(corpus=corpus)
    return LegalCorpusTreeResponse(**result)