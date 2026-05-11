"""
DocumentProcessor.classifier
-----------------------------
Two-stage Egyptian civil-document classifier.

Stage 1 — weighted keyword heuristic over normalized text.
  Strong markers (60 pts each) short-circuit to a direct answer when
  unambiguous (>=1 strong hit, >=30 pt margin over second-best).

Stage 2 — structured LLM call when heuristic is ambiguous.
  Heuristic top-3 candidates are passed as hints; output is validated
  against the taxonomy before being accepted.

Taxonomy lives in config/document_taxonomy.yaml — add new types there.

Public API: classify_document(text: str) -> {"final_type", "confidence", "explanation"}
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from config import get_llm
from config.taxonomy import get_taxonomy, get_unknown_label
from DocumentProcessor.arabic_norm import normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_STRONG_WEIGHT = 60
_WEAK_WEIGHT = 15
_ANTI_WEIGHT = -30
_AMBIGUITY_MARGIN = 30   # min score gap between top-2 for heuristic to win
_HEADER_LINES = 10
_BODY_WORDS = 500
_LLM_TIER = "low"
_LLM_TIMEOUT = 20        # seconds; overrides tier default for fast classification


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------

class _ClassificationResult(BaseModel):
    doc_type: str = Field(description="Exact document type from the provided list")
    confidence: int = Field(ge=0, le=100, description="Classification confidence 0–100")
    reasons: str = Field(description="Short Arabic explanation for the classification")


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def _prepare(text: str) -> Tuple[str, str]:
    """Normalize and split text into (header, body_excerpt).

    Normalizes per-line so newline boundaries survive for header extraction.
    """
    lines = text.split("\n")
    norm_lines = [normalize(line) for line in lines]
    header = "\n".join(norm_lines[:_HEADER_LINES])
    body_excerpt = " ".join(" ".join(norm_lines).split()[:_BODY_WORDS])
    return header, body_excerpt


# ---------------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------------

def _score_candidates(
    search_text: str,
) -> List[Tuple[str, int, int, List[str]]]:
    """Score all taxonomy types against search_text.

    Normalizes search_text before matching (safe to call with raw or pre-normalized text).
    Returns list of (doc_type, score, strong_hits, matched_keywords) sorted descending.
    """
    search_text = normalize(search_text)
    taxonomy = get_taxonomy()
    results = []

    for doc_type, entry in taxonomy["doc_types"].items():
        strong_hits = sum(1 for k in entry["strong"] if k in search_text)
        weak_hits = sum(1 for k in entry["weak"] if k in search_text)
        anti_hits = sum(1 for k in entry["anti"] if k in search_text)
        score = (
            strong_hits * _STRONG_WEIGHT
            + weak_hits * _WEAK_WEIGHT
            + anti_hits * _ANTI_WEIGHT
        )
        matched = (
            [k for k in entry["strong"] if k in search_text]
            + [k for k in entry["weak"] if k in search_text]
        )
        results.append((doc_type, score, strong_hits, matched))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _is_unambiguous(ranked: List[Tuple[str, int, int, List[str]]]) -> bool:
    """True only when top candidate has a strong hit and a clear margin."""
    if not ranked or ranked[0][1] <= 0:
        return False
    _, top_score, top_strong, _ = ranked[0]
    if top_strong == 0:
        return False
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    return (top_score - second_score) >= _AMBIGUITY_MARGIN


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

def _llm_classify(
    header: str,
    body_excerpt: str,
    top3: List[Tuple[str, int, int, List[str]]],
) -> Optional[_ClassificationResult]:
    taxonomy = get_taxonomy()
    allowed_types = list(taxonomy["doc_types"].keys())
    unknown_label = get_unknown_label()

    hints = "\n".join(
        f"  - {t} (score={s}, strong_hits={sh})"
        for t, s, sh, _ in top3[:3]
        if s > 0
    ) or "  (لا تطابق في الفحص الأولي)"

    prompt = f"""أنت محلل مستندات قانونية مصرية متخصص في التصنيف.

أمثلة:
مثال ١:
النص: "باسم الشعب - الدائرة المدنية - فلهذه الأسباب قضت المحكمة بإلزام المدعى عليه..."
المخرج: {{"doc_type": "حكم", "confidence": 95, "reasons": "يحتوي على عبارات باسم الشعب وفلهذه الأسباب"}}

مثال ٢:
النص: "صحيفة دعوى - المدعي: محمد أحمد - المدعى عليه: شركة النيل - الوقائع: ..."
المخرج: {{"doc_type": "صحيفة دعوى", "confidence": 92, "reasons": "يحتوي على بيانات المدعي والمدعى عليه والوقائع"}}

صنّف المستند التالي إلى أحد الأنواع المسموحة فقط.

الأنواع المسموحة:
{allowed_types}

تلميحات الفحص الأولي (قد تكون غير دقيقة):
{hints}

رأس المستند:
{header}

مقتطف من المحتوى:
{body_excerpt}

أرجع النوع الأدق من القائمة المسموحة فقط. إذا لم يطابق أي نوع، أرجع "{unknown_label}"."""

    try:
        llm = get_llm(_LLM_TIER, request_timeout=_LLM_TIMEOUT).with_structured_output(
            _ClassificationResult
        )
        return llm.invoke(prompt)
    except Exception as exc:
        logger.warning("LLM classifier error: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_document(text: str) -> Dict[str, Any]:
    """Classify an Egyptian civil document.

    Returns
    -------
    dict with keys: final_type, confidence, explanation
    """
    unknown = get_unknown_label()

    if not text or not text.strip():
        return {"final_type": unknown, "confidence": 0, "explanation": "Empty document text"}

    t0 = time.monotonic()
    header, body_excerpt = _prepare(text)
    search_text = header + "\n" + body_excerpt
    ranked = _score_candidates(search_text)

    if _is_unambiguous(ranked):
        top_type, top_score, _, matched = ranked[0]
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "classify path=heuristic type=%s confidence=%d latency_ms=%d",
            top_type, min(top_score, 100), elapsed_ms,
        )
        return {
            "final_type": top_type,
            "confidence": min(top_score, 100),
            "explanation": "تم التصنيف بناءً على الكلمات المفتاحية: " + "، ".join(matched),
        }

    # LLM path
    llm_result = _llm_classify(header, body_excerpt, ranked)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    valid_types = set(get_taxonomy()["doc_types"].keys()) | {unknown}

    if llm_result is not None and llm_result.doc_type in valid_types:
        logger.info(
            "classify path=llm type=%s confidence=%d latency_ms=%d",
            llm_result.doc_type, llm_result.confidence, elapsed_ms,
        )
        return {
            "final_type": llm_result.doc_type,
            "confidence": llm_result.confidence,
            "explanation": llm_result.reasons,
        }

    # LLM failed or returned an out-of-taxonomy label — fall back to heuristic top-1
    top_type = ranked[0][0] if ranked and ranked[0][1] > 0 else unknown
    top_score = ranked[0][1] if ranked else 0
    reason = (
        "invalid_type" if llm_result and llm_result.doc_type not in valid_types
        else "llm_failed"
    )
    logger.warning(
        "classify path=llm_fallback type=%s latency_ms=%d reason=%s",
        top_type, elapsed_ms, reason,
    )
    return {
        "final_type": top_type,
        "confidence": min(max(top_score, 0), 100),
        "explanation": "LLM classification unavailable; using heuristic fallback",
    }
