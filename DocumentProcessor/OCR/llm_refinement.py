"""
DocumentProcessor.OCR.llm_refinement
--------------------------------------
LLM post-processing layer for Arabic OCR output.

Responsibilities
----------------
1. Fix character-level OCR confusions common in Arabic script
   (ر/ز، ط/ظ، ه/ة، و/ؤ، ي/ئ، ا/أ/إ/آ … etc.)
2. Insert correct Classical Arabic punctuation (، ؛ ؟ : . " ")
3. Fix spacing anomalies introduced by OCR segmentation

Hard constraints enforced via system prompt
-------------------------------------------
- No words may be added or removed.
- Legal terminology, party names, dates, and numbers must not change.
- The model returns only the corrected text — no commentary.

Graceful degradation
--------------------
Any LLM failure (timeout, quota, network) returns ``raw_text`` unchanged
so the OCR pipeline always produces usable output.  Failures are logged
at WARNING level with enough context for post-mortem analysis.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
أنت محرر نصوص قانونية متخصص في اللغة العربية الفصحى الكلاسيكية.
مهمتك الوحيدة: تصحيح مخرجات OCR الخاصة بالمستندات القانونية المصرية.

القواعد الصارمة التي لا استثناء فيها:
١. لا تُضف أي كلمة لم تكن موجودة في النص الأصلي.
٢. لا تحذف أي كلمة من النص الأصلي.
٣. لا تغيّر المصطلحات القانونية أو أسماء الأطراف أو التواريخ أو الأرقام.
٤. صحّح فقط ما يلي:
   - الأخطاء الحرفية الناتجة عن OCR
     أمثلة: (ر ↔ ز)، (ط ↔ ظ)، (ه ↔ ة)، (و ↔ ؤ)، (ي ↔ ئ)، (ا ↔ أ ↔ إ ↔ آ)
   - علامات الترقيم العربية الناقصة أو الخاطئة: ، ؛ ؟ : . " "
   - المسافات الزائدة أو الناقصة بين الكلمات.
٥. أعد النص المصحح فقط — بدون أي مقدمة أو تعليق أو تفسير.\
"""

# Strip any preamble the model may produce despite the prompt (defensive)
_PREAMBLE_RE = re.compile(
    r"^(النص المصحح|المخرج|الإجابة|الناتج)\s*[:\-]\s*",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_ocr_text(
    raw_text: str,
    page_number: int = 1,
    timeout: int = 60,
    enabled: bool = True,
) -> str:
    """Apply LLM refinement to a page of OCR output.

    Parameters
    ----------
    raw_text:
        Text as produced by the OCR engine (after numeral normalisation).
    page_number:
        Used for log messages only.
    timeout:
        LLM call timeout in seconds.  Passed to ``get_llm`` via
        ``request_timeout`` override.
    enabled:
        Master switch.  When ``False`` the function is a no-op and returns
        ``raw_text`` immediately (zero overhead).

    Returns
    -------
    str
        LLM-refined text, or ``raw_text`` on any failure/bypass.
    """
    if not enabled:
        return raw_text

    if not raw_text or not raw_text.strip():
        return raw_text

    try:
        refined = _call_llm(raw_text, timeout=timeout)
    except Exception as exc:
        logger.warning(
            "LLM refinement failed for page %d (%s: %s) — using raw OCR text",
            page_number,
            type(exc).__name__,
            exc,
        )
        return raw_text

    if not refined or not refined.strip():
        logger.warning(
            "LLM refinement returned empty output for page %d — using raw OCR text",
            page_number,
        )
        return raw_text

    logger.info(
        "LLM refinement complete — page %d: %d chars → %d chars",
        page_number,
        len(raw_text),
        len(refined),
    )
    return refined


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _call_llm(raw_text: str, timeout: int) -> str:
    """Invoke the high-tier LLM and return the stripped response text."""
    from config import get_llm  # noqa: PLC0415 — deferred to avoid circular import
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    llm = get_llm("high", request_timeout=timeout, temperature=0.0)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"النص الأصلي:\n{raw_text}"),
    ]

    response = llm.invoke(messages)

    # LangChain chat models return an AIMessage; plain string otherwise
    text: str = (
        response.content  # type: ignore[attr-defined]
        if hasattr(response, "content")
        else str(response)
    )

    # Strip any preamble the model added despite the prompt
    text = _PREAMBLE_RE.sub("", text.strip()).strip()
    return text
