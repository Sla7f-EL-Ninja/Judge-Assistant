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

Content-length guard (added this version)
-----------------------------------------
The LLM is instructed never to add words, but models are non-deterministic
and occasionally hallucinate content.  In one observed run, page 20 grew
from 900 → 1,247 characters (+38 %) — fabricated text in a legal document
is unacceptable regardless of how rare it is.

After every LLM call, the refined text is validated against two thresholds:

    char ratio  = len(refined) / len(raw)   ≤ MAX_CHAR_RATIO  (1.15)
    word ratio  = words(refined) / words(raw) ≤ MAX_WORD_RATIO  (1.10)

If either limit is exceeded, the raw OCR text is returned unchanged and a
WARNING is logged.  Thresholds were calibrated against 10 observed
refinements across multiple runs — the highest legitimate ratio was 1.023
(page 9 diacritics).  A 15 % char / 10 % word margin gives comfortable
headroom for punctuation normalisation while catching the +38 % case.

Performance
-----------
- Model tier ``"low"`` (gemini-2.5-flash-lite) — no thinking budget.
  Typical call time: 2–5 s per page.
- LLM client is a module-level singleton.  Call ``prewarm_llm()`` at
  startup so the ~18 s init runs in the background during GCV processing.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-length guard thresholds
# ---------------------------------------------------------------------------
# Reject refined text whose length exceeds the raw text by more than these
# ratios.  Validated against observed data — all legitimate corrections
# stayed below 1.03 char ratio.  Set to > 2.0 to disable (not recommended).
MAX_CHAR_RATIO = 1.15   # max allowed  len(refined) / len(raw)
MAX_WORD_RATIO = 1.10   # max allowed  words(refined) / words(raw)

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

_PREAMBLE_RE = re.compile(
    r"^(النص المصحح|المخرج|الإجابة|الناتج)\s*[:\-]\s*",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Singleton LLM
# ---------------------------------------------------------------------------

_llm_instance = None
_llm_lock = threading.Lock()


def _get_llm(llm_tier: str = "low", timeout: int = 60):
    """Return (and cache) the singleton LLM used for OCR refinement."""
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            if _llm_instance is None:
                from config import get_llm  # noqa: PLC0415
                _llm_instance = get_llm(
                    llm_tier,
                    request_timeout=timeout,
                    temperature=0.0,
                )
                logger.info(
                    "LLM refinement client initialised (tier=%s, timeout=%ds)",
                    llm_tier, timeout,
                )
    return _llm_instance


def prewarm_llm(timeout: int = 60, llm_tier: str = "low") -> None:
    """Initialise the LLM singleton in a background thread.

    Call this at the start of ``run_ocr`` so the ~18 s initialisation
    overlaps with GCV chunk processing rather than blocking after it.
    Safe to call from any thread.
    """
    try:
        _get_llm(llm_tier=llm_tier, timeout=timeout)
        logger.debug("LLM singleton pre-warmed successfully")
    except Exception as exc:
        logger.warning("LLM pre-warm failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_ocr_text(
    raw_text: str,
    page_number: int = 1,
    timeout: int = 60,
    enabled: bool = True,
    llm_tier: str = "low",
) -> str:
    """Apply LLM refinement to a page of OCR output.

    Returns the refined text, or ``raw_text`` on any failure, empty
    response, or content-length guard rejection.
    """
    if not enabled:
        return raw_text
    if not raw_text or not raw_text.strip():
        return raw_text

    try:
        refined = _call_llm(raw_text, timeout=timeout, llm_tier=llm_tier)
    except Exception as exc:
        logger.warning(
            "LLM refinement failed for page %d (%s: %s) — using raw OCR text",
            page_number, type(exc).__name__, exc,
        )
        return raw_text

    if not refined or not refined.strip():
        logger.warning(
            "LLM returned empty output for page %d — using raw OCR text",
            page_number,
        )
        return raw_text

    # ---- Content-length guard --------------------------------------------
    # Protects against hallucinations where the model adds content despite
    # the system prompt forbidding it.  Observed case: page 20 grew 900 →
    # 1,247 chars (+38 %) in a single run.  Raw text is safer in that case.
    raw_chars  = max(len(raw_text), 1)
    raw_words  = max(len(raw_text.split()), 1)
    ref_chars  = len(refined)
    ref_words  = len(refined.split())
    char_ratio = ref_chars / raw_chars
    word_ratio = ref_words / raw_words

    if char_ratio > MAX_CHAR_RATIO or word_ratio > MAX_WORD_RATIO:
        logger.warning(
            "Page %d: LLM output rejected — "
            "char ratio=%.2f (limit %.2f), word ratio=%.2f (limit %.2f) — "
            "likely hallucination; falling back to raw OCR text",
            page_number,
            char_ratio, MAX_CHAR_RATIO,
            word_ratio, MAX_WORD_RATIO,
        )
        return raw_text

    logger.info(
        "LLM refinement complete — page %d: %d chars → %d chars",
        page_number, len(raw_text), len(refined),
    )
    return refined


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _call_llm(raw_text: str, timeout: int, llm_tier: str = "low") -> str:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    llm = _get_llm(llm_tier=llm_tier, timeout=timeout)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"النص الأصلي:\n{raw_text}"),
    ]
    response = llm.invoke(messages)
    text: str = (
        response.content if hasattr(response, "content") else str(response)
    )
    return _PREAMBLE_RE.sub("", text.strip()).strip()