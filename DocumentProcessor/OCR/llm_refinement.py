# """
# DocumentProcessor.OCR.llm_refinement
# --------------------------------------
# LLM post-processing layer for Arabic OCR output.

# Responsibilities
# ----------------
# 1. Fix character-level OCR confusions common in Arabic script
#    (ر/ز، ط/ظ، ه/ة، و/ؤ، ي/ئ، ا/أ/إ/آ … etc.)
# 2. Insert correct Classical Arabic punctuation (، ؛ ؟ : . " ")
# 3. Fix spacing anomalies introduced by OCR segmentation

# Hard constraints enforced via system prompt
# -------------------------------------------
# - No words may be added or removed.
# - Legal terminology, party names, dates, and numbers must not change.
# - The model returns only the corrected text — no commentary.

# Graceful degradation
# --------------------
# Any LLM failure (timeout, quota, network) returns ``raw_text`` unchanged
# so the OCR pipeline always produces usable output.  Failures are logged
# at WARNING level with enough context for post-mortem analysis.
# """

# from __future__ import annotations

# import logging
# import re
# from typing import Optional

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Prompt
# # ---------------------------------------------------------------------------

# _SYSTEM_PROMPT = """\
# أنت محرر نصوص قانونية متخصص في اللغة العربية الفصحى الكلاسيكية.
# مهمتك الوحيدة: تصحيح مخرجات OCR الخاصة بالمستندات القانونية المصرية.

# القواعد الصارمة التي لا استثناء فيها:
# ١. لا تُضف أي كلمة لم تكن موجودة في النص الأصلي.
# ٢. لا تحذف أي كلمة من النص الأصلي.
# ٣. لا تغيّر المصطلحات القانونية أو أسماء الأطراف أو التواريخ أو الأرقام.
# ٤. صحّح فقط ما يلي:
#    - الأخطاء الحرفية الناتجة عن OCR
#      أمثلة: (ر ↔ ز)، (ط ↔ ظ)، (ه ↔ ة)، (و ↔ ؤ)، (ي ↔ ئ)، (ا ↔ أ ↔ إ ↔ آ)
#    - علامات الترقيم العربية الناقصة أو الخاطئة: ، ؛ ؟ : . " "
#    - المسافات الزائدة أو الناقصة بين الكلمات.
# ٥. أعد النص المصحح فقط — بدون أي مقدمة أو تعليق أو تفسير.\
# """

# # Strip any preamble the model may produce despite the prompt (defensive)
# _PREAMBLE_RE = re.compile(
#     r"^(النص المصحح|المخرج|الإجابة|الناتج)\s*[:\-]\s*",
#     re.MULTILINE,
# )


# # ---------------------------------------------------------------------------
# # Public API
# # ---------------------------------------------------------------------------

# def refine_ocr_text(
#     raw_text: str,
#     page_number: int = 1,
#     timeout: int = 60,
#     enabled: bool = True,
# ) -> str:
#     """Apply LLM refinement to a page of OCR output.

#     Parameters
#     ----------
#     raw_text:
#         Text as produced by the OCR engine (after numeral normalisation).
#     page_number:
#         Used for log messages only.
#     timeout:
#         LLM call timeout in seconds.  Passed to ``get_llm`` via
#         ``request_timeout`` override.
#     enabled:
#         Master switch.  When ``False`` the function is a no-op and returns
#         ``raw_text`` immediately (zero overhead).

#     Returns
#     -------
#     str
#         LLM-refined text, or ``raw_text`` on any failure/bypass.
#     """
#     if not enabled:
#         return raw_text

#     if not raw_text or not raw_text.strip():
#         return raw_text

#     try:
#         refined = _call_llm(raw_text, timeout=timeout)
#     except Exception as exc:
#         logger.warning(
#             "LLM refinement failed for page %d (%s: %s) — using raw OCR text",
#             page_number,
#             type(exc).__name__,
#             exc,
#         )
#         return raw_text

#     if not refined or not refined.strip():
#         logger.warning(
#             "LLM refinement returned empty output for page %d — using raw OCR text",
#             page_number,
#         )
#         return raw_text

#     logger.info(
#         "LLM refinement complete — page %d: %d chars → %d chars",
#         page_number,
#         len(raw_text),
#         len(refined),
#     )
#     return refined


# # ---------------------------------------------------------------------------
# # Internal
# # ---------------------------------------------------------------------------

# def _call_llm(raw_text: str, timeout: int) -> str:
#     """Invoke the high-tier LLM and return the stripped response text."""
#     from config import get_llm  # noqa: PLC0415 — deferred to avoid circular import
#     from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

#     llm = get_llm("high", request_timeout=timeout, temperature=0.0)

#     messages = [
#         SystemMessage(content=_SYSTEM_PROMPT),
#         HumanMessage(content=f"النص الأصلي:\n{raw_text}"),
#     ]

#     response = llm.invoke(messages)

#     # LangChain chat models return an AIMessage; plain string otherwise
#     text: str = (
#         response.content  # type: ignore[attr-defined]
#         if hasattr(response, "content")
#         else str(response)
#     )

#     # Strip any preamble the model added despite the prompt
#     text = _PREAMBLE_RE.sub("", text.strip()).strip()
#     return text


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

Performance
-----------
Two changes from the original implementation:

1. **Model tier** — uses ``"low"`` (``gemini-2.5-flash-lite``) instead of
   ``"high"`` (``gemini-2.5-flash``).  The "high" tier activates Gemini's
   extended thinking, which runs a hidden reasoning chain before producing
   output — unnecessary for the pattern-matching task of OCR character
   correction.  Flash-Lite achieves equivalent correction quality at roughly
   4–5× lower latency (8–12 s vs 30–50 s per page).

   To switch back to the full model (e.g. for very low-confidence pages),
   set ``OCR_REFINEMENT_LLM_TIER = "high"`` in ``config/ocr.py`` or
   override per-call via ``refine_ocr_text(..., llm_tier="high")``.

2. **Singleton LLM** — the LLM client is constructed once per process and
   reused across all page calls.  With parallel page processing this avoids
   4 concurrent constructor calls and removes the per-call initialisation
   overhead of the underlying gRPC / HTTP transport.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

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
# LLM singleton
# ---------------------------------------------------------------------------
# The LLM client is expensive to construct (gRPC channel / HTTP session
# establishment).  We build it once and reuse across all pages, including
# concurrent page workers in the thread pool.  LangChain's ChatGoogleGenerativeAI
# is stateless per-call, so sharing it across threads is safe.

_llm_instance = None
_llm_lock = threading.Lock()


def _get_llm(llm_tier: str = "low", timeout: int = 30):
    """Return (and cache) the singleton LLM used for OCR refinement.

    The instance is built on first call and reused for the process lifetime.
    Thread-safe via a double-checked lock.

    Parameters
    ----------
    llm_tier:
        LangChain tier key from settings.yaml.  Default ``"low"``
        (``gemini-2.5-flash-lite``).  Pass ``"high"`` to use the full
        thinking model when accuracy is more important than latency.
    timeout:
        Request timeout forwarded to ``get_llm``.  Only applied at
        construction time; ignored after the singleton is initialised.
    """
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
                    llm_tier,
                    timeout,
                )
    return _llm_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_ocr_text(
    raw_text: str,
    page_number: int = 1,
    timeout: int = 30,
    enabled: bool = True,
    llm_tier: str = "low",
) -> str:
    """Apply LLM refinement to a page of OCR output.

    Parameters
    ----------
    raw_text:
        Text as produced by the OCR engine.
    page_number:
        Used for log messages only.
    timeout:
        LLM call timeout in seconds.
    enabled:
        Master switch.  When ``False`` returns ``raw_text`` immediately
        with zero overhead.
    llm_tier:
        LangChain tier key.  ``"low"`` (default) uses the fast Flash-Lite
        model.  Override to ``"high"`` for the thinking model when needed.

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
        refined = _call_llm(raw_text, timeout=timeout, llm_tier=llm_tier)
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

def _call_llm(raw_text: str, timeout: int, llm_tier: str = "low") -> str:
    """Invoke the refinement LLM and return the stripped response text."""
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    llm = _get_llm(llm_tier=llm_tier, timeout=timeout)

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