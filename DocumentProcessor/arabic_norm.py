"""
DocumentProcessor.arabic_norm
------------------------------
Arabic text normalization for keyword matching.

Transforms:
  - NFKC unicode normalization
  - Strip tatweel (kashida)
  - Unify alef forms (أإآٱ → ا)
  - Unify alef maqsura (ى → ي)
  - Collapse whitespace

Note: ta marbuta (ة) is intentionally NOT merged with ha (ه) because
several classification keywords rely on the distinction.
"""

from __future__ import annotations

import re
import unicodedata

_ALEF_FORMS = re.compile(r"[أإآٱ]")
_ALEF_MAQSURA = re.compile(r"ى")
_TATWEEL = re.compile(r"ـ")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Return normalized Arabic text suitable for keyword matching."""
    text = unicodedata.normalize("NFKC", text)
    text = _TATWEEL.sub("", text)
    text = _ALEF_FORMS.sub("ا", text)
    text = _ALEF_MAQSURA.sub("ي", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
