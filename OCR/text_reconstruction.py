"""
OCR.text_reconstruction
-----------------------
Text post-processing: numeral normalization and optional HTML stripping.
"""

from __future__ import annotations

import re

# Translation tables for numeral normalization
_ARABIC_INDIC_TABLE = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
    "0123456789",
)

_PERSIAN_TABLE = str.maketrans(
    "\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9",
    "0123456789",
)

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def normalize_numerals(text: str) -> str:
    """Normalize Arabic-Indic and Persian numerals to Western (ASCII) digits.

    No spell-checking or text correction is performed.
    """
    text = text.translate(_ARABIC_INDIC_TABLE)
    text = text.translate(_PERSIAN_TABLE)
    return text


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    Useful for downstream consumers that do not want QARI's HTML output.
    """
    return _HTML_TAG_PATTERN.sub("", text)
