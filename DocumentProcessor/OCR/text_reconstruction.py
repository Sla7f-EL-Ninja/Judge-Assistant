"""
DocumentProcessor.OCR.text_reconstruction
------------------------------------------
Text post-processing: numeral normalization and optional HTML stripping.
"""

from __future__ import annotations

import re

# Translation tables for numeral normalization
_ARABIC_INDIC_TABLE = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789",
)

_PERSIAN_TABLE = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹",
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
