"""
config.taxonomy
---------------
Loads and caches the document taxonomy from document_taxonomy.yaml.
Keywords are Arabic-normalized at load time so runtime matching is consistent.

Public API:
    get_taxonomy()     -> {"doc_types": {name: {strong, weak, anti}}, "unknown_label": str}
    get_doc_types()    -> list of valid type name strings
    get_unknown_label() -> the unknown sentinel string
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

_TAXONOMY_YAML = Path(__file__).resolve().parent / "document_taxonomy.yaml"

# ---------------------------------------------------------------------------
# Inline normalization (avoids circular import with DocumentProcessor)
# Mirrors DocumentProcessor.arabic_norm.normalize — keep in sync.
# ---------------------------------------------------------------------------

_ALEF_FORMS = re.compile(r"[أإآٱ]")
_ALEF_MAQSURA = re.compile(r"ى")
_TATWEEL = re.compile(r"ـ")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _TATWEEL.sub("", text)
    text = _ALEF_FORMS.sub("ا", text)
    text = _ALEF_MAQSURA.sub("ي", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_taxonomy() -> Dict[str, Any]:
    """Load taxonomy YAML and return normalized keyword dict (cached)."""
    with open(_TAXONOMY_YAML, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    normalized: Dict[str, Any] = {}
    for dtype, entry in raw.get("doc_types", {}).items():
        normalized[dtype] = {
            "strong": [_normalize(k) for k in entry.get("strong", [])],
            "weak":   [_normalize(k) for k in entry.get("weak",   [])],
            "anti":   [_normalize(k) for k in entry.get("anti",   [])],
        }

    return {
        "doc_types":     normalized,
        "unknown_label": raw.get("unknown_label", "مستند غير معروف"),
    }


def get_doc_types() -> List[str]:
    return list(get_taxonomy()["doc_types"].keys())


def get_unknown_label() -> str:
    return get_taxonomy()["unknown_label"]
