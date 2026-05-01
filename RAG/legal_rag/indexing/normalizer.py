# """
# normalizer.py
# -------------
# Arabic text normalization for the Civil Law RAG pipeline.

# Normalizing both the corpus (at ingest) and queries (at retrieval time)
# ensures that the same legal term always maps to the same embedding
# regardless of spelling variants.

# Normalizations applied:
# 1. Strip tashkeel (diacritics / harakat)
# 2. Unify hamza forms  (أ إ آ ء → ا / ء)
# 3. Normalize alif maqsura (ى → ي)
# 4. Normalize ta marbuta (ة → ه) — optional, disabled by default for legal text
# 5. Normalize Western digits → Arabic-Indic (١٢٣ → 123) — optional
# 6. Collapse multiple whitespace to single space

# Usage::

#     from RAG.legal_rag.indexing.normalizer import normalize

#     normalized = normalize("مُوجِبَاتُ الإلتِزَام")
#     # → "موجبات الالتزام"
# """

# from __future__ import annotations

# import re
# import unicodedata

# # ---------------------------------------------------------------------------
# # Unicode ranges for Arabic
# # ---------------------------------------------------------------------------
# _TASHKEEL = re.compile(
#     "["
#     "\u0610-\u061A"   # Arabic sign + sajda etc.
#     "\u064B-\u065F"   # harakat (fathatan … sukun)
#     "\u0670"          # superscript alef
#     "\u06D6-\u06DC"   # small high letters
#     "\u06DF-\u06E4"
#     "\u06E7-\u06E8"
#     "\u06EA-\u06ED"
#     "]"
# )

# # Hamza on alef → bare alef; hamza above / below → ء; alif wasla → alef
# _HAMZA_MAP = str.maketrans({
#     "\u0623": "\u0627",  # أ → ا
#     "\u0625": "\u0627",  # إ → ا
#     "\u0622": "\u0627",  # آ → ا
#     "\u0671": "\u0627",  # ٱ (alif wasla) → ا
# })

# # Alif maqsura → ya
# _ALF_MAQSURA = str.maketrans({"\u0649": "\u064A"})  # ى → ي


# def normalize(text: str, normalize_ta_marbuta: bool = False) -> str:
#     """Return *text* after applying all Arabic normalizations.

#     Args:
#         text:                 Input Arabic string.
#         normalize_ta_marbuta: If True, replace ة with ه.
#                               Disabled by default for legal text where
#                               ta marbuta carries grammatical meaning.

#     Returns:
#         Normalized string.
#     """
#     if not text:
#         return text

#     # 1. Strip diacritics
#     text = _TASHKEEL.sub("", text)

#     # 2. Unify hamza forms
#     text = text.translate(_HAMZA_MAP)

#     # 3. Alif maqsura → ya
#     text = text.translate(_ALF_MAQSURA)

#     # 4. Optional: ta marbuta
#     if normalize_ta_marbuta:
#         text = text.replace("\u0629", "\u0647")  # ة → ه

#     # 5. Unicode NFC (compose canonical forms)
#     text = unicodedata.normalize("NFC", text)

#     # 6. Collapse whitespace
#     text = re.sub(r"\s+", " ", text).strip()

#     return text


"""
normalizer.py
-------------
Arabic text normalization for the Civil Law RAG pipeline.

Normalizing both the corpus (at ingest) and queries (at retrieval time)
ensures that the same legal term always maps to the same embedding
regardless of spelling variants.

Normalizations applied:
1. Strip tashkeel (diacritics / harakat)
2. Unify hamza forms  (أ إ آ ء → ا / ء)
3. Normalize alif maqsura (ى → ي)
4. Normalize ta marbuta (ة → ه) — optional, disabled by default for legal text
5. Normalize Western digits → Arabic-Indic (١٢٣ → 123) — optional
6. Collapse multiple whitespace to single space

Usage::

    from RAG.legal_rag.indexing.normalizer import normalize

    normalized = normalize("مُوجِبَاتُ الإلتِزَام")
    # → "موجبات الالتزام"
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Unicode ranges for Arabic
# ---------------------------------------------------------------------------
_TASHKEEL = re.compile(
    "["
    "\u0610-\u061A"   # Arabic sign + sajda etc.
    "\u064B-\u065F"   # harakat (fathatan … sukun)
    "\u0670"          # superscript alef
    "\u06D6-\u06DC"   # small high letters
    "\u06DF-\u06E4"
    "\u06E7-\u06E8"
    "\u06EA-\u06ED"
    "]"
)

# Hamza on alef → bare alef; hamza above / below → ء; alif wasla → alef
_HAMZA_MAP = str.maketrans({
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0622": "\u0627",  # آ → ا
    "\u0671": "\u0627",  # ٱ (alif wasla) → ا
})

# Alif maqsura → ya
_ALF_MAQSURA = str.maketrans({"\u0649": "\u064A"})  # ى → ي

# Arabic-Indic → Western digits
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic digits to Western digits only.

    Use this when you need digit conversion without full normalization,
    e.g. inside the splitter when preserving Arabic text structure.
    """
    return text.translate(_ARABIC_INDIC)


def normalize(text: str, normalize_ta_marbuta: bool = False) -> str:
    """Return *text* after applying all Arabic normalizations.

    Args:
        text:                 Input Arabic string.
        normalize_ta_marbuta: If True, replace ة with ه.
                              Disabled by default for legal text where
                              ta marbuta carries grammatical meaning.

    Returns:
        Normalized string.
    """
    if not text:
        return text

    # 1. Strip diacritics
    text = _TASHKEEL.sub("", text)

    # 2. Unify hamza forms
    text = text.translate(_HAMZA_MAP)

    # 3. Alif maqsura → ya
    text = text.translate(_ALF_MAQSURA)

    # 4. Optional: ta marbuta
    if normalize_ta_marbuta:
        text = text.replace("\u0629", "\u0647")  # ة → ه

    # 5. Unicode NFC (compose canonical forms)
    text = unicodedata.normalize("NFC", text)

    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text