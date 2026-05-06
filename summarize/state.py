"""
State definition and defendant disambiguation for the summarization pipeline.
"""

from typing import TypedDict, List

from summarize.utils import get_logger, normalize_arabic_for_matching

logger = get_logger("hakim.state")


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class SummarizationState(TypedDict):
    """Shared state flowing through the summarization pipeline."""
    documents: List[dict]            # [{"raw_text": "...", "doc_id": "..."}]
    chunks: List[dict]               # NormalizedChunk dicts
    classified_chunks: List[dict]    # ClassifiedChunk dicts
    bullets: List[dict]              # LegalBullet dicts
    role_aggregations: List[dict]    # RoleAggregation dicts
    themed_roles: List[dict]         # ThemedRole dicts
    role_theme_summaries: List[dict] # RoleThemeSummaries dicts
    case_brief: dict                 # CaseBrief dict
    all_sources: List[str]           # Unique citations
    rendered_brief: str              # Arabic markdown
    party_manifest: dict             # {party: [doc_types]} built in node_0_intake


# ---------------------------------------------------------------------------
# Defendant disambiguation
# ---------------------------------------------------------------------------

# Arabic ordinal pairs: (masculine, feminine) for ranks 1-10.
_ARABIC_ORDINALS = {
    1:  ("الأول",   "الأولى"),
    2:  ("الثاني",  "الثانية"),
    3:  ("الثالث",  "الثالثة"),
    4:  ("الرابع",  "الرابعة"),
    5:  ("الخامس",  "الخامسة"),
    6:  ("السادس",  "السادسة"),
    7:  ("السابع",  "السابعة"),
    8:  ("الثامن",  "الثامنة"),
    9:  ("التاسع",  "التاسعة"),
    10: ("العاشر",  "العاشرة"),
}

# Ordered list of normalized ordinal signals to detect rank from a doc_id.
_ORDINAL_SIGNALS = [
    ("العاشر",    10), ("العاشرة",   10),
    ("التاسع",     9), ("التاسعة",    9),
    ("الثامن",     8), ("الثامنة",    8),
    ("السابع",     7), ("السابعة",    7),
    ("السادس",     6), ("السادسة",    6),
    ("الخامس",     5), ("الخامسة",    5),
    ("الرابع",     4), ("الرابعة",    4),
    ("الثالث",     3), ("الثالثة",    3),
    ("الثاني",     2), ("الثانية",    2),
    ("الاول",      1), ("الاولي",     1),
    ("10", 10), ("9", 9), ("8", 8), ("7", 7), ("6", 6),
    ("5",   5), ("4", 4), ("3", 3), ("2", 2), ("1", 1),
]

_DEFENSE_DOC_TYPES = {"مذكرة دفاع", "مذكرة رد"}


def _get_ordinal(n: int, feminine: bool) -> str:
    """Return Arabic ordinal word for rank n with correct gender."""
    if n in _ARABIC_ORDINALS:
        return _ARABIC_ORDINALS[n][1 if feminine else 0]
    return f"رقم {n}"


def _detect_rank(doc_id: str) -> int | None:
    """Try to extract an ordinal rank from a doc_id string."""
    norm = normalize_arabic_for_matching(doc_id)
    for signal, rank in _ORDINAL_SIGNALS:
        if signal in norm:
            return rank
    return None


def disambiguate_defendants(all_chunks: list) -> list:
    """Post-process chunks to add ordinal suffixes to defendant party labels.

    Only affects chunks where:
      - party == 'المدعى عليه'  (base value assigned by node 0)
      - doc_type in {'مذكرة دفاع', 'مذكرة رد'}  (defense documents only)
      - AND there are >= 2 distinct doc_ids in that filtered set

    Returns new chunk dicts (no mutation of inputs).
    """
    defense_indices = [
        i for i, c in enumerate(all_chunks)
        if c.get("party") == "المدعى عليه"
        and c.get("doc_type") in _DEFENSE_DOC_TYPES
    ]

    seen: dict = {}
    for i in defense_indices:
        did = all_chunks[i].get("doc_id", "")
        if did not in seen:
            seen[did] = i

    distinct_doc_ids = list(seen.keys())

    if len(distinct_doc_ids) <= 1:
        return all_chunks

    logger.info(
        "  disambiguate_defendants: %d defense doc(s) detected → %s",
        len(distinct_doc_ids), distinct_doc_ids,
    )

    # Assign ranks to each doc_id
    claimed: dict = {}
    unranked: list = []

    for did in distinct_doc_ids:
        rank = _detect_rank(did)
        if rank is not None and rank not in claimed:
            claimed[rank] = did
        elif rank is not None and rank in claimed:
            logger.warning(
                "  Rank %d claimed by both '%s' and '%s' — treating latter as unranked.",
                rank, claimed[rank], did,
            )
            unranked.append(did)
        else:
            unranked.append(did)

    rank_counter = 1
    doc_rank: dict = {did: rank for rank, did in claimed.items()}
    for did in unranked:
        while rank_counter in claimed:
            rank_counter += 1
        doc_rank[did] = rank_counter
        claimed[rank_counter] = did
        rank_counter += 1

    # Detect gender and build party string per doc_id
    doc_party: dict = {}
    for did in distinct_doc_ids:
        feminine = "عليها" in did
        rank = doc_rank[did]
        base = "المدعى عليها" if feminine else "المدعى عليه"
        doc_party[did] = f"{base} {_get_ordinal(rank, feminine)}"
        logger.info("  '%s' → '%s'", did, doc_party[did])

    affected = set(defense_indices)

    result = []
    for i, chunk in enumerate(all_chunks):
        if i in affected:
            did = chunk.get("doc_id", "")
            new_party = doc_party.get(did)
            if new_party:
                result.append({**chunk, "party": new_party})
                continue
        result.append(chunk)
    return result
