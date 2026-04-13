"""
graph.py

Defines and constructs the LangGraph workflow for the Summarization pipeline.

The pipeline processes legal case documents through 7 nodes:
  Node 0: Document Intake  — clean, extract metadata, segment into chunks
  Node 1: Role Classification — classify each chunk by legal role
  Node 2: Bullet Extraction — extract atomic legal ideas from each chunk
  Node 3: Aggregation — group bullets into agreed/disputed/party-specific per role
  Node 4A: Thematic Clustering — organize items into themes within each role
  Node 4B: Theme Synthesis — produce 2-3 paragraph summaries per theme
  Node 5: Case Brief Generation — produce a 7-section judge-facing brief

Node 0 processes documents in parallel (S1-1).
Node 4B synthesizes themes in parallel within each role (S1-2).
All nodes are bound via closures — no global registry (S1-5).
"""

import sys
import os
import concurrent.futures
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END

# S1-10: Only add the actual Summerize directory — NOT nonexistent "Node X" subdirs
_summerize_dir = os.path.dirname(os.path.abspath(__file__))
if _summerize_dir not in sys.path:
    sys.path.insert(0, _summerize_dir)

from node_0 import Node0_DocumentIntake
from node_1 import Node1_RoleClassifier
from node_2 import Node2_BulletExtractor
from node_3 import Node3_Aggregator
from node_4a import Node4A_ThematicClustering
from node_4b import Node4B_ThemeSynthesis
from node_5 import Node5_BriefGenerator
from utils import get_logger, normalize_arabic_for_matching

logger = get_logger("hakim.graph")


# ---------------------------------------------------------------------------
# Defendant disambiguation
# ---------------------------------------------------------------------------

# Arabic ordinal pairs: (masculine, feminine) for ranks 1-10.
# For n > 10, _get_ordinal() falls back to 'رقم {n}'.
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
# Each tuple: (normalized_signal, rank). Checked in order — more specific first.
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
    # "الاول" is the normalized form of "الأول" (Hamza stripped)
    ("الاول",      1), ("الاولي",     1),
    # digit fallback
    ("10", 10), ("9", 9), ("8", 8), ("7", 7), ("6", 6),
    ("5",   5), ("4", 4), ("3", 3), ("2", 2), ("1", 1),
]

_DEFENSE_DOC_TYPES = {"مذكرة دفاع", "مذكرة رد"}


def _get_ordinal(n: int, feminine: bool) -> str:
    """Return Arabic ordinal word for rank n with correct gender.
    Falls back to 'رقم {n}' for n > 10."""
    if n in _ARABIC_ORDINALS:
        return _ARABIC_ORDINALS[n][1 if feminine else 0]
    return f"رقم {n}"


def _detect_rank(doc_id: str) -> int | None:
    """Try to extract an ordinal rank from a doc_id string.
    Returns the integer rank or None if no signal found."""
    norm = normalize_arabic_for_matching(doc_id)
    for signal, rank in _ORDINAL_SIGNALS:
        if signal in norm:
            return rank
    return None


def _disambiguate_defendants(all_chunks: list) -> list:
    """Post-process chunks to add ordinal suffixes to defendant party labels.

    Only affects chunks where:
      - party == 'المدعى عليه'  (base value assigned by node 0)
      - doc_type in {'مذكرة دفاع', 'مذكرة رد'}  (defense documents only)
      - AND there are >= 2 distinct doc_ids in that filtered set

    All other chunks (hearing minutes, expert reports, plaintiff docs, etc.)
    are returned unchanged.

    Gender detection: 'عليها' in the original doc_id → feminine suffix.
    Rank detection:   ordinal signal in normalized doc_id → explicit rank;
                      no signal → assigned by first-seen discovery order,
                      filling smallest unused rank slot.

    Returns new chunk dicts (no mutation of inputs).
    """
    # Step 1: Identify defense chunks from multiple defendants
    defense_indices = [
        i for i, c in enumerate(all_chunks)
        if c.get("party") == "المدعى عليه"
        and c.get("doc_type") in _DEFENSE_DOC_TYPES
    ]

    # Collect distinct doc_ids preserving first-seen order
    seen: dict = {}
    for i in defense_indices:
        did = all_chunks[i].get("doc_id", "")
        if did not in seen:
            seen[did] = i  # store first-seen index for ordering

    distinct_doc_ids = list(seen.keys())

    if len(distinct_doc_ids) <= 1:
        # Single defendant (or none) — nothing to do
        return all_chunks

    logger.info(
        "  _disambiguate_defendants: %d defense doc(s) detected → %s",
        len(distinct_doc_ids), distinct_doc_ids,
    )

    # Step 2: Assign ranks to each doc_id
    # First pass: doc_ids with explicit ordinal signals
    claimed: dict = {}   # rank -> doc_id
    unranked: list = []  # doc_ids with no signal, in discovery order

    for did in distinct_doc_ids:
        rank = _detect_rank(did)
        if rank is not None and rank not in claimed:
            claimed[rank] = did
        elif rank is not None and rank in claimed:
            # Collision: two doc_ids claim the same rank → treat this one as unranked
            logger.warning(
                "  Rank %d claimed by both '%s' and '%s' — treating latter as unranked.",
                rank, claimed[rank], did,
            )
            unranked.append(did)
        else:
            unranked.append(did)

    # Second pass: fill unranked by smallest unused rank
    rank_counter = 1
    doc_rank: dict = {did: rank for rank, did in claimed.items()}
    for did in unranked:
        while rank_counter in claimed:
            rank_counter += 1
        doc_rank[did] = rank_counter
        claimed[rank_counter] = did
        rank_counter += 1

    # Step 3: Detect gender and build party string per doc_id
    doc_party: dict = {}
    for did in distinct_doc_ids:
        feminine = "عليها" in did
        rank = doc_rank[did]
        base = "المدعى عليها" if feminine else "المدعى عليه"
        doc_party[did] = f"{base} {_get_ordinal(rank, feminine)}"
        logger.info("  '%s' → '%s'", did, doc_party[did])

    # Step 4: Build index set of affected chunk indices for fast lookup
    affected = set(defense_indices)

    # Step 5: Return new chunk list
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


# ---------------------------------------------------------------------------
# State definition
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
    party_manifest: dict             # Fix 1/7: {party: [doc_types]} built in node_0_intake


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def create_pipeline(llm_or_config):
    """Initialise all nodes and return a compiled LangGraph pipeline.

    Args:
        llm_or_config:
            • A single LLM object — used for all nodes.
            • A dict with optional keys ``"high"`` and ``"low"``:
                - ``"high"`` tier: Nodes 2, 3, 4B, 5 (complex synthesis/reasoning)
                - ``"low"``  tier: Nodes 0, 1, 4A  (classification/clustering)
              If only one key is present it is used for all nodes.

    S1-3: Multi-tier LLM support.
    S1-5: Nodes are bound via closures; no global registry.

    Returns:
        A compiled LangGraph application ready for ``app.invoke(initial_state)``.
    """
    # --- Resolve tiers (S1-3) ---
    if isinstance(llm_or_config, dict):
        llm_high = (
            llm_or_config.get("high")
            or next(iter(llm_or_config.values()))
        )
        llm_low = llm_or_config.get("low", llm_high)
    else:
        llm_high = llm_or_config
        llm_low = llm_or_config

    # --- Instantiate nodes locally (S1-5: no global dict) ---
    _node_0 = Node0_DocumentIntake(llm_low)
    _node_1 = Node1_RoleClassifier(llm_low)
    _node_2 = Node2_BulletExtractor(llm_high)
    _node_3 = Node3_Aggregator(llm_high)
    _node_4a = Node4A_ThematicClustering(llm_low)
    _node_4b = Node4B_ThemeSynthesis(llm_high)
    _node_5 = Node5_BriefGenerator(llm_high)

    logger.info("All pipeline nodes initialised.")

    # -----------------------------------------------------------------------
    # Closure-based wrapper functions
    # Each function closes over the relevant node instance above.
    # -----------------------------------------------------------------------

    def node_0_intake(state: SummarizationState) -> dict:
        """Node 0: Process each document (clean, metadata, segment).

        S1-1: Documents are processed in parallel using a thread pool.
        """
        documents = state.get("documents", [])
        if not documents:
            logger.warning("No documents provided to pipeline.")
            return {"chunks": []}

        logger.info("NODE 0: Document Intake (%d document(s))", len(documents))

        def _process_doc(doc: dict) -> List[dict]:
            doc_id = doc.get("doc_id", "unknown")
            raw_text = doc.get("raw_text", "")
            if not raw_text:
                logger.warning("Skipping empty document: %s", doc_id)
                return []
            logger.info("  Processing: %s (%d chars)", doc_id, len(raw_text))
            result = _node_0.process({"raw_text": raw_text, "doc_id": doc_id})
            chunks = result.get("chunks", [])
            logger.info("  -> %d chunk(s) for %s", len(chunks), doc_id)
            return chunks

        all_chunks: List[dict] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(_process_doc, doc): doc for doc in documents}
            for future in concurrent.futures.as_completed(futures):
                doc = futures[future]
                try:
                    all_chunks.extend(future.result())
                except Exception as exc:
                    logger.error(
                        "Error processing document '%s': %s",
                        doc.get("doc_id", "unknown"), exc,
                    )

        logger.info("  Total chunks after intake: %d", len(all_chunks))

        # S1-8: Light output validation
        if not isinstance(all_chunks, list):
            logger.warning("Node 0 validation: 'chunks' is not a list — resetting.")
            all_chunks = []

        # Defendant disambiguation: if multiple defense documents exist from
        # different defendants, assign ordinal suffixes (الأول, الثاني, ...).
        # Must run BEFORE party_manifest is built so the manifest reflects
        # the correct, disambiguated values.
        all_chunks = _disambiguate_defendants(all_chunks)

        # Fix 1/7: Build party manifest — {party: [doc_types]} — from all chunks.
        # This becomes the ground truth for which parties submitted which doc types,
        # used downstream by nodes 4B and 5 to prevent absent-party fabrication.
        _manifest_sets: dict = {}
        for chunk in all_chunks:
            party = chunk.get("party", "غير محدد")
            doc_type = chunk.get("doc_type", "غير محدد")
            if party not in _manifest_sets:
                _manifest_sets[party] = set()
            _manifest_sets[party].add(doc_type)
        party_manifest = {p: list(dt) for p, dt in _manifest_sets.items()}
        logger.info("  Party manifest: %s", {p: list(dt) for p, dt in _manifest_sets.items()})

        return {"chunks": all_chunks, "party_manifest": party_manifest}

    def node_1_classify(state: SummarizationState) -> dict:
        """Node 1: Classify each chunk by legal role."""
        chunks = state.get("chunks", [])
        if not chunks:
            return {"classified_chunks": []}

        logger.info("NODE 1: Role Classification (%d chunk(s))", len(chunks))
        result = _node_1.process({"chunks": chunks})
        classified = result.get("classified_chunks", [])
        logger.info("  -> %d classified chunk(s)", len(classified))

        if not isinstance(classified, list):
            logger.warning("Node 1 validation: 'classified_chunks' is not a list — resetting.")
            classified = []

        return {"classified_chunks": classified}

    def node_2_extract(state: SummarizationState) -> dict:
        """Node 2: Extract atomic legal bullets from classified chunks."""
        classified_chunks = state.get("classified_chunks", [])
        if not classified_chunks:
            return {"bullets": []}

        logger.info("NODE 2: Bullet Extraction (%d chunk(s))", len(classified_chunks))
        result = _node_2.process({"classified_chunks": classified_chunks})
        bullets = result.get("bullets", [])
        logger.info("  -> %d bullet(s) extracted", len(bullets))

        if not isinstance(bullets, list):
            logger.warning("Node 2 validation: 'bullets' is not a list — resetting.")
            bullets = []

        return {"bullets": bullets}

    def node_3_aggregate(state: SummarizationState) -> dict:
        """Node 3: Aggregate bullets into agreed/disputed/party-specific per role."""
        bullets = state.get("bullets", [])
        if not bullets:
            return {"role_aggregations": []}

        logger.info("NODE 3: Aggregation (%d bullet(s))", len(bullets))
        result = _node_3.process({"bullets": bullets})
        aggregations = result.get("role_aggregations", [])
        logger.info("  -> %d role aggregation(s)", len(aggregations))

        if not isinstance(aggregations, list):
            logger.warning("Node 3 validation: 'role_aggregations' is not a list — resetting.")
            aggregations = []

        return {"role_aggregations": aggregations}

    def node_4a_cluster(state: SummarizationState) -> dict:
        """Node 4A: Thematic clustering within each role."""
        role_aggregations = state.get("role_aggregations", [])
        if not role_aggregations:
            return {"themed_roles": []}

        logger.info("NODE 4A: Thematic Clustering (%d role(s))", len(role_aggregations))
        result = _node_4a.process({"role_aggregations": role_aggregations})
        themed_roles = result.get("themed_roles", [])
        logger.info("  -> %d themed role(s)", len(themed_roles))

        if not isinstance(themed_roles, list):
            logger.warning("Node 4A validation: 'themed_roles' is not a list — resetting.")
            themed_roles = []

        return {"themed_roles": themed_roles}

    def node_4b_synthesize(state: SummarizationState) -> dict:
        """Node 4B: Synthesize structured, citation-anchored summaries per theme."""
        themed_roles = state.get("themed_roles", [])
        if not themed_roles:
            return {"role_theme_summaries": []}

        logger.info("NODE 4B: Theme Synthesis (%d role(s))", len(themed_roles))
        result = _node_4b.process({
            "themed_roles": themed_roles,
            "party_manifest": state.get("party_manifest", {}),  # Fix 7
        })
        summaries = result.get("role_theme_summaries", [])
        logger.info("  -> %d role summary group(s)", len(summaries))

        if not isinstance(summaries, list):
            logger.warning("Node 4B validation: 'role_theme_summaries' is not a list — resetting.")
            summaries = []

        return {"role_theme_summaries": summaries}

    def node_5_brief(state: SummarizationState) -> dict:
        """Node 5: Generate the final judge-facing case brief."""
        role_theme_summaries = state.get("role_theme_summaries", [])
        if not role_theme_summaries:
            return {
                "case_brief": {},
                "all_sources": [],
                "rendered_brief": "",
            }

        logger.info("NODE 5: Case Brief Generation (%d role(s))", len(role_theme_summaries))
        result = _node_5.process({
            "role_theme_summaries": role_theme_summaries,
            "party_manifest": state.get("party_manifest", {}),  # Fix 7
        })
        return {
            "case_brief": result.get("case_brief", {}),
            "all_sources": result.get("all_sources", []),
            "rendered_brief": result.get("rendered_brief", ""),
        }

    # -----------------------------------------------------------------------
    # Graph construction
    # -----------------------------------------------------------------------

    graph = StateGraph(SummarizationState)

    graph.add_node("node_0_intake", node_0_intake)
    graph.add_node("node_1_classify", node_1_classify)
    graph.add_node("node_2_extract", node_2_extract)
    graph.add_node("node_3_aggregate", node_3_aggregate)
    graph.add_node("node_4a_cluster", node_4a_cluster)
    graph.add_node("node_4b_synthesize", node_4b_synthesize)
    graph.add_node("node_5_brief", node_5_brief)

    graph.add_edge(START, "node_0_intake")
    graph.add_edge("node_0_intake", "node_1_classify")
    graph.add_edge("node_1_classify", "node_2_extract")
    graph.add_edge("node_2_extract", "node_3_aggregate")
    graph.add_edge("node_3_aggregate", "node_4a_cluster")
    graph.add_edge("node_4a_cluster", "node_4b_synthesize")
    graph.add_edge("node_4b_synthesize", "node_5_brief")
    graph.add_edge("node_5_brief", END)

    return graph.compile()
