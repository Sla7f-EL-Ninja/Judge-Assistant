"""
Defines and constructs the LangGraph workflow for the Summarization pipeline.

The pipeline processes legal case documents through 7 nodes:
  Node 0: Document Intake  — clean, extract metadata, segment into chunks
  Node 1: Role Classification — classify each chunk by legal role
  Node 2: Bullet Extraction — extract atomic legal ideas from each chunk
  Node 3: Aggregation — group bullets into agreed/disputed/party-specific per role
  Node 4A: Thematic Clustering — organize items into themes within each role
  Node 4B: Theme Synthesis — produce 2-3 paragraph summaries per theme
  Node 5: Case Brief Generation — produce a 7-section judge-facing brief
"""

import concurrent.futures
from typing import List

from langgraph.graph import StateGraph, START, END

from summarize.state import SummarizationState, disambiguate_defendants
from summarize.nodes import (
    Node0_DocumentIntake,
    Node1_RoleClassifier,
    Node2_BulletExtractor,
    Node3_Aggregator,
    Node4A_ThematicClustering,
    Node4B_ThemeSynthesis,
    Node5_BriefGenerator,
)
from summarize.utils import get_logger

logger = get_logger("hakim.graph")


def create_pipeline(llm_or_config):
    """Initialise all nodes and return a compiled LangGraph pipeline.

    Args:
        llm_or_config:
            - A single LLM object — used for all nodes.
            - A dict with optional keys "high" and "low":
                - "high" tier: Nodes 2, 3, 4B, 5 (complex synthesis/reasoning)
                - "low"  tier: Nodes 0, 1, 4A  (classification/clustering)
              If only one key is present it is used for all nodes.

    Returns:
        A compiled LangGraph application ready for app.invoke(initial_state).
    """
    # --- Resolve tiers ---
    if isinstance(llm_or_config, dict):
        llm_high = (
            llm_or_config.get("high")
            or next(iter(llm_or_config.values()))
        )
        llm_low = llm_or_config.get("low", llm_high)
    else:
        llm_high = llm_or_config
        llm_low = llm_or_config

    # --- Instantiate nodes locally ---
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
    # -----------------------------------------------------------------------

    def node_0_intake(state: SummarizationState) -> dict:
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

        if not isinstance(all_chunks, list):
            logger.warning("Node 0 validation: 'chunks' is not a list — resetting.")
            all_chunks = []

        all_chunks = disambiguate_defendants(all_chunks)

        _manifest_sets: dict = {}
        for chunk in all_chunks:
            party = chunk.get("party", "غير محدد")
            doc_type = chunk.get("doc_type", "غير محدد")
            if party not in _manifest_sets:
                _manifest_sets[party] = set()
            _manifest_sets[party].add(doc_type)

        base_labels = {
            b for b in ("المدعى عليه", "المدعى عليها")
            if b in _manifest_sets
            and any(p.startswith(b) and p != b for p in _manifest_sets)
        }
        for base in base_labels:
            variants = [p for p in _manifest_sets if p.startswith(base) and p != base]
            for variant in variants:
                _manifest_sets[variant].update(_manifest_sets[base])
            del _manifest_sets[base]
            logger.info("  Merged ghost party '%s' into: %s", base, variants)

        party_manifest = {p: list(dt) for p, dt in _manifest_sets.items()}
        logger.info("  Party manifest: %s", {p: list(dt) for p, dt in _manifest_sets.items()})

        return {"chunks": all_chunks, "party_manifest": party_manifest}

    def node_1_classify(state: SummarizationState) -> dict:
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
        themed_roles = state.get("themed_roles", [])
        if not themed_roles:
            return {"role_theme_summaries": []}

        logger.info("NODE 4B: Theme Synthesis (%d role(s))", len(themed_roles))
        result = _node_4b.process({
            "themed_roles": themed_roles,
            "party_manifest": state.get("party_manifest", {}),
        })
        summaries = result.get("role_theme_summaries", [])
        logger.info("  -> %d role summary group(s)", len(summaries))

        if not isinstance(summaries, list):
            logger.warning("Node 4B validation: 'role_theme_summaries' is not a list — resetting.")
            summaries = []

        return {"role_theme_summaries": summaries}

    def node_5_brief(state: SummarizationState) -> dict:
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
            "party_manifest": state.get("party_manifest", {}),
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
