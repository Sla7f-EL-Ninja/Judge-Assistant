"""
graph.py

Defines and constructs the LangGraph workflow for the Summarization pipeline.

The pipeline processes legal case documents through 6 nodes:
  Node 0: Document Intake (clean, extract metadata, segment into chunks)
  Node 1: Role Classification (classify each chunk by legal role)
  Node 2: Bullet Extraction (extract atomic legal ideas from each chunk)
  Node 3: Aggregation (group bullets into agreed/disputed/party-specific per role)
  Node 4A: Thematic Clustering (organize items into themes within each role)
  Node 4B: Theme Synthesis (produce 2-3 paragraph summaries per theme)
  Node 5: Case Brief Generation (produce a 7-section judge-facing brief)

Node 0 runs per-document, then chunks are merged before continuing.
Node 4A and 4B are sequential sub-steps combined under one logical step.
"""

import sys
import os
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END

# Ensure Summerize directory is on the path for all node imports
_summerize_dir = os.path.dirname(os.path.abspath(__file__))
for subdir in ["Node 0", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]:
    _path = os.path.join(_summerize_dir, subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)
if _summerize_dir not in sys.path:
    sys.path.insert(0, _summerize_dir)

from node_0 import Node0_DocumentIntake
from node_1 import Node1_RoleClassifier
from node_2 import Node2_BulletExtractor
from node_3 import Node3_Aggregator
from node_4a import Node4A_ThematicClustering
from node_4b import Node4B_ThemeSynthesis
from node_5 import Node5_BriefGenerator


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class SummarizationState(TypedDict):
    """Shared state flowing through the summarization pipeline."""
    # Input: list of documents to process
    documents: List[dict]           # [{"raw_text": "...", "doc_id": "..."}]

    # Node 0 output
    chunks: List[dict]              # [NormalizedChunk dicts]

    # Node 1 output
    classified_chunks: List[dict]   # [ClassifiedChunk dicts]

    # Node 2 output
    bullets: List[dict]             # [LegalBullet dicts]

    # Node 3 output
    role_aggregations: List[dict]   # [RoleAggregation dicts]

    # Node 4A output
    themed_roles: List[dict]        # [ThemedRole dicts]

    # Node 4B output
    role_theme_summaries: List[dict]  # [RoleThemeSummaries dicts]

    # Node 5 output
    case_brief: dict                # CaseBrief dict
    all_sources: List[str]          # Unique citations
    rendered_brief: str             # Arabic markdown


# ---------------------------------------------------------------------------
# Node wrapper functions
# ---------------------------------------------------------------------------

def node_0_intake(state: SummarizationState) -> dict:
    """Node 0: Process each document through intake (clean, metadata, segment).

    Iterates over all documents in state['documents'], runs Node 0 on each,
    and merges all resulting chunks into a single list.
    """
    documents = state.get("documents", [])
    if not documents:
        print("Warning: no documents provided to pipeline.")
        return {"chunks": []}

    node = _get_node("node_0")
    all_chunks = []

    print(f"\n{'=' * 60}")
    print(f"NODE 0: Document Intake ({len(documents)} document(s))")
    print(f"{'=' * 60}")

    for doc in documents:
        doc_id = doc.get("doc_id", "unknown")
        raw_text = doc.get("raw_text", "")
        if not raw_text:
            print(f"  Skipping empty document: {doc_id}")
            continue

        print(f"  Processing document: {doc_id} ({len(raw_text)} chars)")
        result = node.process({"raw_text": raw_text, "doc_id": doc_id})
        chunks = result.get("chunks", [])
        print(f"  -> {len(chunks)} chunk(s) produced")
        all_chunks.extend(chunks)

    print(f"  Total chunks after intake: {len(all_chunks)}")
    return {"chunks": all_chunks}


def node_1_classify(state: SummarizationState) -> dict:
    """Node 1: Classify each chunk by legal role."""
    chunks = state.get("chunks", [])
    if not chunks:
        return {"classified_chunks": []}

    print(f"\n{'=' * 60}")
    print(f"NODE 1: Role Classification ({len(chunks)} chunk(s))")
    print(f"{'=' * 60}")

    node = _get_node("node_1")
    result = node.process({"chunks": chunks})

    classified = result.get("classified_chunks", [])
    print(f"  -> {len(classified)} classified chunk(s)")
    return {"classified_chunks": classified}


def node_2_extract(state: SummarizationState) -> dict:
    """Node 2: Extract atomic legal bullets from classified chunks."""
    classified_chunks = state.get("classified_chunks", [])
    if not classified_chunks:
        return {"bullets": []}

    print(f"\n{'=' * 60}")
    print(f"NODE 2: Bullet Extraction ({len(classified_chunks)} chunk(s))")
    print(f"{'=' * 60}")

    node = _get_node("node_2")
    result = node.process({"classified_chunks": classified_chunks})

    bullets = result.get("bullets", [])
    print(f"  -> {len(bullets)} bullet(s) extracted")
    return {"bullets": bullets}


def node_3_aggregate(state: SummarizationState) -> dict:
    """Node 3: Aggregate bullets into agreed/disputed/party-specific per role."""
    bullets = state.get("bullets", [])
    if not bullets:
        return {"role_aggregations": []}

    print(f"\n{'=' * 60}")
    print(f"NODE 3: Aggregation ({len(bullets)} bullet(s))")
    print(f"{'=' * 60}")

    node = _get_node("node_3")
    result = node.process({"bullets": bullets})

    aggregations = result.get("role_aggregations", [])
    print(f"  -> {len(aggregations)} role aggregation(s)")
    return {"role_aggregations": aggregations}


def node_4a_cluster(state: SummarizationState) -> dict:
    """Node 4A: Thematic clustering within each role."""
    role_aggregations = state.get("role_aggregations", [])
    if not role_aggregations:
        return {"themed_roles": []}

    print(f"\n{'=' * 60}")
    print(f"NODE 4A: Thematic Clustering ({len(role_aggregations)} role(s))")
    print(f"{'=' * 60}")

    node = _get_node("node_4a")
    result = node.process({"role_aggregations": role_aggregations})

    themed_roles = result.get("themed_roles", [])
    print(f"  -> {len(themed_roles)} themed role(s)")
    return {"themed_roles": themed_roles}


def node_4b_synthesize(state: SummarizationState) -> dict:
    """Node 4B: Synthesize 2-3 paragraph summaries per theme."""
    themed_roles = state.get("themed_roles", [])
    if not themed_roles:
        return {"role_theme_summaries": []}

    print(f"\n{'=' * 60}")
    print(f"NODE 4B: Theme Synthesis ({len(themed_roles)} role(s))")
    print(f"{'=' * 60}")

    node = _get_node("node_4b")
    result = node.process({"themed_roles": themed_roles})

    summaries = result.get("role_theme_summaries", [])
    print(f"  -> {len(summaries)} role summary group(s)")
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

    print(f"\n{'=' * 60}")
    print(f"NODE 5: Case Brief Generation ({len(role_theme_summaries)} role(s))")
    print(f"{'=' * 60}")

    node = _get_node("node_5")
    result = node.process({"role_theme_summaries": role_theme_summaries})

    return {
        "case_brief": result.get("case_brief", {}),
        "all_sources": result.get("all_sources", []),
        "rendered_brief": result.get("rendered_brief", ""),
    }


# ---------------------------------------------------------------------------
# Node instance registry (lazy initialization)
# ---------------------------------------------------------------------------

_node_instances = {}


def _get_node(name: str):
    """Retrieve a cached node instance by name."""
    if name not in _node_instances:
        raise RuntimeError(
            f"Node '{name}' not initialized. Call init_nodes(llm) first."
        )
    return _node_instances[name]


def init_nodes(llm):
    """Initialize all node instances with the given LLM.

    Must be called before running the graph.
    """
    global _node_instances
    _node_instances = {
        "node_0": Node0_DocumentIntake(llm),
        "node_1": Node1_RoleClassifier(llm),
        "node_2": Node2_BulletExtractor(llm),
        "node_3": Node3_Aggregator(llm),
        "node_4a": Node4A_ThematicClustering(llm),
        "node_4b": Node4B_ThemeSynthesis(llm),
        "node_5": Node5_BriefGenerator(llm),
    }
    print("All pipeline nodes initialized.")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph summarization pipeline."""
    graph = StateGraph(SummarizationState)

    # Add nodes
    graph.add_node("node_0_intake", node_0_intake)
    graph.add_node("node_1_classify", node_1_classify)
    graph.add_node("node_2_extract", node_2_extract)
    graph.add_node("node_3_aggregate", node_3_aggregate)
    graph.add_node("node_4a_cluster", node_4a_cluster)
    graph.add_node("node_4b_synthesize", node_4b_synthesize)
    graph.add_node("node_5_brief", node_5_brief)

    # Linear pipeline: Node 0 -> 1 -> 2 -> 3 -> 4A -> 4B -> 5
    graph.add_edge(START, "node_0_intake")
    graph.add_edge("node_0_intake", "node_1_classify")
    graph.add_edge("node_1_classify", "node_2_extract")
    graph.add_edge("node_2_extract", "node_3_aggregate")
    graph.add_edge("node_3_aggregate", "node_4a_cluster")
    graph.add_edge("node_4a_cluster", "node_4b_synthesize")
    graph.add_edge("node_4b_synthesize", "node_5_brief")
    graph.add_edge("node_5_brief", END)

    return graph.compile()


def create_pipeline(llm):
    """One-step helper: initialize nodes and return the compiled graph.

    Usage:
        app = create_pipeline(llm)
        result = app.invoke({"documents": [...]})
    """
    init_nodes(llm)
    return build_graph()
