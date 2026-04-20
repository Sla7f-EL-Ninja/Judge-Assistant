"""
graph.py

Builds and compiles the Chat Reasoner LangGraph workflow.

Node topology:
  START → planner → plan_validator →(valid)→ executor_fanout
                             ↑(invalid,retry)          |
                             └──────────────────────────┤
                                                   (Send fan-out)
                                             step_worker × N
                                                   ↓
                                              collector
                                          ↙   ↓       ↘
                              executor_fanout  synthesizer  replanner
                                                   ↓           ↓
                                             trace_writer  plan_validator
                                                   ↓
                                                  END
"""

from langgraph.graph import END, START, StateGraph

from chat_reasoner.nodes.executor import (
    collector_node,
    collector_router,
    executor_dispatch_router,
    executor_fanout_node,
    step_worker_node,
)
from chat_reasoner.nodes.plan_validator import (
    plan_validator_node,
    validator_router,
)
from chat_reasoner.nodes.planner import planner_node
from chat_reasoner.nodes.replanner import replanner_node
from chat_reasoner.nodes.synthesizer import (
    synth_router,
    synthesizer_node,
)
from chat_reasoner.nodes.trace_writer import trace_writer_node
from chat_reasoner.state import ChatReasonerState


def _replanner_router(state: ChatReasonerState) -> str:
    """Route out of replanner: failed (guard tripped) → trace_writer, else → plan_validator."""
    if state.get("status") == "failed":
        return "trace_writer"
    return "plan_validator"


def build_chat_reasoner_graph() -> StateGraph:
    workflow = StateGraph(ChatReasonerState)

    # -- Nodes --
    workflow.add_node("planner", planner_node)
    workflow.add_node("plan_validator", plan_validator_node)
    workflow.add_node("executor_fanout", executor_fanout_node)
    workflow.add_node("step_worker", step_worker_node)
    workflow.add_node("collector", collector_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("trace_writer", trace_writer_node)

    # -- Entry --
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "plan_validator")

    # -- Validator routing --
    workflow.add_conditional_edges(
        "plan_validator",
        validator_router,
        {
            "planner": "planner",
            "replanner": "replanner",
            "executor_fanout": "executor_fanout",
        },
    )

    # -- Fan-out: executor_fanout → Send("step_worker") × N OR → collector --
    workflow.add_conditional_edges(
        "executor_fanout",
        executor_dispatch_router,
        # No path_map: Sends are resolved directly; "collector" is a node name
    )

    # -- Fan-in: every step_worker routes to collector --
    workflow.add_edge("step_worker", "collector")

    # -- Collector routing --
    workflow.add_conditional_edges(
        "collector",
        collector_router,
        {
            "executor_fanout": "executor_fanout",
            "synthesizer": "synthesizer",
            "replanner": "replanner",
            "trace_writer": "trace_writer",
        },
    )

    # -- Replanner routing --
    workflow.add_conditional_edges(
        "replanner",
        _replanner_router,
        {
            "plan_validator": "plan_validator",
            "trace_writer": "trace_writer",
        },
    )

    # -- Synthesizer routing --
    workflow.add_conditional_edges(
        "synthesizer",
        synth_router,
        {
            "trace_writer": "trace_writer",
            "replanner": "replanner",
        },
    )

    # -- Terminal --
    workflow.add_edge("trace_writer", END)

    return workflow.compile()


# Compiled graph — import this to invoke Chat Reasoner
app = build_chat_reasoner_graph()
