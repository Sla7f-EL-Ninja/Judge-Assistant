"""
test_imports.py — every module imports cleanly; compiled graph has .invoke.
"""

import pytest


def test_chat_reasoner_package():
    import chat_reasoner  # noqa: F401


def test_interface_module():
    from chat_reasoner import interface  # noqa: F401


def test_state_module():
    from chat_reasoner import state  # noqa: F401


def test_graph_module():
    from chat_reasoner import graph  # noqa: F401


def test_tools_module():
    from chat_reasoner import tools  # noqa: F401


def test_trace_module():
    from chat_reasoner import trace  # noqa: F401


def test_prompts_module():
    from chat_reasoner import prompts  # noqa: F401


def test_nodes_planner():
    from chat_reasoner.nodes import planner  # noqa: F401


def test_nodes_plan_validator():
    from chat_reasoner.nodes import plan_validator  # noqa: F401


def test_nodes_executor():
    from chat_reasoner.nodes import executor  # noqa: F401


def test_nodes_replanner():
    from chat_reasoner.nodes import replanner  # noqa: F401


def test_nodes_synthesizer():
    from chat_reasoner.nodes import synthesizer  # noqa: F401


def test_nodes_trace_writer():
    from chat_reasoner.nodes import trace_writer  # noqa: F401


def test_graph_app_has_invoke():
    from chat_reasoner.graph import app
    assert hasattr(app, "invoke"), "compiled graph must expose .invoke"


def test_adapter_import():
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter  # noqa: F401


def test_package_init_exports():
    from chat_reasoner import app, AgentAdapter, AgentResult, ChatReasonerState  # noqa: F401
