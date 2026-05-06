"""
test_interface.py — AgentResult duck-typing contract and ChatReasonerAdapter construction.
"""


def test_agent_result_required_fields():
    from chat_reasoner.interface import AgentResult
    r = AgentResult(response="hello")
    assert hasattr(r, "response")
    assert hasattr(r, "sources")
    assert hasattr(r, "raw_output")
    assert hasattr(r, "error")


def test_agent_result_defaults():
    from chat_reasoner.interface import AgentResult
    r = AgentResult(response="test")
    assert r.sources == []
    assert r.raw_output == {}
    assert r.error is None


def test_agent_result_response_value():
    from chat_reasoner.interface import AgentResult
    r = AgentResult(response="مرحبا")
    assert r.response == "مرحبا"


def test_agent_result_full_construction():
    from chat_reasoner.interface import AgentResult
    r = AgentResult(
        response="x",
        sources=["s1", "s2"],
        raw_output={"k": "v"},
        error="err",
    )
    assert r.sources == ["s1", "s2"]
    assert r.raw_output == {"k": "v"}
    assert r.error == "err"


def test_adapter_zero_arg_construction():
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter
    adapter = ChatReasonerAdapter()
    assert adapter is not None


def test_adapter_has_invoke():
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter
    adapter = ChatReasonerAdapter()
    assert callable(getattr(adapter, "invoke", None))
