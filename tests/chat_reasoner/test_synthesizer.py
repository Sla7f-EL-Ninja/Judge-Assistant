"""
test_synthesizer.py — synthesizer_node with real LLM + synth_router branches.
"""

import json
import re
from datetime import datetime, timezone


_ARABIC_RE = re.compile(r"[؀-ۿ]")


def _step_result_dict(step_id, tool, query, response, sources=None):
    return {
        "step_id": step_id,
        "tool": tool,
        "query": query,
        "status": "success",
        "response": response,
        "sources": sources or [],
        "error": None,
        "raw_output": {},
    }


def _base_state(**overrides):
    state = {
        "original_query": "ما هي وقائع القضية وما الأحكام القانونية المنطبقة؟",
        "case_id": "1234",
        "conversation_history": [],
        "escalation_reason": "مقارنة قانونية",
        "plan": [
            {"step_id": "s1", "tool": "case_doc_rag", "query": "وقائع القضية", "depends_on": []},
            {"step_id": "s2", "tool": "civil_law_rag", "query": "الأحكام المنطبقة", "depends_on": []},
        ],
        "step_results": [
            _step_result_dict(
                "s1", "case_doc_rag", "وقائع القضية",
                "رُفعت الدعوى من المدعي بسبب إخلال المدعى عليه بالتزاماته التعاقدية.",
                sources=["حكم_المحكمة", "صحيفة_دعوى"],
            ),
            _step_result_dict(
                "s2", "civil_law_rag", "الأحكام المنطبقة",
                "تنص المادة 163 من القانون المدني على المسؤولية التقصيرية عند الإخلال.",
                sources=["المادة 163 ق.م", "حكم_المحكمة"],
            ),
        ],
        "step_failures": {},
        "replan_count": 0,
        "replan_trigger_step_id": None,
        "replan_trigger_error": None,
        "run_count": 0,
        "synthesis_attempts": 0,
        "final_answer": "",
        "final_sources": [],
        "status": "running",
        "error_message": None,
        "synth_sufficient": True,
        "plan_validation_status": "valid",
        "plan_validation_feedback": "",
        "validator_retry_count": 0,
        "session_id": "1234::test",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tool_calls_log": [],
        "replan_events": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# synthesizer_node (real LLM)
# ---------------------------------------------------------------------------

def test_synthesizer_arabic_output(_preflight):
    from chat_reasoner.nodes.synthesizer import synthesizer_node
    updates = synthesizer_node(_base_state())
    answer = updates.get("final_answer", "")
    assert answer, "final_answer must not be empty"
    assert _ARABIC_RE.search(answer), "final_answer must contain Arabic characters"


def test_synthesizer_increments_synthesis_attempts(_preflight):
    from chat_reasoner.nodes.synthesizer import synthesizer_node
    state = _base_state(synthesis_attempts=0)
    updates = synthesizer_node(state)
    assert updates.get("synthesis_attempts") == 1


def test_synthesizer_deduplicates_sources(_preflight):
    from chat_reasoner.nodes.synthesizer import synthesizer_node
    # Both steps share "حكم_المحكمة" — must appear only once in final_sources
    updates = synthesizer_node(_base_state())
    sources = updates.get("final_sources", [])
    normalized = [s.strip().lower() for s in sources]
    assert len(normalized) == len(set(normalized)), (
        f"Duplicate sources found: {sources}"
    )


# ---------------------------------------------------------------------------
# synth_router branches (pure, no LLM)
# ---------------------------------------------------------------------------

def test_synth_router_sufficient_goes_to_trace_writer():
    from chat_reasoner.nodes.synthesizer import synth_router
    state = {"synth_sufficient": True, "run_count": 0}
    assert synth_router(state) == "trace_writer"


def test_synth_router_insufficient_below_cap_goes_to_replanner():
    from chat_reasoner.nodes.synthesizer import synth_router
    state = {"synth_sufficient": False, "run_count": 1}
    assert synth_router(state) == "replanner"


def test_synth_router_insufficient_at_cap_goes_to_trace_writer():
    from chat_reasoner.nodes.synthesizer import synth_router
    state = {"synth_sufficient": False, "run_count": 2}
    assert synth_router(state) == "trace_writer"


def test_synth_router_insufficient_triggers_replan_error(_preflight):
    from chat_reasoner.nodes.synthesizer import synthesizer_node
    # Force insufficient by running with empty step results (thin context)
    state = _base_state(
        step_results=[],
        run_count=0,
        original_query="سؤال محدد جداً يتطلب مزيداً من البيانات",
    )
    updates = synthesizer_node(state)
    # If LLM marks insufficient, verify trigger is set
    if not updates.get("synth_sufficient", True):
        assert updates.get("replan_trigger_error", "").startswith("synthesizer_insufficient")
        assert updates.get("run_count", 0) == 1
