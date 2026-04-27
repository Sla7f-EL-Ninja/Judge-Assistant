"""
test_validate_output.py — unit tests for validate_output_node.

Tests all routing branches: pass, partial_pass, fail_*, validator_error.
Uses monkeypatching to avoid real LLM calls for deterministic branch coverage.
Real LLM tests are @pytest.mark.expensive.
"""

import pytest

from Supervisor.nodes.validate_output import validate_output_node
from Supervisor.state import ValidationResult
from tests.supervisor.helpers.state_factory import make_state, make_agent_result


def _mock_llm_invoke(monkeypatch, result):
    import Supervisor.nodes.validate_output as vo_mod
    monkeypatch.setattr(vo_mod, "llm_invoke", lambda fn, msgs: result)


class TestValidateOutputEmpty:
    def test_empty_merged_response_is_fail_completeness(self):
        state = make_state(merged_response="", retry_count=0)
        result = validate_output_node(state)
        assert result["validation_status"] == "fail_completeness"
        assert result["retry_count"] == 1


class TestValidateOutputPass:
    def test_pass_status_when_all_ok(self, monkeypatch):
        vr = ValidationResult(
            hallucination_pass=True,
            relevance_pass=True,
            completeness_pass=True,
            coherence_pass=True,
            overall_pass=True,
            feedback="",
        )
        _mock_llm_invoke(monkeypatch, vr)
        state = make_state(
            merged_response="المادة 163 تتعلق بالمسؤولية التقصيرية.",
            classified_query="ما المادة 163؟",
        )
        result = validate_output_node(state)
        assert result["validation_status"] == "pass"
        assert result["final_response"] == "المادة 163 تتعلق بالمسؤولية التقصيرية."
        assert result["validation_feedback"] == ""


class TestValidateOutputPartialPass:
    def test_partial_pass_h_r_coh_ok_completeness_fail(self, monkeypatch):
        vr = ValidationResult(
            hallucination_pass=True,
            relevance_pass=True,
            completeness_pass=False,
            coherence_pass=True,
            overall_pass=False,
            feedback="الإجابة غير مكتملة",
        )
        _mock_llm_invoke(monkeypatch, vr)
        state = make_state(
            merged_response="إجابة جزئية عن المادة 163.",
            classified_query="ما المادة 163؟",
        )
        result = validate_output_node(state)
        assert result["validation_status"] == "partial_pass"
        assert "ملاحظة" in result["final_response"]
        # No retry increment on partial_pass
        assert "retry_count" not in result or result.get("retry_count") == 0

    def test_partial_pass_includes_caveat(self, monkeypatch):
        vr = ValidationResult(
            hallucination_pass=True, relevance_pass=True,
            completeness_pass=False, coherence_pass=True,
            overall_pass=False, feedback="ناقصة",
        )
        _mock_llm_invoke(monkeypatch, vr)
        state = make_state(merged_response="إجابة", classified_query="سؤال")
        result = validate_output_node(state)
        assert "---" in result["final_response"]  # caveat separator


class TestValidateOutputFailures:
    def test_fail_hallucination(self, monkeypatch):
        vr = ValidationResult(
            hallucination_pass=False, relevance_pass=True,
            completeness_pass=True, coherence_pass=True,
            overall_pass=False, feedback="هلوسة في الاستشهادات",
        )
        _mock_llm_invoke(monkeypatch, vr)
        state = make_state(merged_response="إجابة كاذبة", retry_count=0)
        result = validate_output_node(state)
        assert result["validation_status"] == "fail_hallucination"
        assert result["retry_count"] == 1

    def test_fail_relevance(self, monkeypatch):
        vr = ValidationResult(
            hallucination_pass=True, relevance_pass=False,
            completeness_pass=True, coherence_pass=True,
            overall_pass=False, feedback="الإجابة غير ذات صلة",
        )
        _mock_llm_invoke(monkeypatch, vr)
        state = make_state(merged_response="إجابة غير صلة", retry_count=0)
        result = validate_output_node(state)
        assert result["validation_status"] == "fail_relevance"
        assert result["retry_count"] == 1

    def test_fail_completeness(self, monkeypatch):
        vr = ValidationResult(
            hallucination_pass=False, relevance_pass=False,
            completeness_pass=False, coherence_pass=False,
            overall_pass=False, feedback="فشل كامل",
        )
        _mock_llm_invoke(monkeypatch, vr)
        state = make_state(merged_response="إجابة فاشلة", retry_count=1)
        result = validate_output_node(state)
        assert "fail_" in result["validation_status"]
        assert result["retry_count"] == 2

    def test_validator_error_on_exception(self, monkeypatch):
        import Supervisor.nodes.validate_output as vo_mod
        monkeypatch.setattr(
            vo_mod, "llm_invoke",
            lambda fn, msgs: (_ for _ in ()).throw(RuntimeError("LLM down")),
        )
        state = make_state(merged_response="إجابة", retry_count=0)
        result = validate_output_node(state)
        assert result["validation_status"] == "validator_error"
        assert result["retry_count"] == 1

    def test_validator_error_on_none_result(self, monkeypatch):
        import Supervisor.nodes.validate_output as vo_mod
        monkeypatch.setattr(vo_mod, "llm_invoke", lambda fn, msgs: None)
        state = make_state(merged_response="إجابة", retry_count=0)
        result = validate_output_node(state)
        assert result["validation_status"] == "validator_error"


@pytest.mark.expensive
class TestValidateOutputRealLLM:
    def test_good_civil_law_response_passes(self):
        state = make_state(
            merged_response=(
                "المادة 163 من القانون المدني المصري: "
                "كل خطأ سبب ضرراً للغير يلزم من ارتكبه بالتعويض. "
                "وتقوم المسؤولية التقصيرية على ثلاثة أركان: الخطأ، والضرر، وعلاقة السببية."
            ),
            classified_query="ما نص المادة 163 من القانون المدني المصري؟",
            agent_results={
                "civil_law_rag": make_agent_result(
                    response="المادة 163 تنص على...",
                    sources=["المادة 163 — المسؤولية التقصيرية"],
                )
            },
        )
        result = validate_output_node(state)
        assert result["validation_status"] in ("pass", "partial_pass")
