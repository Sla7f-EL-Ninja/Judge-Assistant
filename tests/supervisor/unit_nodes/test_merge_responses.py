"""
test_merge_responses.py — unit tests for merge_responses_node.

Single-agent pass-through is no-LLM. Multi-agent uses real Gemini.
"""

import pytest

from Supervisor.nodes.merge_responses import merge_responses_node
from tests.supervisor.helpers.state_factory import make_state, make_agent_result
from tests.supervisor.helpers.llm_assertions import assert_arabic_response


class TestMergeResponsesSingleAgent:
    def test_passthrough_returns_response(self):
        state = make_state(
            agent_results={"civil_law_rag": make_agent_result(response="نص المادة 163 هو...")},
            agent_errors={},
        )
        result = merge_responses_node(state)
        assert result["merged_response"] == "نص المادة 163 هو..."

    def test_passthrough_preserves_sources(self):
        state = make_state(
            agent_results={"civil_law_rag": make_agent_result(
                response="إجابة",
                sources=["المادة 163", "المادة 165"],
            )},
        )
        result = merge_responses_node(state)
        assert "المادة 163" in result["sources"]
        assert "المادة 165" in result["sources"]

    def test_sources_deduped(self):
        state = make_state(
            agent_results={"civil_law_rag": make_agent_result(
                response="إجابة",
                sources=["المادة 163", "المادة 163"],
            )},
        )
        result = merge_responses_node(state)
        assert result["sources"].count("المادة 163") == 1


class TestMergeResponsesAllFailed:
    def test_empty_results_returns_empty_merged(self):
        state = make_state(
            agent_results={},
            agent_errors={"civil_law_rag": "Connection refused"},
        )
        result = merge_responses_node(state)
        assert result["merged_response"] == ""
        assert "All agents failed" in result["validation_feedback"]

    def test_error_summary_included(self):
        state = make_state(
            agent_results={},
            agent_errors={"civil_law_rag": "timeout", "reason": "LLM error"},
        )
        result = merge_responses_node(state)
        assert "civil_law_rag" in result["validation_feedback"]


class TestMergeResponsesPartialFailure:
    @pytest.mark.expensive
    def test_partial_failure_includes_disclosure(self):
        state = make_state(
            classified_query="اشرح المادة 163 وطبقها على القضية",
            agent_results={
                "civil_law_rag": make_agent_result(response="المادة 163 تنص على..."),
            },
            agent_errors={"reason": "LLM timeout"},
        )
        result = merge_responses_node(state)
        # The Arabic disclosure caveat should mention the failed agent
        assert "reason" in result["merged_response"] or "ملاحظة" in result["merged_response"]


@pytest.mark.expensive
class TestMergeResponsesMultiAgent:
    def test_two_agent_merge_llm(self):
        state = make_state(
            classified_query="اشرح المادة 163 وما علاقتها بالقضية؟",
            agent_results={
                "civil_law_rag": make_agent_result(
                    response="المادة 163 من القانون المدني تتعلق بالمسؤولية التقصيرية وتوجب التعويض.",
                    sources=["المادة 163 — المسؤولية التقصيرية"],
                ),
                "case_doc_rag": make_agent_result(
                    response="وفقاً لملف القضية، طالب المدعي بتعويض قدره 200,000 جنيه.",
                    sources=["تقرير الخبير"],
                ),
            },
            agent_errors={},
        )
        result = merge_responses_node(state)
        assert result["merged_response"]
        assert_arabic_response(result["merged_response"], min_len=50)
        assert len(result["sources"]) >= 1
