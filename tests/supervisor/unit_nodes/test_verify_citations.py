"""
test_verify_citations.py — unit tests for verify_citations_node.

No LLM, no DB. Pure text extraction logic.
"""

from Supervisor.nodes.verify_citations import verify_citations_node
from tests.supervisor.helpers.state_factory import make_state, make_agent_result


class TestVerifyCitationsNode:
    def test_no_op_on_empty_response(self):
        state = make_state(merged_response="")
        result = verify_citations_node(state)
        assert result == {}

    def test_no_op_when_no_articles_cited(self):
        state = make_state(merged_response="هذه إجابة بدون استشهاد بمواد قانونية.")
        result = verify_citations_node(state)
        assert result == {}

    def test_known_article_no_warning(self):
        state = make_state(
            merged_response="تنص المادة 163 على المسؤولية التقصيرية",
            agent_results={
                "civil_law_rag": make_agent_result(
                    sources=["المادة 163 — المسؤولية التقصيرية"],
                    response="المادة 163 تتعلق بالمسؤولية التقصيرية",
                )
            },
        )
        result = verify_citations_node(state)
        assert result == {}

    def test_unknown_article_appends_warning(self):
        state = make_state(
            merged_response="تنص المادة 999 على أحكام خاصة",
            agent_results={
                "civil_law_rag": make_agent_result(
                    sources=["المادة 163 — المسؤولية التقصيرية"],
                    response="المادة 163 فقط",
                )
            },
            validation_feedback="",
        )
        result = verify_citations_node(state)
        assert "validation_feedback" in result
        assert "999" in result["validation_feedback"]
        assert "هلوسة" in result["validation_feedback"] or "غير موجودة" in result["validation_feedback"]

    def test_multiple_unknowns_sorted(self):
        state = make_state(
            merged_response="المادة 999 والمادة 888 في هذه القضية",
            agent_results={
                "civil_law_rag": make_agent_result(
                    sources=["المادة 163"],
                    response="المادة 163",
                )
            },
            validation_feedback="",
        )
        result = verify_citations_node(state)
        feedback = result["validation_feedback"]
        pos_888 = feedback.find("888")
        pos_999 = feedback.find("999")
        assert pos_888 < pos_999, "Articles should be sorted numerically (888 < 999)"

    def test_existing_feedback_preserved(self):
        existing = "ملاحظة سابقة"
        state = make_state(
            merged_response="المادة 999",
            agent_results={"civil_law_rag": make_agent_result(sources=["المادة 1"])},
            validation_feedback=existing,
        )
        result = verify_citations_node(state)
        assert existing in result["validation_feedback"]

    def test_article_in_raw_output_considered_known(self):
        state = make_state(
            merged_response="تنص المادة 200 على عقد البيع",
            agent_results={
                "civil_law_rag": make_agent_result(
                    sources=[],
                    response="",
                    raw_output={"retrieved_text": "المادة 200 — عقد البيع"},
                )
            },
        )
        result = verify_citations_node(state)
        assert result == {}
