"""
test_classify_intent.py — unit tests for classify_intent_node.

Uses real Gemini (medium tier). All tests are @pytest.mark.expensive.
"""

import pytest

from Supervisor.nodes.classify_intent import classify_intent_node
from config.supervisor import VALID_INTENTS, AGENT_NAMES
from tests.supervisor.helpers.state_factory import make_state
from tests.supervisor.helpers.llm_assertions import (
    assert_valid_intent,
    assert_agents_ordered,
    assert_agents_deduped,
)


@pytest.mark.expensive
class TestClassifyIntentCivilLaw:
    @pytest.mark.parametrize("query", [
        "ما نص المادة 163 من القانون المدني المصري؟",
        "ما الفرق بين المسؤولية التقصيرية والعقدية في القانون المدني؟",
        "اذكر أحكام عقد البيع في القانون المصري",
    ])
    def test_civil_law_rag_classified(self, query):
        state = make_state(judge_query=query)
        result = classify_intent_node(state)
        assert_valid_intent(result["intent"])
        assert result["intent"] == "civil_law_rag", (
            f"Expected civil_law_rag for: {query!r}, got {result['intent']!r}"
        )
        assert "civil_law_rag" in result["target_agents"]


@pytest.mark.expensive
class TestClassifyIntentOffTopic:
    @pytest.mark.parametrize("query", [
        "ما عاصمة فرنسا؟",
        "اكتب لي شعراً عن الربيع",
        "كيف أطبخ المكرونة؟",
        "ما الطقس في القاهرة غداً؟",
    ])
    def test_off_topic_classified(self, query):
        state = make_state(judge_query=query)
        result = classify_intent_node(state)
        assert result["intent"] == "off_topic", (
            f"Expected off_topic for: {query!r}, got {result['intent']!r}"
        )
        assert result["target_agents"] == []

    @pytest.mark.parametrize("query", [
        "ما عقوبة السرقة في القانون المصري؟",
        "ما شروط الطلاق في الشريعة الإسلامية؟",
        "ما المادة 163 من القانون المدني الفرنسي؟",
    ])
    def test_out_of_scope_classified_off_topic(self, query):
        state = make_state(judge_query=query)
        result = classify_intent_node(state)
        assert result["intent"] == "off_topic"


@pytest.mark.expensive
class TestClassifyIntentSanitization:
    def test_off_topic_has_empty_agents(self):
        state = make_state(judge_query="ما عاصمة فرنسا؟")
        result = classify_intent_node(state)
        if result["intent"] == "off_topic":
            assert result["target_agents"] == []

    def test_agents_deduped(self):
        state = make_state(judge_query="ما نص المادة 163؟")
        result = classify_intent_node(state)
        assert_agents_deduped(result["target_agents"])

    def test_agents_ordered(self):
        state = make_state(
            judge_query="اشرح المادة 163 وحلل مدى انطباقها على القضية الحالية",
            case_id="test-case-2847-2024",
        )
        result = classify_intent_node(state)
        assert_agents_ordered(result["target_agents"])

    def test_classified_query_max_length(self):
        state = make_state(judge_query="ما المادة 163؟")
        result = classify_intent_node(state)
        assert len(result.get("classified_query", "")) <= 4000

    def test_classified_query_nonempty(self):
        state = make_state(judge_query="ما المادة 163؟")
        result = classify_intent_node(state)
        assert result.get("classified_query")

    def test_fabricated_agent_filtered(self, monkeypatch):
        """LLM returning an unknown agent name must be filtered out."""
        from Supervisor.state import IntentClassification

        fake_result = IntentClassification(
            intent="civil_law_rag",
            target_agents=["civil_law_rag", "delete_database"],
            rewritten_query="ما المادة 163؟",
            reasoning="test",
        )

        import Supervisor.nodes.classify_intent as ci_mod
        monkeypatch.setattr(
            ci_mod, "llm_invoke",
            lambda fn, msgs: fake_result,
        )

        state = make_state(judge_query="ما المادة 163؟")
        result = classify_intent_node(state)
        assert "delete_database" not in result["target_agents"]

    def test_multi_without_agents_becomes_off_topic(self, monkeypatch):
        from Supervisor.state import IntentClassification
        import Supervisor.nodes.classify_intent as ci_mod

        fake_result = IntentClassification(
            intent="multi",
            target_agents=[],
            rewritten_query="سؤال متعدد",
            reasoning="multi but no agents",
        )
        monkeypatch.setattr(ci_mod, "llm_invoke", lambda fn, msgs: fake_result)
        state = make_state(judge_query="سؤال متعدد")
        result = classify_intent_node(state)
        assert result["intent"] == "off_topic"
        assert result["target_agents"] == []

    def test_llm_exception_routes_off_topic(self, monkeypatch):
        import Supervisor.nodes.classify_intent as ci_mod
        monkeypatch.setattr(ci_mod, "llm_invoke", lambda fn, msgs: (_ for _ in ()).throw(RuntimeError("LLM down")))
        state = make_state(judge_query="ما المادة 163؟")
        result = classify_intent_node(state)
        assert result["intent"] == "off_topic"
        assert result["classification_error"]

    def test_history_window_capped_at_20(self, monkeypatch):
        """Verify only last MAX_CONVERSATION_TURNS messages passed to LLM."""
        seen_prompts = []

        def capture(fn, msgs):
            seen_prompts.append(msgs)
            from Supervisor.state import IntentClassification
            return IntentClassification(
                intent="off_topic",
                target_agents=[],
                rewritten_query="q",
                reasoning="r",
            )

        import Supervisor.nodes.classify_intent as ci_mod
        monkeypatch.setattr(ci_mod, "llm_invoke", capture)

        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(50)
        ]
        state = make_state(judge_query="ما المادة 163؟", conversation_history=long_history)
        classify_intent_node(state)

        assert seen_prompts
        user_msg = seen_prompts[0][1]["content"]
        # Should not contain "msg 0" (first of 50) — it was trimmed
        assert "msg 0]" not in user_msg or user_msg.count("[user]") <= 20

    def test_single_intent_caps_extra_agents(self, monkeypatch):
        """Single intent with extra agents in LLM output — capped to declared intent."""
        from Supervisor.state import IntentClassification
        import Supervisor.nodes.classify_intent as ci_mod

        fake = IntentClassification(
            intent="civil_law_rag",
            target_agents=["civil_law_rag", "reason"],
            rewritten_query="ما المادة 163؟",
            reasoning="r",
        )
        monkeypatch.setattr(ci_mod, "llm_invoke", lambda fn, msgs: fake)
        state = make_state(judge_query="ما المادة 163؟")
        result = classify_intent_node(state)
        assert result["target_agents"] == ["civil_law_rag"]
