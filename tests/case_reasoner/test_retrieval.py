# """
# test_retrieval.py — Unit tests for retrieve_law_node and retrieve_facts_node.

# Patch targets: tools.civil_law_rag_tool, tools.case_documents_rag_tool
# (Tools imported lazily inside node functions via `from tools import X`)
# """

# import pathlib
# import sys
# from unittest.mock import MagicMock, patch, call

# import pytest

# _CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
# if str(_CR_DIR) not in sys.path:
#     sys.path.insert(0, str(_CR_DIR))

# from nodes.retrieval import retrieve_law_node, retrieve_facts_node


# def _make_law_state(**kwargs):
#     base = {
#         "issue_title": "التعويض عن الإخلال بالعقد",
#         "source_text": "نص مصدري",
#         "case_id": "test-001",
#     }
#     base.update(kwargs)
#     return base


# def _make_fact_state(**kwargs):
#     base = {
#         "issue_title": "التعويض عن الإخلال بالعقد",
#         "source_text": "نص مصدري",
#         "case_id": "test-001",
#     }
#     base.update(kwargs)
#     return base


# def _make_civil_law_result(answer="نص قانوني مسترد", sources=None):
#     return {
#         "answer": answer,
#         "sources": sources or [{"article": 148, "content": "نص المادة 148", "title": "عقد"}],
#         "classification": "",
#         "retrieval_confidence": 0.9,
#         "citation_integrity": 0.9,
#         "from_cache": False,
#         "error": None,
#     }


# def _make_fact_result(final_answer="وقائع القضية المسترداة"):
#     return {
#         "final_answer": final_answer,
#         "sub_answers": [],
#         "error": None,
#     }


# # ---------------------------------------------------------------------------
# # retrieve_law_node
# # ---------------------------------------------------------------------------

# @pytest.mark.unit
# class TestRetrieveLawNode:
#     """T-RETRIEVAL-01: retrieve_law_node calls only civil_law_rag_tool."""

#     def test_returns_law_retrieval_result_and_articles(self):
#         with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_tool:
#             output = retrieve_law_node(_make_law_state())

#         assert "law_retrieval_result" in output
#         assert "retrieved_articles" in output
#         assert len(output["retrieved_articles"]) == 1
#         assert output["retrieved_articles"][0]["article_number"] == 148

#     def test_parse_articles_normalizes_arabic_digits(self):
#         result = _make_civil_law_result(sources=[
#             {"article": "٢٤٨", "content": "نص المادة", "title": "عقد"},
#         ])
#         with patch("tools.civil_law_rag_tool", return_value=result):
#             output = retrieve_law_node(_make_law_state())

#         assert output["retrieved_articles"][0]["article_number"] == 248

#     def test_query_uses_issue_title_and_source_text(self):
#         with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_tool:
#             retrieve_law_node(_make_law_state(
#                 issue_title="مسألة التعويض",
#                 source_text="نص الخلاف",
#             ))

#         call_args = mock_tool.call_args[0][0]  # first positional arg
#         assert "مسألة التعويض" in call_args
#         assert "نص الخلاف" in call_args

#     def test_query_truncated_to_500_chars(self):
#         long_title = "ع" * 300
#         long_source = "ن" * 300
#         captured = []

#         def capture_call(query):
#             captured.append(query)
#             return _make_civil_law_result()

#         with patch("tools.civil_law_rag_tool", side_effect=capture_call):
#             retrieve_law_node(_make_law_state(issue_title=long_title, source_text=long_source))

#         assert len(captured[0]) <= 500

#     def test_empty_result_on_tool_failure(self):
#         with patch("tools.civil_law_rag_tool", side_effect=RuntimeError("tool error")):
#             output = retrieve_law_node(_make_law_state())

#         assert output["law_retrieval_result"]["answer"] == ""
#         assert output["retrieved_articles"] == []

#     def test_error_log_on_failure(self):
#         with patch("tools.civil_law_rag_tool", side_effect=RuntimeError("tool error")):
#             output = retrieve_law_node(_make_law_state())

#         assert "error_log" in output
#         assert len(output["error_log"]) > 0

#     def test_intermediate_steps_logged(self):
#         with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()):
#             output = retrieve_law_node(_make_law_state())

#         assert "intermediate_steps" in output

#     def test_law_tool_only_civil_law_rag_called(self):
#         """Constraint #3: law retrieval node NEVER calls case_documents_rag_tool."""
#         with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_law, \
#              patch("tools.case_documents_rag_tool") as mock_fact:
#             retrieve_law_node(_make_law_state())

#         mock_law.assert_called_once()
#         mock_fact.assert_not_called()

#     def test_multiple_articles_parsed(self):
#         result = _make_civil_law_result(sources=[
#             {"article": 148, "content": "نص 148"},
#             {"article": 176, "content": "نص 176"},
#         ])
#         with patch("tools.civil_law_rag_tool", return_value=result):
#             output = retrieve_law_node(_make_law_state())

#         nums = {a["article_number"] for a in output["retrieved_articles"]}
#         assert nums == {148, 176}


# # ---------------------------------------------------------------------------
# # retrieve_facts_node
# # ---------------------------------------------------------------------------

# @pytest.mark.unit
# class TestRetrieveFactsNode:
#     """T-RETRIEVAL-02: retrieve_facts_node calls only case_documents_rag_tool."""

#     def test_returns_retrieved_facts(self):
#         with patch("tools.case_documents_rag_tool", return_value=_make_fact_result("وقائع مهمة")):
#             output = retrieve_facts_node(_make_fact_state())

#         assert "retrieved_facts" in output
#         assert output["retrieved_facts"] == "وقائع مهمة"

#     def test_passes_case_id_to_tool(self):
#         with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()) as mock_tool:
#             retrieve_facts_node(_make_fact_state(case_id="CASE-XYZ-007"))

#         call_kwargs = mock_tool.call_args
#         case_id_passed = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1].get("case_id")
#         assert case_id_passed == "CASE-XYZ-007"

#     def test_partial_error_logged_as_warning(self, caplog):
#         import logging
#         result = _make_fact_result()
#         result["error"] = "نتيجة جزئية"
#         with patch("tools.case_documents_rag_tool", return_value=result):
#             with caplog.at_level(logging.WARNING):
#                 retrieve_facts_node(_make_fact_state())
#         # Warning should be logged but node should not fail

#     def test_empty_result_on_tool_failure(self):
#         with patch("tools.case_documents_rag_tool", side_effect=RuntimeError("tool error")):
#             output = retrieve_facts_node(_make_fact_state())

#         assert output["retrieved_facts"] == ""
#         assert output["fact_retrieval_result"]["final_answer"] == ""

#     def test_error_log_on_failure(self):
#         with patch("tools.case_documents_rag_tool", side_effect=RuntimeError("tool error")):
#             output = retrieve_facts_node(_make_fact_state())

#         assert "error_log" in output
#         assert len(output["error_log"]) > 0

#     def test_intermediate_steps_logged(self):
#         with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()):
#             output = retrieve_facts_node(_make_fact_state())

#         assert "intermediate_steps" in output

#     def test_fact_node_only_calls_case_doc_tool(self):
#         """Constraint #3: fact retrieval node NEVER calls civil_law_rag_tool."""
#         with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()) as mock_fact, \
#              patch("tools.civil_law_rag_tool") as mock_law:
#             retrieve_facts_node(_make_fact_state())

#         mock_fact.assert_called_once()
#         mock_law.assert_not_called()

#     def test_returns_fact_retrieval_result(self):
#         fact_result = {"final_answer": "وقائع", "sub_answers": ["جزء أول"], "error": None}
#         with patch("tools.case_documents_rag_tool", return_value=fact_result):
#             output = retrieve_facts_node(_make_fact_state())

#         assert "fact_retrieval_result" in output
#         assert output["fact_retrieval_result"]["final_answer"] == "وقائع"


"""
test_retrieval.py — Unit tests for retrieve_law_node and retrieve_facts_node.

After the query generation update, both nodes consume law_queries / fact_queries
lists (one entry per element) instead of building a single generic query.

Patch targets: tools.civil_law_rag_tool, tools.case_documents_rag_tool
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch, call

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.retrieval import retrieve_law_node, retrieve_facts_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_law_queries(*pairs):
    """Build law_queries list from (element_id, query) pairs."""
    return [{"element_id": eid, "query": q} for eid, q in pairs]


def _make_fact_queries(*pairs):
    """Build fact_queries list from (element_id, query) pairs."""
    return [{"element_id": eid, "query": q} for eid, q in pairs]


def _make_law_state(**kwargs):
    base = {
        "issue_title": "التعويض عن الإخلال بالعقد",
        "source_text": "نص مصدري",
        "case_id": "test-001",
        "law_queries": _make_law_queries(
            ("E1", "ما هي أحكام القانون المدني المصري المتعلقة بوجود العقد؟"),
            ("E2", "ما هي أحكام القانون المدني المصري المتعلقة بالإخلال بالعقد؟"),
        ),
    }
    base.update(kwargs)
    return base


def _make_fact_state(**kwargs):
    base = {
        "issue_title": "التعويض عن الإخلال بالعقد",
        "source_text": "نص مصدري",
        "case_id": "test-001",
        "fact_queries": _make_fact_queries(
            ("E1", "ما الوقائع المتعلقة بوجود العقد في مستندات القضية؟"),
            ("E2", "ما الوقائع المتعلقة بالإخلال في مستندات القضية؟"),
        ),
    }
    base.update(kwargs)
    return base


def _make_civil_law_result(answer="نص قانوني مسترد", sources=None):
    return {
        "answer": answer,
        "sources": sources or [{"article": 148, "content": "نص المادة 148", "title": "عقد"}],
        "classification": "",
        "retrieval_confidence": 0.9,
        "citation_integrity": 0.9,
        "from_cache": False,
        "error": None,
    }


def _make_fact_result(final_answer="وقائع القضية المسترداة"):
    return {
        "final_answer": final_answer,
        "sub_answers": [],
        "error": None,
    }


# ---------------------------------------------------------------------------
# retrieve_law_node — happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrieveLawNodeHappy:
    """T-RETRIEVAL-01: retrieve_law_node loops over law_queries."""

    def test_returns_law_retrieval_result_and_articles(self):
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()):
            output = retrieve_law_node(_make_law_state())

        assert "law_retrieval_result" in output
        assert "retrieved_articles" in output

    def test_calls_tool_once_per_element_query(self):
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_tool:
            retrieve_law_node(_make_law_state())

        # 2 queries in _make_law_state → 2 calls
        assert mock_tool.call_count == 2

    def test_each_query_string_sent_to_tool(self):
        queries = _make_law_queries(
            ("E1", "سؤال قانوني خاص بالعنصر الأول"),
            ("E2", "سؤال قانوني خاص بالعنصر الثاني"),
        )
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_tool:
            retrieve_law_node(_make_law_state(law_queries=queries))

        called_queries = [c[0][0] for c in mock_tool.call_args_list]
        assert "سؤال قانوني خاص بالعنصر الأول" in called_queries
        assert "سؤال قانوني خاص بالعنصر الثاني" in called_queries

    def test_articles_deduplicated_across_queries(self):
        """Same article returned by two queries → appears only once in output."""
        result_148 = _make_civil_law_result(sources=[{"article": 148, "content": "نص المادة"}])
        with patch("tools.civil_law_rag_tool", return_value=result_148):
            output = retrieve_law_node(_make_law_state())

        nums = [a["article_number"] for a in output["retrieved_articles"]]
        assert nums.count(148) == 1

    def test_articles_from_all_queries_merged(self):
        """Two queries returning different articles → both in output."""
        call_count = [0]

        def side_effect(query):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_civil_law_result(sources=[{"article": 148, "content": "نص 148"}])
            return _make_civil_law_result(sources=[{"article": 176, "content": "نص 176"}])

        with patch("tools.civil_law_rag_tool", side_effect=side_effect):
            output = retrieve_law_node(_make_law_state())

        nums = {a["article_number"] for a in output["retrieved_articles"]}
        assert 148 in nums
        assert 176 in nums

    def test_parse_articles_normalizes_arabic_digits(self):
        result = _make_civil_law_result(sources=[
            {"article": "٢٤٨", "content": "نص المادة", "title": "عقد"},
        ])
        with patch("tools.civil_law_rag_tool", return_value=result):
            output = retrieve_law_node(_make_law_state(
                law_queries=_make_law_queries(("E1", "سؤال"))
            ))

        assert output["retrieved_articles"][0]["article_number"] == 248

    def test_intermediate_steps_logged(self):
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()):
            output = retrieve_law_node(_make_law_state())

        assert "intermediate_steps" in output
        assert len(output["intermediate_steps"]) > 0

    def test_law_tool_only_called_not_fact_tool(self):
        """Constraint #3: law retrieval node NEVER calls case_documents_rag_tool."""
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_law, \
             patch("tools.case_documents_rag_tool") as mock_fact:
            retrieve_law_node(_make_law_state())

        assert mock_law.call_count >= 1
        mock_fact.assert_not_called()

    def test_query_truncated_to_500_chars(self):
        long_query = "ع" * 600
        captured = []

        def capture(query):
            captured.append(query)
            return _make_civil_law_result()

        with patch("tools.civil_law_rag_tool", side_effect=capture):
            retrieve_law_node(_make_law_state(
                law_queries=_make_law_queries(("E1", long_query))
            ))

        assert len(captured[0]) <= 500


# ---------------------------------------------------------------------------
# retrieve_law_node — fallback (no queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrieveLawNodeFallback:
    """Fallback behavior when law_queries is empty or missing."""

    def test_fallback_when_law_queries_empty(self):
        """Empty law_queries → falls back to generic query from issue_title + source_text."""
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_tool:
            output = retrieve_law_node(_make_law_state(law_queries=[]))

        # Still calls the tool once with a fallback query
        assert mock_tool.call_count == 1
        assert "retrieved_articles" in output

    def test_fallback_query_contains_issue_title(self):
        captured = []

        def capture(query):
            captured.append(query)
            return _make_civil_law_result()

        with patch("tools.civil_law_rag_tool", side_effect=capture):
            retrieve_law_node(_make_law_state(
                issue_title="مسألة التعويض الخاصة",
                law_queries=[],
            ))

        assert "مسألة التعويض الخاصة" in captured[0]

    def test_empty_result_on_tool_failure(self):
        with patch("tools.civil_law_rag_tool", side_effect=RuntimeError("tool error")):
            output = retrieve_law_node(_make_law_state(
                law_queries=_make_law_queries(("E1", "سؤال"))
            ))

        assert output["retrieved_articles"] == []

    def test_error_log_on_failure(self):
        with patch("tools.civil_law_rag_tool", side_effect=RuntimeError("tool error")):
            output = retrieve_law_node(_make_law_state(
                law_queries=_make_law_queries(("E1", "سؤال"))
            ))

        assert "error_log" in output
        assert len(output["error_log"]) > 0

    def test_partial_failure_continues_remaining_queries(self):
        """One query fails → other queries still processed."""
        call_count = [0]

        def side_effect(query):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first query failed")
            return _make_civil_law_result(sources=[{"article": 176, "content": "نص"}])

        with patch("tools.civil_law_rag_tool", side_effect=side_effect):
            output = retrieve_law_node(_make_law_state())

        # Second query still produced an article
        nums = {a["article_number"] for a in output["retrieved_articles"]}
        assert 176 in nums


# ---------------------------------------------------------------------------
# retrieve_facts_node — happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrieveFactsNodeHappy:
    """T-RETRIEVAL-02: retrieve_facts_node loops over fact_queries."""

    def test_returns_retrieved_facts(self):
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result("وقائع مهمة")):
            output = retrieve_facts_node(_make_fact_state(
                fact_queries=_make_fact_queries(("E1", "سؤال"))
            ))

        assert "retrieved_facts" in output
        assert "وقائع مهمة" in output["retrieved_facts"]

    def test_calls_tool_once_per_element_query(self):
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()) as mock_tool:
            retrieve_facts_node(_make_fact_state())

        # 2 queries in _make_fact_state → 2 calls
        assert mock_tool.call_count == 2

    def test_passes_case_id_to_every_call(self):
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()) as mock_tool:
            retrieve_facts_node(_make_fact_state(case_id="CASE-XYZ-007"))

        for c in mock_tool.call_args_list:
            case_id_passed = c[0][1] if len(c[0]) > 1 else c[1].get("case_id")
            assert case_id_passed == "CASE-XYZ-007"

    def test_facts_labeled_with_element_id(self):
        """Each element's facts prefixed with [element_id] in retrieved_facts."""
        call_count = [0]

        def side_effect(query, case_id):
            call_count[0] += 1
            return _make_fact_result(f"وقائع العنصر {call_count[0]}")

        with patch("tools.case_documents_rag_tool", side_effect=side_effect):
            output = retrieve_facts_node(_make_fact_state())

        facts = output["retrieved_facts"]
        assert "[E1]" in facts
        assert "[E2]" in facts

    def test_facts_from_all_elements_concatenated(self):
        """Facts from all element queries appear in retrieved_facts."""
        responses = [
            _make_fact_result("وقائع متعلقة بالعنصر الأول"),
            _make_fact_result("وقائع متعلقة بالعنصر الثاني"),
        ]
        call_count = [0]

        def side_effect(query, case_id):
            r = responses[call_count[0]]
            call_count[0] += 1
            return r

        with patch("tools.case_documents_rag_tool", side_effect=side_effect):
            output = retrieve_facts_node(_make_fact_state())

        assert "وقائع متعلقة بالعنصر الأول" in output["retrieved_facts"]
        assert "وقائع متعلقة بالعنصر الثاني" in output["retrieved_facts"]

    def test_intermediate_steps_logged(self):
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()):
            output = retrieve_facts_node(_make_fact_state())

        assert "intermediate_steps" in output

    def test_fact_node_only_calls_case_doc_tool(self):
        """Constraint #3: fact retrieval node NEVER calls civil_law_rag_tool."""
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()) as mock_fact, \
             patch("tools.civil_law_rag_tool") as mock_law:
            retrieve_facts_node(_make_fact_state())

        assert mock_fact.call_count >= 1
        mock_law.assert_not_called()

    def test_returns_fact_retrieval_result(self):
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result("وقائع")):
            output = retrieve_facts_node(_make_fact_state(
                fact_queries=_make_fact_queries(("E1", "سؤال"))
            ))

        assert "fact_retrieval_result" in output
        # Update the assertion to include the element label and newline
        assert output["fact_retrieval_result"]["final_answer"] == "[E1]:\nوقائع"

    def test_partial_error_logged_as_warning(self, caplog):
        import logging
        result = _make_fact_result()
        result["error"] = "نتيجة جزئية"
        with patch("tools.case_documents_rag_tool", return_value=result):
            with caplog.at_level(logging.WARNING):
                retrieve_facts_node(_make_fact_state(
                    fact_queries=_make_fact_queries(("E1", "سؤال"))
                ))
        # Warning should be logged but node should not fail


# ---------------------------------------------------------------------------
# retrieve_facts_node — fallback (no queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrieveFactsNodeFallback:
    """Fallback behavior when fact_queries is empty or missing."""

    def test_fallback_when_fact_queries_empty(self):
        """Empty fact_queries → falls back to generic query from issue_title + source_text."""
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()) as mock_tool:
            output = retrieve_facts_node(_make_fact_state(fact_queries=[]))

        assert mock_tool.call_count == 1
        assert "retrieved_facts" in output

    def test_fallback_query_contains_issue_title(self):
        captured = []

        def capture(query, case_id):
            captured.append(query)
            return _make_fact_result()

        with patch("tools.case_documents_rag_tool", side_effect=capture):
            retrieve_facts_node(_make_fact_state(
                issue_title="مسألة خاصة جداً",
                fact_queries=[],
            ))

        assert "مسألة خاصة جداً" in captured[0]

    def test_empty_facts_on_tool_failure(self):
        with patch("tools.case_documents_rag_tool", side_effect=RuntimeError("tool error")):
            output = retrieve_facts_node(_make_fact_state(
                fact_queries=_make_fact_queries(("E1", "سؤال"))
            ))

        assert output["retrieved_facts"] == ""

    def test_error_log_on_failure(self):
        with patch("tools.case_documents_rag_tool", side_effect=RuntimeError("tool error")):
            output = retrieve_facts_node(_make_fact_state(
                fact_queries=_make_fact_queries(("E1", "سؤال"))
            ))

        assert "error_log" in output
        assert len(output["error_log"]) > 0

    def test_partial_failure_continues_remaining_queries(self):
        """One element query fails → others still processed."""
        call_count = [0]

        def side_effect(query, case_id):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first query failed")
            return _make_fact_result("وقائع من العنصر الثاني")

        with patch("tools.case_documents_rag_tool", side_effect=side_effect):
            output = retrieve_facts_node(_make_fact_state())

        assert "وقائع من العنصر الثاني" in output["retrieved_facts"]

    def test_empty_answer_elements_excluded_from_facts(self):
        """Elements whose tool returns empty string don't create empty labeled sections."""
        responses = [
            _make_fact_result(""),            # E1 empty → should not appear
            _make_fact_result("وقائع E2"),    # E2 has content
        ]
        call_count = [0]

        def side_effect(query, case_id):
            r = responses[call_count[0]]
            call_count[0] += 1
            return r

        with patch("tools.case_documents_rag_tool", side_effect=side_effect):
            output = retrieve_facts_node(_make_fact_state())

        # E1 produced no content → its label should not appear
        assert "[E1]" not in output["retrieved_facts"]
        assert "وقائع E2" in output["retrieved_facts"]