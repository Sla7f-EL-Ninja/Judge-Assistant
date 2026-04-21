"""
test_tools.py — real tool invocations against live systems.

All tests require seeded_case (real docs for case_id="1234" in Qdrant/Mongo).
fetch_summary_report reads the pre-existing summary already in Mongo.
"""

import pytest


# ---------------------------------------------------------------------------
# fetch_summary_report
# ---------------------------------------------------------------------------

def test_fetch_summary_success(seeded_case):
    from chat_reasoner.tools import _run_fetch_summary_report
    step = {"step_id": "s_sum", "tool": "fetch_summary_report", "query": "ملخص القضية"}
    result = _run_fetch_summary_report(step, seeded_case["case_id"], [])
    assert result.status == "success", f"Expected success, got {result.status}: {result.error}"
    assert result.response, "Summary response must not be empty"
    assert isinstance(result.sources, list)


def test_fetch_summary_skipped_nonexistent_case():
    from chat_reasoner.tools import _run_fetch_summary_report
    step = {"step_id": "s_sum", "tool": "fetch_summary_report", "query": "ملخص"}
    result = _run_fetch_summary_report(step, "__nonexistent_case_id__", [])
    assert result.status == "skipped"
    assert result.raw_output.get("reason"), "skipped result must include reason"


def test_fetch_summary_failure_on_bad_uri(monkeypatch):
    import config.supervisor as sup_mod
    monkeypatch.setattr(
        sup_mod, "MONGO_URI",
        "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=200&connectTimeoutMS=200",
    )
    from chat_reasoner.tools import _run_fetch_summary_report
    step = {"step_id": "s_sum", "tool": "fetch_summary_report", "query": "ملخص"}
    result = _run_fetch_summary_report(step, "1234", [])
    assert result.status == "failure"
    assert result.error and result.error.startswith("mongo error:")


# ---------------------------------------------------------------------------
# _run_case_doc_rag
# ---------------------------------------------------------------------------

def test_run_case_doc_rag_success(seeded_case):
    from chat_reasoner.tools import _run_case_doc_rag
    step = {"step_id": "s_rag", "tool": "case_doc_rag", "query": "ما هي وقائع القضية؟"}
    result = _run_case_doc_rag(step, seeded_case["case_id"], [])
    assert result.status == "success", (
        f"case_doc_rag failed: {result.error}"
    )
    assert result.response, "case_doc_rag response must not be empty"


def test_run_case_doc_rag_sources(seeded_case):
    from chat_reasoner.tools import _run_case_doc_rag
    step = {"step_id": "s_rag2", "tool": "case_doc_rag", "query": "من هو الخبير في القضية؟"}
    result = _run_case_doc_rag(step, seeded_case["case_id"], [])
    # sources should be a list (may be empty if RAG found no docs, but status still success)
    assert isinstance(result.sources, list)


# ---------------------------------------------------------------------------
# _run_civil_law_rag
# ---------------------------------------------------------------------------

def test_run_civil_law_rag_success(seeded_case):
    from chat_reasoner.tools import _run_civil_law_rag
    step = {
        "step_id": "s_civil",
        "tool": "civil_law_rag",
        "query": "ما حكم المسؤولية التقصيرية في القانون المدني المصري؟",
    }
    result = _run_civil_law_rag(step, seeded_case["case_id"], [])
    assert result.status == "success", f"civil_law_rag failed: {result.error}"
    assert result.response, "civil_law_rag response must not be empty"


# ---------------------------------------------------------------------------
# dispatch_tool
# ---------------------------------------------------------------------------

def test_dispatch_tool_unknown_tool(seeded_case):
    from chat_reasoner.tools import dispatch_tool
    step = {"step_id": "s_bad", "tool": "nonexistent_tool", "query": "test"}
    result = dispatch_tool(step, seeded_case["case_id"], [])
    assert result.status == "failure"
    assert result.error
