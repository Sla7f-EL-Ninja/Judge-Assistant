"""
test_graph.py — Unit tests for Summerize/graph.py

Tests:
    T-GRAPH-01: SummarizationState has all 10 expected keys
    T-GRAPH-02: create_pipeline() assigns correct tier LLM to each node
"""

import pathlib
import sys
from unittest.mock import MagicMock

import pytest

from summarize.state import SummarizationState
from summarize.graph import create_pipeline


@pytest.mark.unit
class TestSummarizationState:
    """T-GRAPH-01: SummarizationState TypedDict has all 10 expected keys."""

    EXPECTED_KEYS = {
        "documents",
        "chunks",
        "classified_chunks",
        "bullets",
        "role_aggregations",
        "themed_roles",
        "role_theme_summaries",
        "case_brief",
        "all_sources",
        "rendered_brief",
        "party_manifest"
    }

    def test_has_all_expected_keys(self):
        annotations = SummarizationState.__annotations__
        assert set(annotations.keys()) == self.EXPECTED_KEYS

    def test_documents_is_list(self):
        from typing import get_args, get_origin
        import typing
        ann = SummarizationState.__annotations__["documents"]
        # Should be List[dict]
        assert get_origin(ann) is list

    def test_chunks_is_list(self):
        from typing import get_origin
        ann = SummarizationState.__annotations__["chunks"]
        assert get_origin(ann) is list

    def test_rendered_brief_is_str(self):
        ann = SummarizationState.__annotations__["rendered_brief"]
        assert ann is str

    def test_case_brief_is_dict(self):
        ann = SummarizationState.__annotations__["case_brief"]
        assert ann is dict


@pytest.mark.unit
class TestCreatePipeline:
    """T-GRAPH-02: create_pipeline() tiered LLM assignment."""

    def _make_mock_llm(self, name="llm"):
        llm = MagicMock(name=name)
        # with_structured_output must return a parser mock
        llm.with_structured_output.return_value = MagicMock(name=f"{name}_parser")
        return llm

    def test_single_llm_creates_pipeline(self):
        """A single LLM creates a valid compiled LangGraph pipeline."""
        llm = self._make_mock_llm("single")
        pipeline = create_pipeline(llm)
        # The pipeline should be callable (compilable LangGraph app)
        assert hasattr(pipeline, "invoke") or callable(pipeline)

    def test_dict_config_high_low_creates_pipeline(self):
        """Dict config with 'high' and 'low' keys creates pipeline."""
        llm_high = self._make_mock_llm("high")
        llm_low = self._make_mock_llm("low")
        pipeline = create_pipeline({"high": llm_high, "low": llm_low})
        assert hasattr(pipeline, "invoke") or callable(pipeline)

    def test_high_tier_used_for_node2(self):
        """Nodes 2, 3, 4B, 5 use the 'high' LLM tier."""
        llm_high = self._make_mock_llm("high")
        llm_low = self._make_mock_llm("low")
        create_pipeline({"high": llm_high, "low": llm_low})
        # Node 2, 3, 4B, 5 use high — verify with_structured_output called on high
        assert llm_high.with_structured_output.call_count >= 4

    def test_low_tier_used_for_node0(self):
        """Nodes 0, 1, 4A use the 'low' LLM tier."""
        llm_high = self._make_mock_llm("high")
        llm_low = self._make_mock_llm("low")
        create_pipeline({"high": llm_high, "low": llm_low})
        # Node 0, 1, 4A use low — verify with_structured_output called on low
        assert llm_low.with_structured_output.call_count >= 3

    def test_single_llm_used_for_all_nodes(self):
        """Single LLM (not dict) is used for all 7 nodes."""
        llm = self._make_mock_llm("single")
        create_pipeline(llm)
        # All 7 nodes call with_structured_output on the same LLM
        assert llm.with_structured_output.call_count == 7

    def test_dict_with_only_high_uses_high_for_low(self):
        """Dict with only 'high' key uses high for all nodes."""
        llm_high = self._make_mock_llm("high")
        create_pipeline({"high": llm_high})
        assert llm_high.with_structured_output.call_count == 7
