"""
test_schemas.py -- Validate Pydantic schema contracts across the codebase.

Ensures required fields are declared correctly and that removing a required
field triggers a validation error.

Marker: unit
"""

import pytest
from pydantic import ValidationError


@pytest.mark.unit
class TestIntentClassification:
    """Validate IntentClassification schema from Supervisor/state.py."""

    def test_required_fields_present(self):
        from Supervisor.state import IntentClassification

        required = IntentClassification.model_json_schema().get("required", [])
        for field in ("intent", "target_agents", "rewritten_query", "reasoning"):
            assert field in required, (
                f"IntentClassification.{field} must be declared as required"
            )

    def test_missing_required_field_raises(self):
        from Supervisor.state import IntentClassification

        with pytest.raises(ValidationError, match="intent"):
            IntentClassification(
                target_agents=["civil_law_rag"],
                rewritten_query="test",
                reasoning="test",
            )


@pytest.mark.unit
class TestValidationResult:
    """Validate ValidationResult schema from Supervisor/state.py."""

    def test_required_fields_present(self):
        from Supervisor.state import ValidationResult

        required = ValidationResult.model_json_schema().get("required", [])
        for field in (
            "hallucination_pass",
            "relevance_pass",
            "completeness_pass",
            "overall_pass",
            "feedback",
        ):
            assert field in required, (
                f"ValidationResult.{field} must be declared as required"
            )

    def test_missing_required_field_raises(self):
        from Supervisor.state import ValidationResult

        with pytest.raises(ValidationError, match="hallucination_pass"):
            ValidationResult(
                relevance_pass=True,
                completeness_pass=True,
                overall_pass=True,
                feedback="ok",
            )


@pytest.mark.unit
class TestAgentResult:
    """Validate AgentResult schema from Supervisor/agents/base.py."""

    def test_required_fields_present(self):
        from Supervisor.agents.base import AgentResult

        required = AgentResult.model_json_schema().get("required", [])
        assert "response" in required, (
            "AgentResult.response must be declared as required"
        )

    def test_defaults_are_applied(self):
        from Supervisor.agents.base import AgentResult

        result = AgentResult(response="test response")
        assert result.sources == [], "AgentResult.sources should default to empty list"
        assert result.raw_output == {}, "AgentResult.raw_output should default to empty dict"
        assert result.error is None, "AgentResult.error should default to None"

    def test_missing_response_raises(self):
        from Supervisor.agents.base import AgentResult

        with pytest.raises(ValidationError, match="response"):
            AgentResult()


@pytest.mark.unit
class TestNormalizedChunk:
    """Validate NormalizedChunk schema from Summerize/schemas.py."""

    def test_required_fields_present(self):
        from Summerize.schemas import NormalizedChunk

        required = NormalizedChunk.model_json_schema().get("required", [])
        for field in (
            "chunk_id",
            "doc_id",
            "page_number",
            "paragraph_number",
            "clean_text",
            "doc_type",
            "party",
        ):
            assert field in required, (
                f"NormalizedChunk.{field} must be declared as required"
            )

    def test_missing_required_field_raises(self):
        from Summerize.schemas import NormalizedChunk

        with pytest.raises(ValidationError):
            NormalizedChunk(
                chunk_id="abc",
                doc_id="doc1",
                # missing page_number, paragraph_number, clean_text, doc_type, party
            )


@pytest.mark.unit
class TestQueryRequest:
    """Validate QueryRequest schema from api/schemas/query.py."""

    def test_required_fields_present(self):
        from api.schemas.query import QueryRequest

        required = QueryRequest.model_json_schema().get("required", [])
        assert "query" in required, (
            "QueryRequest.query must be declared as required"
        )

    def test_missing_query_raises(self):
        from api.schemas.query import QueryRequest

        with pytest.raises(ValidationError, match="query"):
            QueryRequest()

    def test_empty_query_raises(self):
        from api.schemas.query import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_valid_query_accepted(self):
        from api.schemas.query import QueryRequest

        req = QueryRequest(query="ما هي شروط صحة العقد؟")
        assert req.query == "ما هي شروط صحة العقد؟", (
            "QueryRequest should accept valid Arabic query"
        )
