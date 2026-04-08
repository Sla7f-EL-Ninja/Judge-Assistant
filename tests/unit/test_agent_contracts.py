"""
test_agent_contracts.py -- Validate agent adapter return type contracts.

Pure contract tests (unit) check that adapter classes exist and return
AgentResult. Behavioral tests that call real agents are in the behavioral
test directory.

Marker: unit
"""

import pytest

from Supervisor.agents.base import AgentAdapter, AgentResult


@pytest.mark.unit
class TestAgentAdapterInterface:
    """Validate that all adapters implement the AgentAdapter interface."""

    @pytest.mark.parametrize(
        "adapter_path,adapter_class",
        [
            ("Supervisor.agents.summarize_adapter", "SummarizeAdapter"),
            ("Supervisor.agents.civil_law_rag_adapter", "CivilLawRAGAdapter"),
            ("Supervisor.agents.case_doc_rag_adapter", "CaseDocRAGAdapter"),
            ("Supervisor.agents.case_reasoner_adapter", "CaseReasonerAdapter"),
            ("Supervisor.agents.ocr_adapter", "OCRAdapter"),
        ],
    )
    def test_adapter_is_subclass_of_agent_adapter(self, adapter_path, adapter_class):
        import importlib

        module = importlib.import_module(adapter_path)
        cls = getattr(module, adapter_class)
        assert issubclass(cls, AgentAdapter), (
            f"{adapter_class} must be a subclass of AgentAdapter"
        )

    @pytest.mark.parametrize(
        "adapter_path,adapter_class",
        [
            ("Supervisor.agents.summarize_adapter", "SummarizeAdapter"),
            ("Supervisor.agents.civil_law_rag_adapter", "CivilLawRAGAdapter"),
            ("Supervisor.agents.case_doc_rag_adapter", "CaseDocRAGAdapter"),
            ("Supervisor.agents.case_reasoner_adapter", "CaseReasonerAdapter"),
            ("Supervisor.agents.ocr_adapter", "OCRAdapter"),
        ],
    )
    def test_adapter_has_invoke_method(self, adapter_path, adapter_class):
        import importlib

        module = importlib.import_module(adapter_path)
        cls = getattr(module, adapter_class)
        assert hasattr(cls, "invoke"), (
            f"{adapter_class} must implement invoke() method"
        )
        assert callable(getattr(cls, "invoke")), (
            f"{adapter_class}.invoke must be callable"
        )


@pytest.mark.unit
class TestAgentResultContract:
    """Validate AgentResult structure and serialization."""

    def test_agent_result_serialization(self):
        result = AgentResult(
            response="Test response",
            sources=["source1", "source2"],
            raw_output={"key": "value"},
            error=None,
        )
        data = result.model_dump()
        assert data["response"] == "Test response", (
            "Serialized response must match input"
        )
        assert len(data["sources"]) == 2, (
            "Serialized sources must preserve list length"
        )

    def test_agent_result_from_dict(self):
        data = {
            "response": "Deserialized response",
            "sources": ["src"],
            "raw_output": {},
            "error": None,
        }
        result = AgentResult(**data)
        assert result.response == "Deserialized response", (
            "AgentResult must be constructable from dict"
        )
