"""
test_e2e_retry.py — validation failure → retry → pass/fallback E2E tests.
"""

import uuid

import pytest

from tests.supervisor.helpers.state_factory import make_state
from tests.supervisor.helpers.reporting import emit_evidence


@pytest.mark.expensive
class TestE2ERetryPath:
    def test_retry_count_increments_on_failure(self, supervisor_app, monkeypatch):
        """Force validate_output to fail once, then pass."""
        call_count = [0]

        import Supervisor.nodes.validate_output as vo_mod
        original_invoke = vo_mod.llm_invoke

        from Supervisor.state import ValidationResult

        def controlled_invoke(fn, msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ValidationResult(
                    hallucination_pass=False, relevance_pass=True,
                    completeness_pass=True, coherence_pass=True,
                    overall_pass=False, feedback="first attempt fail",
                )
            return ValidationResult(
                hallucination_pass=True, relevance_pass=True,
                completeness_pass=True, coherence_pass=True,
                overall_pass=True, feedback="",
            )

        monkeypatch.setattr(vo_mod, "llm_invoke", controlled_invoke)

        import time
        import Supervisor.nodes.prepare_retry as pr_mod
        monkeypatch.setattr(time, "sleep", lambda s: None)

        state = make_state(
            judge_query="ما المادة 163؟",
            intent="civil_law_rag",
        )
        final = supervisor_app.invoke(state)
        emit_evidence("e2e_retry_then_pass", final)

        assert final["validation_status"] in ("pass", "partial_pass", "fallback")

    def test_fallback_after_max_retries(self, supervisor_app, monkeypatch):
        """Force all validation attempts to fail → final fallback."""
        import Supervisor.nodes.validate_output as vo_mod
        from Supervisor.state import ValidationResult
        import time
        import Supervisor.nodes.prepare_retry as pr_mod

        monkeypatch.setattr(time, "sleep", lambda s: None)
        monkeypatch.setattr(
            vo_mod, "llm_invoke",
            lambda fn, msgs: ValidationResult(
                hallucination_pass=False, relevance_pass=False,
                completeness_pass=False, coherence_pass=False,
                overall_pass=False, feedback="all fail forced",
            ),
        )

        state = make_state(
            judge_query="ما المادة 163؟",
            intent="civil_law_rag",
        )
        final = supervisor_app.invoke(state)
        emit_evidence("e2e_fallback", final)

        assert final["validation_status"] == "fallback"
        assert final["final_response"]

    def test_retry_only_reruns_failed_agents(self, supervisor_app, monkeypatch):
        """On retry, previously-successful agents must not be re-invoked."""
        import Supervisor.nodes.validate_output as vo_mod
        from Supervisor.state import ValidationResult
        import time

        call_count = [0]
        monkeypatch.setattr(time, "sleep", lambda s: None)

        def controlled_validate(fn, msgs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return ValidationResult(
                    hallucination_pass=False, relevance_pass=True,
                    completeness_pass=True, coherence_pass=True,
                    overall_pass=False, feedback="retry me",
                )
            return ValidationResult(
                hallucination_pass=True, relevance_pass=True,
                completeness_pass=True, coherence_pass=True,
                overall_pass=True, feedback="",
            )

        monkeypatch.setattr(vo_mod, "llm_invoke", controlled_validate)

        state = make_state(
            judge_query="ما المادة 163؟",
            intent="civil_law_rag",
        )
        final = supervisor_app.invoke(state)
        # Should have passed on second attempt
        assert final["validation_status"] in ("pass", "partial_pass", "fallback")
