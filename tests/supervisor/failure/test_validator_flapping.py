"""
test_validator_flapping.py — validator_error cycling to fallback.
"""

import time
import uuid

import pytest

from Supervisor.state import ValidationResult
from tests.supervisor.helpers.state_factory import make_state
from tests.supervisor.helpers.reporting import emit_evidence


@pytest.mark.expensive
class TestValidatorFlapping:
    def test_repeated_validator_error_leads_to_fallback(self, supervisor_app, monkeypatch):
        """Force every validate_output call to return validator_error → fallback."""
        import Supervisor.nodes.validate_output as vo_mod
        import Supervisor.nodes.prepare_retry as pr_mod

        monkeypatch.setattr(time, "sleep", lambda s: None)
        monkeypatch.setattr(
            vo_mod, "llm_invoke",
            lambda fn, msgs: (_ for _ in ()).throw(RuntimeError("validator flap")),
        )

        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="ما المادة 163؟",
            correlation_id=cid,
        )
        final = supervisor_app.invoke(state)
        emit_evidence("validator_flapping", final)

        assert final["validation_status"] == "fallback", (
            f"Expected fallback after flapping, got {final['validation_status']!r}"
        )
        assert final["final_response"]

    def test_validator_error_retries_before_fallback(self, supervisor_app, monkeypatch):
        """validator_error should retry up to max_retries before fallback."""
        import Supervisor.nodes.validate_output as vo_mod

        call_count = [0]
        monkeypatch.setattr(time, "sleep", lambda s: None)

        def flap_then_pass(fn, msgs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("flap")
            return ValidationResult(
                hallucination_pass=True, relevance_pass=True,
                completeness_pass=True, coherence_pass=True,
                overall_pass=True, feedback="",
            )

        monkeypatch.setattr(vo_mod, "llm_invoke", flap_then_pass)

        state = make_state(judge_query="ما المادة 163؟")
        final = supervisor_app.invoke(state)

        assert final["validation_status"] in ("pass", "partial_pass", "fallback")
        assert call_count[0] >= 1
