"""reporting.py — JSON evidence emitter per test."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_REPORTS_DIR = Path(__file__).parent.parent / "reports"
_RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _run_dir() -> Path:
    d = _REPORTS_DIR / _RUN_TS
    d.mkdir(parents=True, exist_ok=True)
    return d


def emit_evidence(
    test_name: str,
    state: Dict[str, Any],
    latency_s: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a JSON evidence record for a test run."""
    if os.getenv("SKIP_EVIDENCE"):
        return

    record = {
        "test": test_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_s": round(latency_s, 3),
        "intent": state.get("intent"),
        "target_agents": state.get("target_agents"),
        "retry_count": state.get("retry_count"),
        "validation_status": state.get("validation_status"),
        "agent_errors": state.get("agent_errors"),
        "sources": state.get("sources", [])[:10],
        "judge_query": state.get("judge_query", "")[:200],
        "final_response_excerpt": (state.get("final_response") or "")[:300],
        "correlation_id": state.get("correlation_id"),
    }
    if extra:
        record.update(extra)

    out = _run_dir() / f"{test_name.replace('/', '_').replace('::', '__')}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
