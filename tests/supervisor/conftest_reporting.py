"""
conftest_reporting.py — pytest session hooks for summary report generation.

Import this from conftest.py or place it alongside conftest.py.
Generates reports/<timestamp>/SUMMARY.md after the test session.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

_REPORTS_DIR = Path(__file__).parent / "reports"
_session_start = time.time()
_evidence_records: List[Dict[str, Any]] = []


def pytest_sessionfinish(session, exitstatus):
    """Generate SUMMARY.md after the test session completes."""
    try:
        _write_summary(session)
    except Exception as exc:
        print(f"\n[reporting] Summary generation failed: {exc}")


def _write_summary(session):
    if os.getenv("SKIP_EVIDENCE"):
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = _REPORTS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Collect evidence files
    all_evidence = []
    for f in run_dir.glob("*.json"):
        try:
            all_evidence.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass

    total = session.testscollected or 0
    passed = getattr(session, "_numPassed", 0)
    failed = getattr(session, "_numFailed", 0)

    duration = time.time() - _session_start

    lines = [
        f"# Supervisor Test Run — {ts}",
        "",
        "## Headline metrics",
        f"- Total tests collected: {total}",
        f"- Duration: {duration:.1f}s",
        f"- Evidence files: {len(all_evidence)}",
        "",
        "## How to run",
        "```bash",
        "# Cheap tier (routing + unit_nodes)",
        "pytest tests/supervisor/routing tests/supervisor/unit_nodes -v",
        "",
        "# Full suite",
        "RUN_EXPENSIVE=1 pytest tests/supervisor -v --tb=short",
        "```",
        "",
        "## Known gaps (xfail)",
        "- G13: PII detection not implemented",
        "- CivilLawRAG `from_cache=True` bypasses error-prefix check",
        "- Concurrent dedup ingestion TOCTOU race",
    ]

    summary_path = run_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
