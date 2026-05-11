"""
Integration tests for DocumentProcessor.classifier using real fixture files.

Requires a live LLM API key. Skipped automatically if GOOGLE_API_KEY is absent.
Run:  pytest tests/integration/test_classifier_real.py -v -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from DocumentProcessor.classifier import classify_document
from config.taxonomy import get_doc_types, get_unknown_label

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent.parent / "CASE_RAG" / "fixtures"
MIN_CONFIDENCE = 60


def _read(filename: str) -> str:
    return (FIXTURES / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixture → expected type mapping
# ---------------------------------------------------------------------------

CASES = [
    ("صحيفة_دعوى.txt",                        "صحيفة دعوى"),
    ("محضر_جلسة_25_03_2024.txt",               "محضر جلسة"),
    ("مذكرة_بدفاع_المدعى_عليه_الأول.txt",     "مذكرة بدفاع"),
    ("مذكرة_بدفاع_المدعى_عليها_الثانية.txt",  "مذكرة بدفاع"),
    ("تقرير_الخبير.txt",                       "تقرير خبير"),
    # Forensic report — no "تقرير الخبير" header; heuristic will miss it → LLM path
    ("تقرير_الطب_الشرعي.txt",                  "تقرير خبير"),
]


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def require_api_key():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set — skipping integration tests")


# ---------------------------------------------------------------------------
# Per-file correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected_type", CASES)
def test_classifies_correctly(filename: str, expected_type: str):
    text = _read(filename)
    result = classify_document(text)

    assert result["final_type"] == expected_type, (
        f"{filename}: expected '{expected_type}', got '{result['final_type']}'\n"
        f"confidence={result['confidence']}, explanation={result['explanation']}"
    )
    assert result["confidence"] >= MIN_CONFIDENCE, (
        f"{filename}: confidence {result['confidence']} below threshold {MIN_CONFIDENCE}"
    )


# ---------------------------------------------------------------------------
# Return shape invariant — must hold on every real file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,_", CASES)
def test_return_shape(filename: str, _: str):
    result = classify_document(_read(filename))
    assert set(result.keys()) >= {"final_type", "confidence", "explanation"}
    assert isinstance(result["confidence"], int)
    assert result["final_type"] in get_doc_types() + [get_unknown_label()]


# ---------------------------------------------------------------------------
# Forensic report specifically exercises the LLM path
# (its header won't match "تقرير الخبير" strong keyword)
# ---------------------------------------------------------------------------

def test_forensic_report_hits_llm_path(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="DocumentProcessor.classifier"):
        classify_document(_read("تقرير_الطب_الشرعي.txt"))

    paths_logged = [r.message for r in caplog.records if "classify" in r.message]
    assert any("path=llm" in m for m in paths_logged), (
        "Expected forensic report to route through LLM path — check strong keyword coverage"
    )


# ---------------------------------------------------------------------------
# Dual-defence memos must both classify as مذكرة بدفاع (not just one)
# ---------------------------------------------------------------------------

def test_both_defence_memos_classified_consistently():
    r1 = classify_document(_read("مذكرة_بدفاع_المدعى_عليه_الأول.txt"))
    r2 = classify_document(_read("مذكرة_بدفاع_المدعى_عليها_الثانية.txt"))
    assert r1["final_type"] == r2["final_type"] == "مذكرة بدفاع"
