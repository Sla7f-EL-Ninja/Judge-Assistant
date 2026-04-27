"""llm_assertions.py — fuzzy / structural assertions for non-deterministic LLM output."""

import re
from typing import List, Optional, Set

from tests.supervisor.helpers.arabic_text import arabic_ratio, extract_article_numbers


def assert_arabic_response(
    text: str,
    min_len: int = 50,
    min_ratio: float = 0.3,
    label: str = "response",
) -> None:
    assert text, f"{label} is empty"
    assert len(text) >= min_len, f"{label} too short: {len(text)} chars (min {min_len})"
    ratio = arabic_ratio(text)
    assert ratio >= min_ratio, (
        f"{label} Arabic ratio {ratio:.2f} < {min_ratio}: {text[:200]}"
    )


def assert_mentions_entity(text: str, entity: str, label: str = "response") -> None:
    assert entity in text, (
        f"{label} does not mention expected entity {entity!r}.\nGot: {text[:300]}"
    )


def assert_sources_valid(sources: list, label: str = "sources") -> None:
    assert isinstance(sources, list), f"{label} is not a list"
    for src in sources:
        assert isinstance(src, str) and src.strip(), f"Empty/invalid source entry: {src!r}"


def assert_article_sources(sources: list) -> None:
    for src in sources:
        assert re.search(r"المادة\s+\d+", src), (
            f"Source does not match article pattern: {src!r}"
        )


def assert_no_injection_leak(text: str) -> None:
    markers = ["[بداية", "محتوى غير موثوق", "[نهاية", "ignore previous"]
    for m in markers:
        assert m not in text, f"Injection marker leaked into response: {m!r}"


def assert_cited_in_sources(cited: Set[str], sources: List[str], response: str = "") -> None:
    """Every article number cited in response must appear in sources."""
    all_source_text = " ".join(sources) + " " + response
    source_articles = extract_article_numbers(all_source_text)
    leaked = cited - source_articles
    assert not leaked, (
        f"Response cites articles {leaked} not found in sources.\n"
        f"Sources: {sources[:5]}"
    )


def assert_valid_intent(intent: str, valid_intents: Optional[List[str]] = None) -> None:
    if valid_intents is None:
        valid_intents = ["civil_law_rag", "case_doc_rag", "reason", "multi", "off_topic"]
    assert intent in valid_intents, f"Invalid intent: {intent!r}"


def assert_agents_ordered(agents: List[str]) -> None:
    order = {"civil_law_rag": 0, "case_doc_rag": 1, "reason": 2}
    positions = [order.get(a, 99) for a in agents]
    assert positions == sorted(positions), (
        f"Agents not in topological order: {agents}"
    )


def assert_agents_deduped(agents: List[str]) -> None:
    assert len(agents) == len(set(agents)), f"Duplicate agents: {agents}"
