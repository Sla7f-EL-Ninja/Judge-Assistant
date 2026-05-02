"""
test_main.py
------------
Manual smoke-test for the legal_rag engine.

Run from the project root:
    python RAG/legal_rag/test_main.py
    python RAG/legal_rag/test_main.py --corpus evidence
    python RAG/legal_rag/test_main.py --corpus civil
    python RAG/legal_rag/test_main.py --corpus procedures
    python RAG/legal_rag/test_main.py --corpus all
    python RAG/legal_rag/test_main.py --query "ما هي شروط صحة العقد؟"

Note: --corpus now controls which test-question suite to run.
      Corpus routing is handled automatically by the unified graph.
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
import time
import pytest

# ── make sure project root is on the path ────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))  # .../RAG/legal_rag
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))     # project root
sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _h(text: str) -> str:    return f"{BOLD}{CYAN}{text}{RESET}"
def _ok(text: str) -> str:   return f"{GREEN}{text}{RESET}"
def _warn(text: str) -> str: return f"{YELLOW}{text}{RESET}"
def _err(text: str) -> str:  return f"{RED}{text}{RESET}"
def _sep() -> None:           print("\n" + "─" * 70 + "\n")


# ── default test queries ──────────────────────────────────────────────────────
CIVIL_QUERIES = [
    "ما هي شروط صحة العقد؟",
    "ما نص المادة 89 من القانون المدني؟",
    "ما هي أحكام التعويض عن الضرر؟",
]

EVIDENCE_QUERIES = [
    "ما هي طرق الإثبات في المواد المدنية؟",
    "ما نص المادة 1 من قانون الإثبات؟",
    "متى يجوز الإثبات بالبينة الشهادة؟",
]

PROCEDURES_QUERIES = [
    "ما هي أنواع الدعاوى المدنية؟",
    "ما نص المادة 10 من قانون المرافعات؟",
    "ما هي شروط رفع الدعوى؟",
]


# ── indexing check ────────────────────────────────────────────────────────────

def check_and_index(corpus_name: str) -> bool:
    """Returns True if corpus is ready, False if indexing failed."""
    print(_h(f"[{corpus_name}] Checking index..."))
    try:
        if corpus_name == "civil":
            from RAG.legal_rag.civil_law_rag import ensure_indexed
        elif corpus_name == "evidence":
            from RAG.legal_rag.evidence_rag import ensure_indexed
        elif corpus_name == "procedures":
            from RAG.legal_rag.procedures_rag import ensure_indexed
        else:
            print("Invalid corpus name: must be 'civil', 'evidence', or 'procedures'.")
            return False

        ensure_indexed()
        print(_ok(f"  ✓ {corpus_name} corpus is indexed and ready."))
        return True
    except FileNotFoundError as e:
        print(_err(f"  ✗ Source file not found: {e}"))
        print(_warn("    Place the tagged .txt file in the corpus docs/ folder and retry."))
        return False
    except Exception as e:
        print(_err(f"  ✗ Indexing error: {e}"))
        return False


# ── single query runner ───────────────────────────────────────────────────────

def run_query(query: str) -> None:
    """Run a single query through the unified graph and print results."""
    from RAG.legal_rag.service import ask_question

    print(_h(f"\n[unified graph] Query:"), query)

    try:
        t0      = time.perf_counter()
        result  = ask_question(query)
        elapsed = time.perf_counter() - t0

        # ── result summary ────────────────────────────────────────────────
        print(f"  {'From cache:':<26} {_ok('yes') if result.from_cache else 'no'}")
        print(f"  {'Resolved corpus:':<26} {_ok(result.corpus) if result.corpus else _warn('— not resolved —')}")
        print(f"  {'Classification:':<26} {result.classification or '—'}")
        print(f"  {'Retrieval confidence:':<26} {result.retrieval_confidence or '—'}")
        print(f"  {'Citation integrity:':<26} {result.citation_integrity or '—'}")
        print(f"  {'Sources retrieved:':<26} {len(result.sources)}")
        print(f"  {'Time:':<26} {elapsed:.2f}s")

        # ── corpus routing scores (observability) ─────────────────────────
        if result.corpus_routing_scores:
            print(f"  {'Corpus routing scores:':<26}")
            for entry in result.corpus_routing_scores:
                bar   = "█" * int(entry.get("confidence", 0) * 20)
                print(
                    f"    {entry.get('corpus_name', '?'):<12} "
                    f"{entry.get('confidence', 0):.2f}  {bar}"
                    f"  {entry.get('reason', '')}"
                )

        if result.sources:
            indices = [str(s["article"]) for s in result.sources[:5]]
            print(f"  {'Top article indices:':<26} {', '.join(indices)}")

        print(f"\n  {BOLD}Answer:{RESET}")
        answer = result.answer or "— no answer —"
        for line in answer.splitlines():
            print(f"    {line}")

    except Exception as e:
        print(_err(f"  ✗ ERROR: {e}"))
        import traceback
        traceback.print_exc()


# ── corpus test suite ─────────────────────────────────────────────────────────
def test_corpus(corpus_name: str, custom_query: str | None = None) -> None:
    _sep()
    label = (
        "Civil Law (القانون المدني)"       if corpus_name == "civil"
        else "Evidence Law (قانون الإثبات)" if corpus_name == "evidence"
        else "Procedures Law (قانون المرافعات)"
    )
    print(_h(f"══ Testing corpus suite: {label} ══"))
    print(_warn("  (corpus routing is automatic — --corpus only selects test questions)"))

    if not check_and_index(corpus_name):
        print(_warn("  Skipping query tests — corpus not ready."))
        return

    queries = [custom_query] if custom_query else (
        CIVIL_QUERIES      if corpus_name == "civil"
        else EVIDENCE_QUERIES  if corpus_name == "evidence"
        else PROCEDURES_QUERIES
    )

    for q in queries:
        print()
        run_query(q)

    # ── cache hit test (repeat first query) ──────────────────────────────
    if not custom_query and queries:
        print(_h("\n  [Cache hit test] Repeating first query..."))
        run_query(queries[0])


# ── off-topic / validation edge cases ────────────────────────────────────────

def test_edge_cases() -> None:
    _sep()
    print(_h("══ Edge case tests (unified graph) ══"))

    edge_cases = [
        ("Off-topic",       "ما هو أفضل مطعم في القاهرة؟"),
        ("Non-Arabic",      "What is the law on contracts?"),
        ("Too short",       "عقد"),
        ("Textual / range", "ما نص المواد من 89 إلى 92؟"),
        ("Cross-corpus",    "هل يجوز الإثبات بالشهادة في عقد تجاوز قيمته عشرة آلاف جنيه؟"),
    ]

    from RAG.legal_rag.service import ask_question

    for label, query in edge_cases:
        print(f"\n  {BOLD}[{label}]{RESET} {query}")
        try:
            result = ask_question(query)
            short  = (result.answer or "")[:120].replace("\n", " ")
            print(f"    resolved corpus  : {result.corpus or '—'}")
            print(f"    classification   : {result.classification or '—'}")
            print(f"    answer snippet   : {short}{'…' if len(result.answer or '') > 120 else ''}")
            if result.corpus_routing_scores:
                scores_str = ", ".join(
                    f"{e['corpus_name']}={e['confidence']:.2f}"
                    for e in result.corpus_routing_scores
                )
                print(f"    routing scores   : {scores_str}")
        except Exception as e:
            print(f"    {_ok('Raised exception (expected):')} {e}")


# ── graph build smoke test ────────────────────────────────────────────────────

def test_graph_build() -> None:
    """The graph is now unified — one build, no corpus argument."""
    _sep()
    print(_h("══ Graph build test (unified graph) ══"))
    try:
        from RAG.legal_rag.graph import build_graph
        g = build_graph()
        print(_ok(f"  ✓ Unified graph compiled successfully: {type(g).__name__}"))
    except Exception as e:
        print(_err(f"  ✗ Graph build failed: {e}"))


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the legal_rag engine.")
    parser.add_argument(
        "--corpus",
        choices=["civil", "evidence", "procedures", "all"],
        default="civil",
        help=(
            "Which question suite to run (default: civil). "
            "Corpus routing is automatic — this only selects test questions."
        ),
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a single custom Arabic query instead of the default suite.",
    )
    parser.add_argument(
        "--skip-edge-cases", action="store_true",
        help="Skip the edge-case tests.",
    )
    args = parser.parse_args()

    corpora = ["civil", "evidence", "procedures"] if args.corpus == "all" else [args.corpus]

    print(_h("\n══════════════════════════════════════════"))
    print(_h("   legal_rag engine — smoke test"))
    print(_h("   (unified graph / automatic corpus routing)"))
    print(_h("══════════════════════════════════════════"))

    # Graph build is now a single test, not per-corpus.
    test_graph_build()

    for corpus in corpora:
        test_corpus(corpus, custom_query=args.query)

    if not args.skip_edge_cases and not args.query:
        test_edge_cases()

    _sep()
    print(_ok("Done."))


if __name__ == "__main__":
    main()
