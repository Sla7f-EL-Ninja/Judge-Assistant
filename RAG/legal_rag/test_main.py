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
    python RAG/legal_rag/test_main.py --query "ما هي شروط صحة العقد؟" --corpus civil
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
import time

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

def _h(text: str) -> str:   return f"{BOLD}{CYAN}{text}{RESET}"
def _ok(text: str) -> str:  return f"{GREEN}{text}{RESET}"
def _warn(text: str) -> str:return f"{YELLOW}{text}{RESET}"
def _err(text: str) -> str: return f"{RED}{text}{RESET}"
def _sep() -> None:          print("\n" + "─" * 70 + "\n")


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
            print("Invalid corpus name for indexing check: must be 'civil', 'evidence', or 'procedures'.")
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

def run_query(query: str, corpus_name: str) -> None:
    print(_h(f"\n[{corpus_name}] Query:"), query)

    try:
        if corpus_name == "civil":
            from RAG.legal_rag.civil_law_rag import ask_question
        elif corpus_name == "evidence":
            from RAG.legal_rag.evidence_rag import ask_question
        elif corpus_name == "procedures":
            from RAG.legal_rag.procedures_rag import ask_question
        else:
            print("Invalid corpus name for query: must be 'civil', 'evidence', or 'procedures'.")
            return

        t0     = time.perf_counter()
        result = ask_question(query)
        elapsed = time.perf_counter() - t0

        # ── result summary ────────────────────────────────────────────────
        print(f"  {'From cache:':<22} {_ok('yes') if result.from_cache else 'no'}")
        print(f"  {'Classification:':<22} {result.classification or '—'}")
        print(f"  {'Retrieval confidence:':<22} {result.retrieval_confidence or '—'}")
        print(f"  {'Citation integrity:':<22} {result.citation_integrity or '—'}")
        print(f"  {'Sources retrieved:':<22} {len(result.sources)}")
        print(f"  {'Time:':<22} {elapsed:.2f}s")

        if result.sources:
            indices = [str(s['article']) for s in result.sources[:5]]
            print(f"  {'Top article indices:':<22} {', '.join(indices)}")

        print(f"\n  {BOLD}Answer:{RESET}")
        # print answer wrapped at 70 chars
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
    label = "Civil Law (القانون المدني)" if corpus_name == "civil" \
            else "Evidence Law (قانون الإثبات)" if corpus_name == "evidence" \
            else "Procedures Law (قانون الإجراءات)"
    print(_h(f"══ Testing corpus: {label} ══"))

    if not check_and_index(corpus_name):
        print(_warn(f"  Skipping query tests — corpus not ready."))
        return

    queries = [custom_query] if custom_query else (
        CIVIL_QUERIES if corpus_name == "civil" else EVIDENCE_QUERIES if corpus_name == "evidence" else PROCEDURES_QUERIES
    )

    for q in queries:
        print()
        run_query(q, corpus_name)

    # ── cache hit test (repeat first query) ──────────────────────────────
    if not custom_query and len(queries) > 0:
        print(_h("\n  [Cache hit test] Repeating first query..."))
        run_query(queries[0], corpus_name)


# ── off-topic / validation edge cases ────────────────────────────────────────

def test_edge_cases(corpus_name: str) -> None:
    _sep()
    print(_h(f"══ Edge case tests ({corpus_name}) ══"))

    edge_cases = [
        ("Off-topic",            "ما هو أفضل مطعم في القاهرة؟"),
        ("Non-Arabic",           "What is the law on contracts?"),
        ("Too short",            "عقد"),
        ("Textual / range",      "ما نص المواد من 89 إلى 92؟"),
    ]

    for label, query in edge_cases:
        print(f"\n  {BOLD}[{label}]{RESET} {query}")
        try:
            if corpus_name == "civil":
                from RAG.legal_rag.civil_law_rag import ask_question
            elif corpus_name == "evidence":
                from RAG.legal_rag.evidence_rag import ask_question
            elif corpus_name == "procedures":
                from RAG.legal_rag.procedures_rag import ask_question
            else:
                print("Invalid corpus name for edge case test: must be 'civil', 'evidence', or 'procedures'.")
                return
            result = ask_question(query)
            short  = (result.answer or "")[:120].replace("\n", " ")
            print(f"    classification : {result.classification}")
            print(f"    answer snippet : {short}{'…' if len(result.answer or '') > 120 else ''}")
        except Exception as e:
            print(f"    {_ok('Raised exception (expected):')} {e}")


# ── graph build smoke test ────────────────────────────────────────────────────

def test_graph_build(corpus_name: str) -> None:
    _sep()
    print(_h(f"══ Graph build test ({corpus_name}) ══"))
    try:
        if corpus_name == "civil":
            from RAG.legal_rag.civil_law_rag import build_graph
        elif corpus_name == "evidence":
            from RAG.legal_rag.evidence_rag import build_graph
        elif corpus_name == "procedures":
            from RAG.legal_rag.procedures_rag import build_graph
        else:
            print("Invalid corpus name for graph build test: must be 'civil', 'evidence', or 'procedures'.")
            return

        g = build_graph()
        print(_ok(f"  ✓ Graph compiled successfully: {type(g).__name__}"))
    except Exception as e:
        print(_err(f"  ✗ Graph build failed: {e}"))


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the legal_rag engine.")
    parser.add_argument(
        "--corpus", choices=["civil", "evidence", "procedures", "all"], default="civil",
        help="Which corpus to test (default: civil)",
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
    print(_h("══════════════════════════════════════════"))

    for corpus in corpora:
        test_graph_build(corpus)
        test_corpus(corpus, custom_query=args.query)
        if not args.skip_edge_cases and not args.query:
            test_edge_cases(corpus)

    _sep()
    print(_ok("Done."))


if __name__ == "__main__":
    main()
