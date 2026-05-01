"""
TEST 2 — Cross-session semantic long-term memory
Session A establishes a fact. Session B (new thread_id, same case_id) recalls it.
Pass if: Session B semantic_facts non-empty, answer references the Session A fact.
"""
import sys
import os
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)

CASE_ID = "test_case_001"
USER_ID = "test_judge_001"
UNIQUE_FACT_KEYWORDS = ["شقة", "2019", "القاهرة"]

BASE_STATE = {
    "judge_query": "",
    "case_id": CASE_ID,
    "user_id": USER_ID,
    "uploaded_files": [],
    "intent": None,
    "target_agents": [],
    "classified_query": None,
    "agent_results": {},
    "agent_errors": {},
    "validation_status": None,
    "validation_feedback": None,
    "retry_count": 0,
    "max_retries": 2,
    "document_classifications": [],
    "merged_response": None,
    "final_response": None,
    "sources": [],
    "case_summary": None,
    "case_doc_titles": [],
    "correlation_id": "test-t2",
    "classification_error": None,
    "conversation_history": [],
    "turn_count": 0,
    "running_summary": None,
    "semantic_facts": [],
    "procedural_prefs": None,
    "session_id": None,
}


def run():
    from Supervisor.graph import get_app_persistent

    app = get_app_persistent()

    # --- Session A ---
    session_a = "session_LT_A_v2"
    thread_a = f"{USER_ID}:{CASE_ID}:{session_a}"
    config_a = {"configurable": {"thread_id": thread_a, "user_id": USER_ID}}

    print(f"[T2] Session A: thread_id={thread_a}")
    state_a = dict(BASE_STATE)
    state_a["session_id"] = session_a
    state_a["judge_query"] = "ما حكم بيع شقة في القاهرة عام 2019 في القانون المدني؟"
    result_a = app.invoke(state_a, config=config_a)
    print(f"[T2] Session A response: {str(result_a.get('final_response', ''))[:100]}")
    print(f"[T2] Session A validation_status: {result_a.get('validation_status')}")

    # Check write_long_term_memory ran via state history
    write_ran = False
    try:
        for hist in app.get_state_history(config_a):
            metadata = getattr(hist, "metadata", {}) or {}
            if "write_long_term_memory" in str(metadata):
                write_ran = True
                break
    except Exception:
        pass
    print(f"[T2] write_long_term_memory in trace: {write_ran}")

    # --- Session B ---
    session_b = "session_LT_B_v2"
    thread_b = f"{USER_ID}:{CASE_ID}:{session_b}"
    config_b = {"configurable": {"thread_id": thread_b, "user_id": USER_ID}}

    print(f"[T2] Session B: thread_id={thread_b}")
    state_b = dict(BASE_STATE)
    state_b["session_id"] = session_b
    state_b["judge_query"] = "ما شروط صحة عقد بيع الشقق في القانون المدني؟"

    result_b = app.invoke(state_b, config=config_b)
    answer_b = result_b.get("final_response", "") or ""
    semantic_facts_b = result_b.get("semantic_facts", []) or []
    print(f"[T2] Session B semantic_facts count: {len(semantic_facts_b)}")
    print(f"[T2] Session B answer: {answer_b[:200]}")

    c1 = len(semantic_facts_b) > 0
    c2 = any(kw in answer_b for kw in UNIQUE_FACT_KEYWORDS)

    if c1 and c2:
        print(f"PASS: Session B loaded {len(semantic_facts_b)} semantic facts and recalled the Session A fact.")
        return True
    else:
        reasons = []
        if not c1:
            reasons.append("Session B semantic_facts empty")
        if not c2:
            reasons.append(f"Session B answer missing keywords {UNIQUE_FACT_KEYWORDS}")
        print(f"FAIL: {'; '.join(reasons)}")
        return False


if __name__ == "__main__":
    try:
        ok = run()
        sys.exit(0 if ok else 1)
    except Exception:
        print("FAIL: exception during test")
        traceback.print_exc()
        sys.exit(1)
