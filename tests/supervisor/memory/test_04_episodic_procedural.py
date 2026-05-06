"""
TEST 4 — Episodic & procedural memory reflection
Run one full validated session, then trigger reflection immediately (after=0).
Check store for semantic facts (sync), episodic, and procedural entries.

Actual namespaces:
  semantic:   ("case", case_id, "facts")
  episodic:   ("case", case_id, "episodes")
  procedural: ("user", user_id, "prefs")

Fix applied
-----------
time.sleep(5) was a hard race against LLM reflection threads that take 5-15 s.
Replaced with a poll loop (2 s intervals, 30 s ceiling) that exits as soon as
both episodic and procedural namespaces are non-empty, or times out and reports
whatever was found.
"""
import sys
import os
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)

# Patch delay to 0 before any Supervisor imports
import config.supervisor as sup_cfg
sup_cfg.EPISODIC_REFLECT_DELAY_S = 0

CASE_ID = "test_case_ep_001"
USER_ID = "test_judge_ep_001"
SESSION_ID = "session_EP_v2"

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
    "correlation_id": "test-t4",
    "classification_error": None,
    "conversation_history": [],
    "turn_count": 0,
    "running_summary": None,
    "semantic_facts": [],
    "procedural_prefs": None,
    "session_id": SESSION_ID,
}

_POLL_INTERVAL_S = 2
_POLL_TIMEOUT_S  = 30   # LLM reflection easily takes 5-15 s; 30 s is safe


def _wait_for_reflection(store, case_id: str, user_id: str) -> tuple:
    """Poll store until both episodic and procedural are non-empty, or timeout."""
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        episodes = store.search(("case", case_id, "episodes"), limit=20)
        prefs    = store.search(("user", user_id,  "prefs"),   limit=20)
        if episodes and prefs:
            return episodes, prefs
        remaining = deadline - time.monotonic()
        print(f"[T4] Polling — episodic={len(episodes)}, procedural={len(prefs)} "
              f"(≤{int(remaining)}s left)...")
        time.sleep(_POLL_INTERVAL_S)
    # Final read after timeout
    return (
        store.search(("case", case_id, "episodes"), limit=20),
        store.search(("user", user_id,  "prefs"),   limit=20),
    )


def run():
    from Supervisor.graph import get_app_persistent
    from Supervisor.services.memory import (
        get_store,
        get_reflection_executor,
        get_episodic_manager,
        get_procedural_manager,
    )

    app = get_app_persistent()
    thread_id = f"{USER_ID}:{CASE_ID}:{SESSION_ID}"
    config = {"configurable": {"thread_id": thread_id, "user_id": USER_ID}}

    print(f"[T4] thread_id={thread_id}")

    # Run one full session
    state = dict(BASE_STATE)
    state["judge_query"] = "ما حكم بيع شقة في القاهرة عام 2019 في القانون المدني؟"
    print("[T4] Running full session...")
    result = app.invoke(state, config=config)
    validation     = result.get("validation_status", "")
    final_response = result.get("final_response", "") or ""
    print(f"[T4] validation_status={validation}, response length={len(final_response)}")

    messages = [
        {"role": "user",      "content": state["judge_query"]},
        {"role": "assistant", "content": final_response},
    ]

    # Trigger reflection immediately (after=0)
    print("[T4] Triggering episodic + procedural reflection with after=0...")
    executor = get_reflection_executor()

    try:
        executor.schedule(get_episodic_manager(CASE_ID),   {"messages": messages}, after=0)
    except Exception as exc:
        print(f"[T4] episodic schedule error: {exc}")

    try:
        executor.schedule(get_procedural_manager(USER_ID), {"messages": messages}, after=0)
    except Exception as exc:
        print(f"[T4] procedural schedule error: {exc}")

    # Poll instead of fixed sleep — exits early once both namespaces are populated
    store    = get_store()
    semantic = store.search(("case", CASE_ID, "facts"), limit=20)

    print(f"[T4] Polling for reflection (up to {_POLL_TIMEOUT_S}s)...")
    episodes, prefs = _wait_for_reflection(store, CASE_ID, USER_ID)

    print(f"[T4] semantic facts:   {len(semantic)}")
    print(f"[T4] episodic entries: {len(episodes)}")
    print(f"[T4] procedural prefs: {len(prefs)}")

    c_semantic   = len(semantic)   > 0
    c_episodic   = len(episodes)   > 0
    c_procedural = len(prefs)      > 0

    if c_semantic and c_episodic and c_procedural:
        print("PASS: all three namespaces non-empty after reflection.")
        return True

    # Partial pass — semantic is sync, episodic/procedural are async
    if c_semantic:
        skipped = []
        if not c_episodic:
            skipped.append("episodic")
        if not c_procedural:
            skipped.append("procedural")

        from Supervisor.services.memory import _NoopReflectionExecutor
        if isinstance(executor, _NoopReflectionExecutor):
            print(f"PASS (partial): semantic facts stored. "
                  f"Episodic/procedural SKIP — ReflectionExecutor is noop (langmem unavailable).")
            return True
        else:
            print(f"FAIL: semantic OK but {skipped} empty after {_POLL_TIMEOUT_S}s — "
                  f"reflection did not complete in time.")
            return False
    else:
        reasons = []
        if not c_semantic:
            reasons.append("semantic facts empty (write_long_term_memory did not write, "
                           "likely validation_status not pass)")
        if not c_episodic:
            reasons.append("episodic empty")
        if not c_procedural:
            reasons.append("procedural empty")
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