"""
Run all four supervisor memory tests in order.
Prints PASS/FAIL per test with reason. Full traceback on failure.
"""
import subprocess
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

TESTS = [
    ("TEST 1 — Short-term summarization",    "tests/supervisor/memory/test_01_short_term.py"),
    ("TEST 2 — Cross-session long-term",     "tests/supervisor/memory/test_02_long_term.py"),
    ("TEST 3 — Crash safety",                "tests/supervisor/memory/test_03_crash_safety.py"),
    ("TEST 4 — Episodic & procedural",       "tests/supervisor/memory/test_04_episodic_procedural.py"),
]

results = []
for name, path in TESTS:
    abs_path = os.path.join(ROOT, path)
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    proc = subprocess.run(
        [sys.executable, abs_path],
        capture_output=False,  # stream output live
        cwd=ROOT,
    )
    passed = proc.returncode == 0
    results.append((name, passed))

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
all_pass = True
for name, passed in results:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  {status}  {name}")

sys.exit(0 if all_pass else 1)
