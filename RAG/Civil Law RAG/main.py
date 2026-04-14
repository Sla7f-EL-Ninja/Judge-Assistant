"""
main.py

Entry point for manual system operations.

Purpose:
---------
Allows controlled execution of:
- Law indexing
- Maintenance tasks
- Future administrative utilities

This file is NOT responsible for:
- Running the LangGraph pipeline
- Handling API requests
- Serving a mobile app

It is strictly an operational entry script.

Design Principle:
-----------------
Explicit execution.
Critical operations like indexing should not occur implicitly.
"""
import sys
import os

# Bootstrap: add project root to sys.path so `config` package is resolvable
# when this file is run directly (e.g. python main.py).
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

import io
from indexer import index_civil_law
from graph import app, default_state_template
from nodes import State

# Force stdout/stderr to use UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
def ask_question(query: str, db) -> str:
    """
    Processes a user query through the LangGraph workflow.

    Args:
        query (str): The user's question in Arabic, ideally related to Egyptian Civil Law.
        db: The Qdrant vectorstore instance.

    Returns:
        str: The final answer produced by the system.
    """
    # Initialize a fresh state for this query
    state: State = default_state_template.copy()
    state["last_query"] = query
    state["db"] = db

    # Run the query through the graph
    result_state = app.invoke(state)

    # Return the final answer
    return result_state.get("final_answer", "تعذر الحصول على إجابة.")


if __name__ == "__main__":
    # Ensure vectorstore exists and get its instance
    db = index_civil_law()  # now returns the db

    print("=== Egyptian Civil Law AI Judge Assistant ===\n")
    while True:
        user_input = input("ادخل سؤالك (أو 'خروج' لإنهاء البرنامج): ").strip()
        if user_input.lower() == "خروج":
            print("شكراً لاستخدام النظام. إلى اللقاء!")
            break

        answer = ask_question(user_input, db)  # pass db explicitly
        print("\n💡 الإجابة:\n", answer)
        print("\n" + "-"*50 + "\n")