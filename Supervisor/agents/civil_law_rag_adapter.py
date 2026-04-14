"""
civil_law_rag_adapter.py

Adapter for the Civil Law RAG agent (RAG/Civil Law RAG/graph.py).

Performance fix
---------------
The original implementation evicted all RAG-owned modules from
``sys.modules`` on *every* invocation so that Python would re-import them
from the RAG directory.  This had two severe consequences:

1. The compiled LangGraph ``app`` was rebuilt from scratch on every call
   (graph compilation is expensive -- it re-resolves all node references,
   re-validates edges, and re-creates the state machine).

2. Any in-process caches inside the RAG package (e.g. module-level LRU
   caches, singleton clients) were destroyed and recreated each time,
   making every request behave like a cold start even after warm-up.

Fix: do the path manipulation and import *once*, cache the resulting
``app`` and ``default_state_template`` as class-level attributes, and
reuse them on every subsequent call.  The module-name collision problem
(RAG ``config`` vs API ``config``) is solved at import time by temporarily
front-loading the RAG directory onto sys.path -- we still do this, but
only once, and we no longer delete modules that are already cached.
"""

import logging

from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult
import sys, os

RAG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "RAG", "Civil Law RAG")
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

for p in (PROJECT_ROOT, RAG_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

logger = logging.getLogger(__name__)


def _load_rag_app():
    """Import and return (app, default_state_template) from the Civil Law RAG.

    Called exactly once; result is stored on the class.  Uses a targeted
    sys.path manipulation so that the RAG's own ``config``, ``graph``, etc.
    resolve correctly without colliding with the API-level packages.
    """
    rag_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "RAG", "Civil Law RAG",
    ))

    # Put RAG dir at the front of sys.path so its modules shadow any
    # same-named packages that are already on the path.
    if rag_dir in sys.path:
        sys.path.remove(rag_dir)
    sys.path.insert(0, rag_dir)

    # Only evict modules that haven't been imported yet from the correct
    # location.  We check whether the module's __file__ (if set) lives
    # inside rag_dir; if it does, it was already loaded from the right
    # place and we leave it alone.  Only evict modules whose origin is
    # *outside* rag_dir (i.e. the API's own config/graph/etc.).
    _RAG_MODULES = ("config", "graph", "nodes", "state", "edges", "utils")
    for mod_name in list(sys.modules.keys()):
        if mod_name not in _RAG_MODULES and not any(
            mod_name.startswith(f"{m}.") for m in _RAG_MODULES
        ):
            continue
        mod = sys.modules[mod_name]
        mod_file = getattr(mod, "__file__", None) or ""
        if not mod_file.startswith(rag_dir):
            # This copy came from somewhere other than rag_dir -- evict it
            # so Python re-imports from rag_dir on the next import statement.
            del sys.modules[mod_name]

    from dotenv import load_dotenv
    load_dotenv()

    from graph import app  # noqa: E402  (resolved from rag_dir)
    from config.rag import default_state_template  # noqa: E402

    return app, default_state_template


class CivilLawRAGAdapter(AgentAdapter):
    """Thin wrapper around the Civil Law RAG LangGraph workflow.

    The compiled graph (``_app``) and default state template
    (``_default_state_template``) are loaded once and reused across all
    invocations, avoiding the per-call compilation overhead that was the
    primary cause of the cache-speedup test failure.
    """

    _app = None
    _default_state_template = None

    @classmethod
    def _get_app(cls):
        """Return the cached (app, template) pair, loading on first call."""
        if cls._app is None:
            logger.info("Loading Civil Law RAG graph (first call)...")
            cls._app, cls._default_state_template = _load_rag_app()
            logger.info("Civil Law RAG graph loaded and cached.")
        return cls._app, cls._default_state_template

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        try:
            app, default_state_template = self._get_app()

            # Build initial state from the template
            state = dict(default_state_template)
            state["last_query"] = query

            result = app.invoke(state)

            final_answer = result.get("final_answer", "")
            last_results = result.get("last_results", [])

            sources = []
            for doc in last_results:
                if hasattr(doc, "metadata"):
                    meta = doc.metadata
                    ref = meta.get("article", meta.get("source", ""))
                    if ref:
                        sources.append(str(ref))

            return AgentResult(
                response=final_answer or "",
                sources=sources,
                raw_output={
                    "final_answer": final_answer,
                    "classification": result.get("classification"),
                    "retrieval_confidence": result.get("retrieval_confidence"),
                },
            )

        except Exception as exc:
            error_msg = f"Civil Law RAG adapter error: {exc}"
            logger.exception(error_msg)
            # Reset cached app so the next call retries the import in case
            # the failure was due to a transient initialisation error.
            CivilLawRAGAdapter._app = None
            CivilLawRAGAdapter._default_state_template = None
            return AgentResult(response="", error=error_msg)