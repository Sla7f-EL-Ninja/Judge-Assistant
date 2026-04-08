# RAG/Civil Law RAG/config.py -- SHIM: re-export from consolidated config
import sys
import os

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.rag import *  # noqa: F401,F403
from config import get_llm  # noqa: F401
