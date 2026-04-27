"""
memory.py

Singleton factories for the Supervisor memory subsystem.

Provides:
  - get_checkpointer()        — MongoDBSaver for short-term state checkpointing
  - get_store()               — BaseStore backed by MongoDB for long-term memories
  - get_semantic_manager()    — langmem manager for case facts
  - get_episodic_manager()    — langmem manager for session episodes
  - get_procedural_manager()  — langmem manager for judge preferences
  - get_reflection_executor() — shared ReflectionExecutor for async reflection

All public functions are failure-safe: if the backing service is unavailable
they return a no-op stub and log a warning so the graph continues.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pymongo
from pymongo import MongoClient

from config.supervisor import (
    CHECKPOINT_COLL,
    EPISODIC_REFLECT_DELAY_S,
    MONGO_DB,
    MONGO_URI,
    PROCEDURAL_INJECT_MAX_CHARS,
    SEMANTIC_FACTS_TOP_K,
    STORE_COLL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — libs may not be installed until requirements.txt is updated
# ---------------------------------------------------------------------------

def _import_mongodb_saver():
    from langgraph.checkpoint.mongodb import MongoDBSaver  # noqa: PLC0415
    return MongoDBSaver


def _import_inmemory_saver():
    from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415
    return MemorySaver


def _import_langmem():
    import langmem  # noqa: PLC0415
    return langmem


def _import_base_store():
    from langgraph.store.base import BaseStore, Item, SearchItem  # noqa: PLC0415
    return BaseStore, Item, SearchItem


def _import_inmemory_store():
    from langgraph.store.memory import InMemoryStore  # noqa: PLC0415
    return InMemoryStore


def _get_default_model() -> Any:
    """Lazy-load the medium-tier LLM for use in langmem managers."""
    from config import get_llm  # noqa: PLC0415
    return get_llm("medium")


# ---------------------------------------------------------------------------
# MongoDB-backed BaseStore implementation
# ---------------------------------------------------------------------------

class _MongoMemoryStore:
    """Minimal BaseStore interface backed by a MongoDB collection.

    Implements put / get / search used by langmem managers.
    Namespaces are serialised as underscore-joined strings for Mongo keys.
    """

    def __init__(self, collection: pymongo.collection.Collection) -> None:
        self._col = collection
        self._col.create_index([("namespace", 1), ("key", 1)], unique=True, background=True)

    # ------------------------------------------------------------------
    # Sync helpers (langmem may call sync or async depending on version)
    # ------------------------------------------------------------------

    def _ns(self, namespace: Tuple[str, ...]) -> str:
        return "/".join(namespace)

    def put(self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any]) -> None:
        ns = self._ns(namespace)
        self._col.update_one(
            {"namespace": ns, "key": key},
            {"$set": {"namespace": ns, "key": key, "value": value, "updated_at": time.time()}},
            upsert=True,
        )

    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Dict[str, Any]]:
        doc = self._col.find_one({"namespace": self._ns(namespace), "key": key})
        return doc["value"] if doc else None

    def search(
        self,
        namespace: Tuple[str, ...],
        *,
        query: Optional[str] = None,
        limit: int = SEMANTIC_FACTS_TOP_K,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return up to *limit* items from namespace. No vector search — full scan."""
        docs = (
            self._col.find({"namespace": self._ns(namespace)})
            .skip(offset)
            .limit(limit)
        )
        return [{"key": d["key"], "value": d["value"]} for d in docs]

    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        self._col.delete_one({"namespace": self._ns(namespace), "key": key})

    def list_namespaces(
        self,
        *,
        prefix: Optional[Tuple[str, ...]] = None,
        suffix: Optional[Tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tuple[str, ...]]:
        query: Dict[str, Any] = {}
        if prefix:
            query["namespace"] = {"$regex": f"^{self._ns(prefix)}"}
        raw = self._col.distinct("namespace", query)
        return [tuple(ns.split("/")) for ns in raw[offset : offset + limit]]

    # Async shims — langmem uses async; wrap sync calls in run_in_executor
    async def aput(self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any]) -> None:
        self.put(namespace, key, value)

    async def aget(self, namespace: Tuple[str, ...], key: str) -> Optional[Dict[str, Any]]:
        return self.get(namespace, key)

    async def asearch(
        self,
        namespace: Tuple[str, ...],
        *,
        query: Optional[str] = None,
        limit: int = SEMANTIC_FACTS_TOP_K,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return self.search(namespace, query=query, limit=limit, offset=offset)

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        self.delete(namespace, key)

    async def alist_namespaces(self, **kwargs: Any) -> List[Tuple[str, ...]]:
        return self.list_namespaces(**kwargs)


# ---------------------------------------------------------------------------
# No-op stubs (returned when backing service is unavailable)
# ---------------------------------------------------------------------------

class _NoopStore:
    """Drop-in store stub used in crash-safety tests only."""

    def put(self, *a: Any, **kw: Any) -> None: ...
    def get(self, *a: Any, **kw: Any) -> None: ...
    def search(self, *a: Any, **kw: Any) -> List: return []
    def delete(self, *a: Any, **kw: Any) -> None: ...
    def list_namespaces(self, *a: Any, **kw: Any) -> List: return []
    async def aput(self, *a: Any, **kw: Any) -> None: ...
    async def aget(self, *a: Any, **kw: Any) -> None: ...
    async def asearch(self, *a: Any, **kw: Any) -> List: return []
    async def adelete(self, *a: Any, **kw: Any) -> None: ...
    async def alist_namespaces(self, *a: Any, **kw: Any) -> List: return []


class _NoopManager:
    """Drop-in manager stub used when langmem is unavailable."""

    def invoke(self, *a: Any, **kw: Any) -> None: ...
    async def ainvoke(self, *a: Any, **kw: Any) -> None: ...


class _NoopReflectionExecutor:
    def schedule(self, *a: Any, **kw: Any) -> None: ...


class _LocalScheduleExecutor:
    """Simple background scheduler replacing langmem.ReflectionExecutor.

    Calls manager.invoke(payload) in a daemon thread after `after` seconds.
    langmem's LocalReflectionExecutor has no .schedule() method — this adapter
    provides the API our nodes expect without depending on langmem's executor.
    """

    def schedule(self, manager: Any, payload: Dict[str, Any], *, after: int = 0) -> None:
        def _run() -> None:
            if after > 0:
                time.sleep(after)
            try:
                manager.invoke(payload)
            except Exception as exc:
                logger.warning("_LocalScheduleExecutor: reflection invoke failed — %s", exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_lock = threading.Lock()

# Sentinel: distinguishes "not yet initialised" from "initialised to fallback"
# Avoids retrying a blocked/failed MongoDB connection on every call.
_UNSET: Any = object()

_mongo_client: Any = _UNSET
_checkpointer: Any = _UNSET
_store: Any = _UNSET
_reflection_executor: Any = _UNSET

_MONGO_TIMEOUT_MS = 3000   # serverSelection + connect + socket
_INIT_TIMEOUT_S   = 5      # hard wall-clock limit for any singleton init


def _get_mongo_client() -> Optional[MongoClient]:
    global _mongo_client
    if _mongo_client is _UNSET:
        with _lock:
            if _mongo_client is _UNSET:
                result: list = []
                exc_box: list = []

                def _connect():
                    try:
                        client = MongoClient(
                            MONGO_URI,
                            serverSelectionTimeoutMS=_MONGO_TIMEOUT_MS,
                            connectTimeoutMS=_MONGO_TIMEOUT_MS,
                            socketTimeoutMS=_MONGO_TIMEOUT_MS,
                        )
                        client.admin.command("ping")
                        result.append(client)
                    except Exception as exc:
                        exc_box.append(exc)

                t = threading.Thread(target=_connect, daemon=True)
                t.start()
                t.join(timeout=_INIT_TIMEOUT_S)
                if result:
                    _mongo_client = result[0]
                    logger.info("memory: MongoDB client connected")
                else:
                    err = exc_box[0] if exc_box else TimeoutError("connection timed out")
                    logger.warning("memory: MongoDB unavailable — %s", err)
                    _mongo_client = None
    return _mongo_client  # type: ignore[return-value]


def get_checkpointer() -> Any:
    """Return MongoDBSaver singleton; MemorySaver fallback on failure."""
    global _checkpointer
    if _checkpointer is _UNSET:
        with _lock:
            if _checkpointer is _UNSET:
                result: list = []
                exc_box: list = []

                def _init():
                    try:
                        MongoDBSaver = _import_mongodb_saver()
                        client = _get_mongo_client()
                        if client is None:
                            raise RuntimeError("MongoDB client unavailable")
                        saver = MongoDBSaver(
                            client,
                            db_name=MONGO_DB,
                            collection_name=CHECKPOINT_COLL,
                        )
                        result.append(saver)
                    except Exception as exc:
                        exc_box.append(exc)

                t = threading.Thread(target=_init, daemon=True)
                t.start()
                t.join(timeout=_INIT_TIMEOUT_S)
                if result:
                    _checkpointer = result[0]
                    logger.info("memory: MongoDBSaver checkpointer ready")
                else:
                    err = exc_box[0] if exc_box else TimeoutError("init timed out")
                    logger.warning("memory: checkpointer init failed — %s; falling back to MemorySaver", err)
                    try:
                        _checkpointer = _import_inmemory_saver()()
                    except Exception as fb_exc:
                        logger.warning("memory: MemorySaver fallback failed — %s; using None", fb_exc)
                        _checkpointer = None
    return _checkpointer


def get_store() -> Any:
    """Return _MongoMemoryStore singleton; _NoopStore fallback on failure."""
    global _store
    if _store is _UNSET:
        with _lock:
            if _store is _UNSET:
                result: list = []
                exc_box: list = []

                def _init():
                    try:
                        client = _get_mongo_client()
                        if client is None:
                            raise RuntimeError("MongoDB client unavailable")
                        col = client[MONGO_DB][STORE_COLL]
                        result.append(_MongoMemoryStore(col))
                    except Exception as exc:
                        exc_box.append(exc)

                t = threading.Thread(target=_init, daemon=True)
                t.start()
                t.join(timeout=_INIT_TIMEOUT_S)
                if result:
                    _store = result[0]
                    logger.info("memory: MongoMemoryStore ready (collection=%s)", STORE_COLL)
                else:
                    err = exc_box[0] if exc_box else TimeoutError("init timed out")
                    logger.warning("memory: store init failed — %s; using InMemoryStore (not persistent)", err)
                    try:
                        _store = _import_inmemory_store()()
                    except Exception as fb_exc:
                        logger.warning("memory: InMemoryStore fallback failed — %s; using NoopStore", fb_exc)
                        _store = _NoopStore()
    return _store


def get_reflection_executor() -> Any:
    """Return shared _LocalScheduleExecutor.

    langmem.ReflectionExecutor (LocalReflectionExecutor) exposes only .submit(),
    not .schedule() — our nodes call .schedule(manager, payload, after=N).
    Use _LocalScheduleExecutor which implements exactly that API.
    """
    global _reflection_executor
    if _reflection_executor is _UNSET:
        with _lock:
            if _reflection_executor is _UNSET:
                _reflection_executor = _LocalScheduleExecutor()
                logger.info("memory: _LocalScheduleExecutor ready")
    return _reflection_executor


# Manager factories — not singletons; keyed by namespace (cheap to recreate)

def get_semantic_manager(case_id: str) -> Any:
    """Return langmem manager for case semantic facts."""
    try:
        langmem = _import_langmem()
        return langmem.create_memory_store_manager(
            _get_default_model(),
            store=get_store(),
            namespace=("case", case_id, "facts"),
        )
    except Exception as exc:
        logger.warning("memory: semantic manager init failed — %s; using noop", exc)
        return _NoopManager()


def get_episodic_manager(case_id: str) -> Any:
    """Return langmem manager for case session episodes."""
    try:
        langmem = _import_langmem()
        return langmem.create_memory_store_manager(
            _get_default_model(),
            store=get_store(),
            namespace=("case", case_id, "episodes"),
        )
    except Exception as exc:
        logger.warning("memory: episodic manager init failed — %s; using noop", exc)
        return _NoopManager()


def get_procedural_manager(user_id: str) -> Any:
    """Return langmem manager for judge behavioral preferences."""
    try:
        langmem = _import_langmem()
        return langmem.create_memory_store_manager(
            _get_default_model(),
            store=get_store(),
            namespace=("user", user_id, "prefs"),
        )
    except Exception as exc:
        logger.warning("memory: procedural manager init failed — %s; using noop", exc)
        return _NoopManager()
