# Troubleshooting

## Known Issues

### 1. sys.modules Cache Collision (Civil Law RAG Adapter)

**Location**: [`Supervisor/agents/civil_law_rag_adapter.py`](../Supervisor/agents/civil_law_rag_adapter.py:20)

**Problem**: The Civil Law RAG pipeline has modules with the same names as the main project (`config`, `graph`, `nodes`, etc.). The adapter must manipulate `sys.path` and `sys.modules` to force Python to import from the RAG directory instead of the cached project modules.

**Symptoms**:
- `AttributeError` on imported modules (wrong module loaded)
- Stale state if modules are imported elsewhere before the adapter runs
- Intermittent failures in multi-threaded scenarios

**Current mitigation**: The adapter evicts conflicting modules from `sys.modules` before each import:
```python
_RAG_MODULES = ("config", "graph", "nodes", "state", "edges", "utils")
for mod in list(sys.modules.keys()):
    if mod in _RAG_MODULES or any(mod.startswith(f"{m}.") for m in _RAG_MODULES):
        del sys.modules[mod]
```

**Future fix**: Restructure RAG pipeline modules to use unique package names (e.g., `rag_civil_law.graph` instead of just `graph`).

---

### 2. Summarizer MongoDB Fallback Field Names

**Location**: [`Supervisor/agents/summarize_adapter.py`](../Supervisor/agents/summarize_adapter.py:36)

**Problem**: The summarize adapter falls back to MongoDB when no documents are provided in the context. It reads documents using hardcoded field names:

```python
raw_text = doc.get("text", "")
doc_id = str(doc.get("title") or doc.get("source_file") or doc.get("_id", "unknown"))
```

If the MongoDB document schema changes (e.g., `text` renamed to `content`), the summarizer silently returns empty documents without raising an error.

**Impact**: Summarization produces empty or nonsensical output for pre-ingested documents.

**Fix**: Add a validation step that warns when fetched documents have empty `raw_text`.

---

### 3. Synchronous Graph in Async API (BUG-7 related)

**Location**: [`api/services/query_service.py`](../api/services/query_service.py:133)

**Problem**: The LangGraph supervisor uses synchronous LLM calls and synchronous pymongo under the hood. The API wraps this in `asyncio.to_thread()`:

```python
events = await asyncio.to_thread(_stream_graph_sync, state)
```

This means each query blocks an OS thread. Under high concurrency, the thread pool can be exhausted.

**Symptoms**:
- Requests hang under load
- `asyncio.to_thread` timeout errors

**Mitigation**: Keep the thread pool size reasonable and consider rate limiting to prevent overload. The default `asyncio` thread pool is sufficient for moderate load.

---

### 4. Error Message Sanitization (BUG-7)

**Location**: [`api/services/query_service.py`](../api/services/query_service.py:135)

**Problem**: Raw exception messages from the supervisor graph could leak internal details (file paths, stack traces, database URIs) to the client.

**Fix applied**: The query service now logs the full exception server-side but sends only a generic error message to the client:

```python
logger.exception("Supervisor graph failed: %s", exc)
yield _format_sse("error", {
    "detail": "An internal error occurred while processing the query",
    "code": "INTERNAL_ERROR",
})
```

---

### 5. Conversation Not Found on Query (BUG-11)

**Location**: [`api/services/query_service.py`](../api/services/query_service.py:97)

**Problem**: If a `conversation_id` is explicitly provided in a query request but doesn't exist in MongoDB, the system previously created a new conversation silently, losing the intended context.

**Fix applied**: The query service now checks if the conversation exists and emits an error event if not found:

```python
if conversation_id:
    conv = await get_conversation(db, conversation_id, user_id)
    if conv is None:
        yield _format_sse("error", {
            "detail": "Conversation not found",
            "code": "CONVERSATION_NOT_FOUND",
        })
        yield _format_sse("done", {})
        return
```

---

### 6. ChromaDB Remnants in requirements.txt

**Problem**: `requirements.txt` still lists `chromadb` as a dependency despite the migration to Qdrant. This adds unnecessary installation time and potential dependency conflicts.

**Impact**: Larger Docker images, slower builds, confusing dependency tree.

**Fix**: Remove `chromadb` from `requirements.txt` once all ChromaDB references are confirmed removed from the codebase.

---

### 7. Incomplete docker-compose.yml

**Problem**: The `docker-compose.yml` only defines `mongo`, `api`, and `streamlit` services. Qdrant, Redis, MinIO, and PostgreSQL are not included, meaning they must be started separately for a complete deployment.

**Impact**: New developers may not realize these services are needed, leading to confusing startup errors.

**Fix**: Add all required services to `docker-compose.yml` or provide a `docker-compose.full.yml` overlay. See [DEPLOYMENT.md](DEPLOYMENT.md) for manual service startup commands.

---

### 8. Default JWT Secret

**Problem**: The default JWT secret in `config/settings.yaml` is `123456`. This is insecure and must be changed in production.

**Impact**: Anyone who knows the default secret can forge valid JWT tokens.

**Fix**: Always set `JA_API_JWT_SECRET` in production environments. The system should log a warning on startup if the default secret is detected.

---

### 9. Embedding Model Download Timeout

**Problem**: On first startup, the `BAAI/bge-m3` embedding model (approximately 2 GB) is downloaded from HuggingFace. This can exceed Docker health check timeouts, causing the container to be marked unhealthy and restarted in a loop.

**Symptoms**:
- Container repeatedly restarts during first deployment
- Health check fails with timeout
- Logs show: `Downloading (BAAI/bge-m3)...`

**Mitigation**:
- The Docker health check `start_period` is set to 120s (increased from default)
- Use the `huggingface_cache` Docker volume to persist the model between container restarts
- Pre-download the model in the Docker image build step (add to Dockerfile)

---

### 10. Rate Limiting Without Redis

**Problem**: If Redis is unavailable, the rate limiting system fails open (allows all requests). This means there is no protection against abuse without Redis.

**Mitigation**: Redis availability is logged at startup. Monitor the warning message:
```
WARNING:api.app:Redis connection failed (non-fatal, caching disabled): ...
```

---

## Debugging Guide

### Inspecting Supervisor State

To debug what the supervisor graph is doing, enable DEBUG logging:

```bash
export JA_API_DEBUG=true
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Each node logs its input/output state. Look for these log messages:

```
INFO:Supervisor.nodes.classify_intent:Intent classified: civil_law_rag
INFO:Supervisor.nodes.dispatch_agents:Dispatching to agent: civil_law_rag
INFO:Supervisor.nodes.dispatch_agents:Agent civil_law_rag completed successfully
INFO:Supervisor.nodes.validate_output:Validation result: pass
```

### Inspecting SSE Events

Use curl with `-N` (no buffering) to see raw SSE events:

```bash
curl -N -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "case_id": ""}'
```

### Checking MongoDB State

```bash
# List collections
mongosh --eval "db.getCollectionNames()"

# Check cases
mongosh --eval "db.cases.find().limit(5).pretty()"

# Check conversations
mongosh --eval "db.conversations.find().limit(5).pretty()"

# Check indexes
mongosh --eval "db.cases.getIndexes()"
```

### Checking Qdrant State

```bash
# List collections
curl http://localhost:6333/collections

# Check collection info
curl http://localhost:6333/collections/judicial_docs

# Count vectors
curl http://localhost:6333/collections/judicial_docs/points/count
```

### Checking Redis State

```bash
# Connect
redis-cli

# List rate limit keys
KEYS rate_limit:*

# Check a specific rate limit
GET rate_limit:user-123

# List cache keys
KEYS cache:*
```
