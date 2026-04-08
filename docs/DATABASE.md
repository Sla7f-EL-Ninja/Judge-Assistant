# Database Reference

Judge Assistant uses five database systems. All connections are managed via the FastAPI lifespan handler in [`api/app.py`](../api/app.py:42). MongoDB and Qdrant are required; Redis, MinIO, and PostgreSQL degrade gracefully if unavailable.

## Connection Lifecycle

```python
# Startup (in order)
await connect_mongo(settings)      # Required
connect_qdrant(settings)           # Required (non-fatal on failure)
await connect_redis(settings)      # Optional -- caching disabled if down
connect_minio(settings)            # Optional -- falls back to local disk
await connect_postgres(settings)   # Optional -- user management disabled
```

---

## MongoDB

**Client**: Motor (async) via [`api/db/mongodb.py`](../api/db/mongodb.py)
**Default URI**: `mongodb://localhost:27017/`
**Default Database**: `Rag`

### Connection Pool Settings

| Setting | YAML Key | Default |
|---|---|---|
| Min pool size | `mongodb.min_pool_size` | 5 |
| Max pool size | `mongodb.max_pool_size` | 50 |
| Server selection timeout | `mongodb.server_selection_timeout_ms` | 5000 ms |

### Collections

Collection name constants are defined in [`api/db/collections.py`](../api/db/collections.py):

#### `cases`

Stores case metadata.

| Field | Type | Description |
|---|---|---|
| `_id` | `str` | Case identifier |
| `user_id` | `str` | Owner user ID |
| `title` | `str` | Case title |
| `description` | `str` | Case description |
| `status` | `str` | Case status |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last update timestamp |

**Indexes**:
- `(user_id, status)` -- compound, for filtered list queries
- `created_at` -- for sorting

#### `conversations`

Stores conversation sessions tied to cases.

| Field | Type | Description |
|---|---|---|
| `_id` | `str` | Conversation identifier |
| `case_id` | `str` | Parent case |
| `user_id` | `str` | Owner user ID |
| `turns` | `List[dict]` | Ordered list of `{query, response, ...}` |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last update timestamp |

**Indexes**:
- `(case_id, user_id)` -- compound, for filtered list queries
- `created_at` -- for sorting

#### `summaries`

Stores auto-generated case summaries.

| Field | Type | Description |
|---|---|---|
| `_id` | `str` | Summary identifier |
| `case_id` | `str` | Parent case (unique) |
| `content` | `str/dict` | Summary content |
| `created_at` | `datetime` | Creation timestamp |

**Indexes**:
- `case_id` -- unique index (one summary per case)

#### `files`

Stores uploaded file metadata.

| Field | Type | Description |
|---|---|---|
| `_id` | `str` | File identifier |
| `user_id` | `str` | Uploader user ID |
| `filename` | `str` | Original filename |
| `content_type` | `str` | MIME type |
| `size` | `int` | File size in bytes |
| `storage_path` | `str` | Path in MinIO or local disk |
| `created_at` | `datetime` | Upload timestamp |

**Indexes**:
- `user_id` -- for ownership queries

#### `documents`

Stores processed document metadata (post-OCR, post-classification).

| Field | Type | Description |
|---|---|---|
| `_id` | `str` | Document identifier |
| `case_id` | `str` | Parent case |
| `doc_type` | `str` | Document type classification |
| `text` | `str` | Extracted/processed text content |
| `title` | `str` | Document title |
| `source_file` | `str` | Reference to originating file |
| `created_at` | `datetime` | Processing timestamp |

**Indexes**:
- `case_id` -- for filtered retrieval
- `(case_id, doc_type)` -- compound, for type-filtered retrieval

> **Critical**: The `text`, `title`, and `source_file` field names are used by the summarize adapter's MongoDB fallback (see [`Supervisor/agents/summarize_adapter.py`](../Supervisor/agents/summarize_adapter.py:36)). Changing these field names will silently break summarization for pre-ingested documents.

#### Legacy Collection: `Document Storage`

The default MongoDB collection name in `config/settings.yaml` is `Document Storage`. This is the collection used by the Supervisor's synchronous pymongo access for document retrieval by case_id.

---

## Qdrant (Vector Store)

**Client**: `qdrant-client` via [`api/db/qdrant.py`](../api/db/qdrant.py)
**Default Host**: `localhost:6333` (HTTP), `localhost:6334` (gRPC)
**Preferred Protocol**: gRPC

### Collections

#### `judicial_docs`

Civil law article embeddings for the Civil Law RAG pipeline.

| Property | Value |
|---|---|
| YAML key | `qdrant.collection` |
| Vector size | 1024 |
| Distance metric | COSINE |
| Embedding model | `BAAI/bge-m3` |
| Payload indexes | `case_id` (keyword), `doc_type` (keyword) |

#### `case_docs`

Case-specific document embeddings for the Case Document RAG pipeline.

| Property | Value |
|---|---|
| YAML key | `qdrant.case_collection` |
| Vector size | 1024 |
| Distance metric | COSINE |
| Embedding model | `BAAI/bge-m3` |
| Payload indexes | `case_id` (keyword), `doc_type` (keyword) |

### Collection Auto-Creation

On startup, [`_ensure_collection()`](../api/db/qdrant.py:52) checks if the collection exists and creates it if missing, including payload indexes on `case_id` and `doc_type`.

---

## Redis

**Client**: `redis.asyncio` via [`api/db/redis.py`](../api/db/redis.py)
**Default URL**: `redis://localhost:6379/0`
**Max Connections**: 20

### Key Patterns

#### Cache Keys

General LLM response caching to avoid redundant API calls.

| Pattern | TTL | Description |
|---|---|---|
| `cache:{hash}` | 3600s (1 hour, configurable via `redis.cache_ttl_seconds`) | Cached LLM responses |

**Cache helpers**: [`cache_get(key)`](../api/db/redis.py:63) and [`cache_set(key, value, ttl_seconds)`](../api/db/redis.py:73)

#### Rate Limiting Keys

Sliding window counter pattern for per-user rate limiting.

| Pattern | TTL | Description |
|---|---|---|
| `rate_limit:{user_id}` | 60s (configurable via `redis.rate_limit_window_seconds`) | Request counter per user |

**Rate limit**: 100 requests per 60-second window (configurable via `redis.rate_limit_requests`).

**Behavior**: [`check_rate_limit()`](../api/db/redis.py:87) returns `True` if allowed, `False` if rate-limited. Fails open (allows requests) if Redis is down.

### Graceful Degradation

All Redis operations are wrapped in try/except blocks. If Redis is unavailable:
- `cache_get()` returns `None` (cache miss)
- `cache_set()` silently fails
- `check_rate_limit()` returns `True` (allows all requests)

---

## MinIO (S3-Compatible Object Storage)

**Client**: `minio` Python SDK via [`api/db/minio_client.py`](../api/db/minio_client.py)
**Default Endpoint**: `localhost:9000`
**Default Credentials**: `minioadmin` / `minioadmin`

### Bucket

| Property | Value |
|---|---|
| Bucket name | `judge-assistant-files` |
| YAML key | `minio.bucket` |
| Auto-created | Yes, on startup |

### Operations

| Function | Description |
|---|---|
| [`upload_file(object_name, data, content_type)`](../api/db/minio_client.py:75) | Upload bytes, returns object name |
| [`download_file(object_name)`](../api/db/minio_client.py:109) | Download file contents as bytes |
| [`get_presigned_url(object_name, expires_seconds)`](../api/db/minio_client.py:122) | Generate temporary download URL (default 1 hour) |
| [`delete_file(object_name)`](../api/db/minio_client.py:147) | Remove a file from the bucket |

### Graceful Degradation

If MinIO is unavailable at startup, the system falls back to local disk storage using the `upload_dir` setting (default `./uploads`).

---

## PostgreSQL

**Client**: SQLAlchemy async via [`api/db/postgres.py`](../api/db/postgres.py)
**Default URL**: `postgresql+asyncpg://postgres:postgres@localhost:5432/judge_assistant`

### Purpose

User management, role-based access control (RBAC), and audit logging for legal compliance.

### Tables

#### `users`

| Column | Type | Constraints |
|---|---|---|
| `id` | `String(64)` | PK, auto-generated `user_{uuid}` |
| `email` | `String(255)` | Unique, not null, indexed |
| `hashed_password` | `String(255)` | Not null |
| `full_name` | `String(255)` | Not null |
| `role` | `Enum(judge, clerk, admin, viewer)` | Not null, default `viewer` |
| `is_active` | `Boolean` | Default `True` |
| `created_at` | `DateTime(tz)` | Server default `now()` |
| `updated_at` | `DateTime(tz)` | Server default `now()`, auto-update |

#### `user_permissions`

Fine-grained permissions beyond role defaults.

| Column | Type | Constraints |
|---|---|---|
| `id` | `Integer` | PK, auto-increment |
| `user_id` | `String(64)` | FK -> `users.id`, cascade delete, indexed |
| `permission` | `Enum(...)` | Not null |
| `granted_at` | `DateTime(tz)` | Server default `now()` |

**Permission types**: `cases:read`, `cases:write`, `cases:delete`, `files:read`, `files:write`, `files:delete`, `queries:execute`, `conversations:read`, `conversations:delete`, `summaries:read`, `users:manage`, `audit:read`

#### `audit_logs`

Audit trail for legal compliance (relationship defined on `User` model).

### Connection Pool

| Setting | YAML Key | Default |
|---|---|---|
| Pool size | `postgresql.pool_size` | 10 |
| Max overflow | `postgresql.max_overflow` | 20 |

### Graceful Degradation

If PostgreSQL is unavailable at startup, user management features are disabled but the rest of the system continues to operate.
