# Deployment Guide

## Docker Compose (Current)

The current [`docker-compose.yml`](../docker-compose.yml) defines three services:

### Services

#### `mongo`
- **Image**: `mongo:7`
- **Port**: `27017`
- **Volume**: `mongo_data:/data/db`
- **Health check**: `mongosh --eval "db.adminCommand('ping')"` every 10s

#### `api`
- **Build**: From `Dockerfile` in project root
- **Port**: `8000`
- **Depends on**: `mongo` (healthy)
- **Environment**: `JA_MONGODB_URI=mongodb://mongo:27017/`
- **Volumes**:
  - `./uploads:/app/uploads` -- File upload directory
  - `./chroma_data:/app/chroma_data` -- Legacy ChromaDB data
  - `huggingface_cache:/root/.cache/huggingface` -- Model cache
- **Health check**: `curl -f http://localhost:8000/api/v1/health` every 30s

#### `streamlit`
- **Build**: From `streamlit_app/Dockerfile`
- **Port**: `8501`
- **Depends on**: `api` (healthy)
- **Profile**: `testing` (only starts with `--profile testing`)
- **Environment**: `API_BASE_URL=http://api:8000`

### Missing Services

The following services are used by the application but are **not yet included** in `docker-compose.yml`:

| Service | Default Config | Required |
|---|---|---|
| **Qdrant** | `localhost:6333` | Yes (vector search) |
| **Redis** | `localhost:6379` | No (caching/rate limiting) |
| **MinIO** | `localhost:9000` | No (file storage) |
| **PostgreSQL** | `localhost:5432` | No (user management) |

To run these services locally alongside Docker Compose, start them separately:

```bash
# Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Redis
docker run -d --name redis -p 6379:6379 redis:7

# MinIO
docker run -d --name minio -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# PostgreSQL
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=judge_assistant \
  postgres:16
```

---

## Dockerfile

The API image is built from [`Dockerfile`](../Dockerfile):

```dockerfile
FROM python:3.11-slim

# System deps for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .
RUN mkdir -p /app/uploads /app/chroma_data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

**Key details**:
- Uses CPU-only PyTorch (`--extra-index-url` for CPU wheels)
- 60-second start period for health checks (embedding model may download on first run)
- Creates upload and legacy ChromaDB directories

---

## Health Checks

### API Health Endpoint

```bash
curl http://localhost:8000/api/v1/health
```

Returns connectivity status for all database backends:

```json
{
  "status": "healthy",
  "mongodb": "connected",
  "qdrant": "connected",
  "redis": "connected",
  "minio": "connected",
  "postgresql": "connected"
}
```

### Docker Health Checks

| Service | Command | Interval | Timeout | Start Period |
|---|---|---|---|---|
| `mongo` | `mongosh --eval "db.adminCommand('ping')"` | 10s | 5s | 10s |
| `api` | `curl -f http://localhost:8000/api/v1/health` | 30s | 10s | 120s |

---

## Scaling Considerations

### Horizontal Scaling

- The API is stateless (all state is in databases), so multiple API instances can run behind a load balancer
- MongoDB sessions are not pinned; any API instance can handle any request
- Redis-based rate limiting works across instances (shared counter)

### Performance Bottlenecks

1. **Embedding model loading** -- The `BAAI/bge-m3` model is loaded into memory on first use. Cache the HuggingFace directory.
2. **LLM API calls** -- Each query involves 3+ LLM calls (classify, merge/pass-through, validate). Caching identical queries in Redis helps.
3. **Synchronous graph execution** -- The LangGraph supervisor uses synchronous LLM calls, wrapped in `asyncio.to_thread()`. Each query blocks a thread.

### Resource Requirements

| Component | Min Memory | Recommended |
|---|---|---|
| API (with embedding model) | 2 GB | 4 GB |
| MongoDB | 512 MB | 2 GB |
| Qdrant | 512 MB | 2 GB+ (depends on collection size) |
| Redis | 128 MB | 512 MB |
| MinIO | 256 MB | 1 GB |

---

## Monitoring and Logging

### Logging

The API uses Python's built-in `logging` module. Log levels are controlled by the `debug` setting:
- `debug: true` -- DEBUG level, verbose output
- `debug: false` -- INFO level (default)

Key loggers:
- `api.app` -- Startup/shutdown, connection status
- `api.services.query_service` -- Query execution, SSE events
- `Supervisor.nodes.*` -- Graph node execution
- `Supervisor.agents.*` -- Agent adapter invocations

### Structured Logs

Each log message includes the module name and log level. Example:

```
INFO:api.app:Connecting to MongoDB at mongodb://localhost:27017/ ...
INFO:api.app:MongoDB connected (db=Rag)
INFO:api.app:Qdrant connected at localhost:6333 (collection=judicial_docs)
WARNING:api.app:Redis connection failed (non-fatal, caching disabled): ...
```

---

## Backup Strategy

### MongoDB

```bash
# Backup
mongodump --uri="mongodb://localhost:27017/Rag" --out=/backup/$(date +%Y%m%d)

# Restore
mongorestore --uri="mongodb://localhost:27017/Rag" /backup/20250115/
```

### Qdrant

Qdrant supports snapshots:
```bash
# Create snapshot
curl -X POST http://localhost:6333/collections/judicial_docs/snapshots

# List snapshots
curl http://localhost:6333/collections/judicial_docs/snapshots
```

### MinIO

MinIO supports `mc mirror` for backup:
```bash
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mirror local/judge-assistant-files /backup/minio/
```

### PostgreSQL

```bash
pg_dump -h localhost -U postgres judge_assistant > backup.sql
```
