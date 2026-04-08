# Setup Guide

## Prerequisites

- **Python 3.11+**
- **Docker and Docker Compose** (recommended for infrastructure services)
- **API keys** for at least one LLM provider:
  - [Groq API key](https://console.groq.com/) (default for all tiers)
  - [Google AI API key](https://makersuite.google.com/app/apikey) (optional)

## Infrastructure Services

The system requires (or optionally uses) these external services:

| Service | Required | Default Address | Purpose |
|---|---|---|---|
| MongoDB | Yes | `localhost:27017` | Case data, conversations, documents |
| Qdrant | Yes | `localhost:6333` | Vector embeddings |
| Redis | No | `localhost:6379` | Caching, rate limiting |
| MinIO | No | `localhost:9000` | File storage (falls back to local disk) |
| PostgreSQL | No | `localhost:5432` | User management, RBAC |

---

## Quick Start with Docker Compose

The fastest way to get the API running with MongoDB:

```bash
# 1. Clone the repository
git clone https://github.com/hassann16541-create/Judge-Assistant.git
cd Judge-Assistant

# 2. Create environment file with your API keys
cat > .env << 'EOF'
GROQ_API_KEY=gsk_your_groq_key_here
# GOOGLE_API_KEY=your_google_key_here
EOF

# 3. Start MongoDB + API
make up
# Or: docker compose up -d

# 4. Check health
curl http://localhost:8000/api/v1/health

# 5. Start everything including Streamlit UI
make up-all
# Or: docker compose --profile testing up -d
```

### Makefile Targets

| Target | Command | Description |
|---|---|---|
| `make up` | `docker compose up -d` | Start MongoDB + API |
| `make up-all` | `docker compose --profile testing up -d` | Start everything including Streamlit |
| `make down` | `docker compose down` | Stop all services |
| `make down-clean` | `docker compose down -v` | Stop and remove all data volumes |
| `make logs` | `docker compose logs -f api` | Watch API logs |
| `make build` | `docker compose build --no-cache` | Rebuild images |
| `make shell` | `docker compose exec api bash` | Open bash in API container |
| `make mongo-shell` | `docker compose exec mongo mongosh` | Open MongoDB shell |

---

## Local Development Setup

### 1. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Start infrastructure services

You need at least MongoDB and Qdrant running locally:

```bash
# MongoDB
docker run -d --name mongo -p 27017:27017 mongo:7

# Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Redis (optional)
docker run -d --name redis -p 6379:6379 redis:7

# MinIO (optional)
docker run -d --name minio -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# PostgreSQL (optional)
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=judge_assistant \
  postgres:16
```

### 3. Configure environment

Create a `.env` file in the project root:

```bash
# Required: LLM API keys (not stored in YAML)
GROQ_API_KEY=gsk_your_groq_key_here
GOOGLE_API_KEY=your_google_key_here  # Optional

# Optional: Override any YAML setting using JA_ prefix
# JA_MONGODB_URI=mongodb://localhost:27017/
# JA_API_JWT_SECRET=your-production-secret
```

Or create `config/settings.local.yaml` for local overrides (gitignored):

```yaml
api:
  jwt_secret: my-local-secret
  debug: true

mongodb:
  uri: mongodb://localhost:27017/

redis:
  url: redis://localhost:6379/0
```

### 4. Run the API

```bash
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. OpenAPI docs at `http://localhost:8000/docs`.

---

## Environment Variables

### API Keys (Required)

These are NOT stored in YAML. They are read directly by LangChain provider classes.

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for LLM calls (default provider for all tiers) |
| `GOOGLE_API_KEY` | Google AI API key (optional, used if provider set to `google`) |

### JA_ Prefix Override Convention

Any YAML setting can be overridden via environment variable using the `JA_` prefix convention defined in [`config/__init__.py`](../config/__init__.py):

```
JA_{SECTION}_{KEY}
```

Nested keys are joined with `_` and uppercased. Values are auto-cast to match the YAML type (bool, int, float, list, str).

**Examples**:

| Environment Variable | YAML Path | Type |
|---|---|---|
| `JA_LLM_HIGH_MODEL` | `llm.high.model` | `str` |
| `JA_MONGODB_URI` | `mongodb.uri` | `str` |
| `JA_API_JWT_SECRET` | `api.jwt_secret` | `str` |
| `JA_API_DEBUG` | `api.debug` | `bool` |
| `JA_REDIS_URL` | `redis.url` | `str` |
| `JA_QDRANT_HOST` | `qdrant.host` | `str` |
| `JA_QDRANT_PORT` | `qdrant.port` | `int` |
| `JA_MINIO_ENDPOINT` | `minio.endpoint` | `str` |
| `JA_MINIO_SECURE` | `minio.secure` | `bool` |
| `JA_OCR_LANGUAGE` | `ocr.language` | `str` |
| `JA_SUPERVISOR_MAX_RETRIES` | `supervisor.max_retries` | `int` |

### All Settings (from `config/api.py`)

| Setting | YAML Path | Default | Description |
|---|---|---|---|
| `app_name` | `api.app_name` | `Judge Assistant API` | Application name |
| `app_version` | `api.app_version` | `0.1.0` | API version |
| `debug` | `api.debug` | `false` | Debug mode |
| `cors_origins` | `api.cors_origins` | `*` | Comma-separated CORS origins |
| `jwt_secret` | `api.jwt_secret` | `123456` | JWT signing secret |
| `jwt_algorithm` | `api.jwt_algorithm` | `HS256` | JWT algorithm |
| `max_upload_bytes` | `api.max_upload_bytes` | `20971520` (20 MB) | Max file upload size |
| `allowed_mime_types` | `api.allowed_mime_types` | PDF, PNG, JPEG, TIFF, BMP, WebP | Allowed upload types |
| `upload_dir` | `api.upload_dir` | `./uploads` | Local upload fallback directory |
| `langgraph_module` | `api.langgraph_module` | `Supervisor.graph` | LangGraph module path |
| `mongo_uri` | `mongodb.uri` | `mongodb://localhost:27017/` | MongoDB connection URI |
| `mongo_db` | `mongodb.database` | `Rag` | MongoDB database name |
| `mongo_min_pool_size` | `mongodb.min_pool_size` | `5` | Min connection pool size |
| `mongo_max_pool_size` | `mongodb.max_pool_size` | `50` | Max connection pool size |
| `qdrant_host` | `qdrant.host` | `localhost` | Qdrant host |
| `qdrant_port` | `qdrant.port` | `6333` | Qdrant HTTP port |
| `qdrant_grpc_port` | `qdrant.grpc_port` | `6334` | Qdrant gRPC port |
| `qdrant_collection` | `qdrant.collection` | `judicial_docs` | Main Qdrant collection |
| `qdrant_vector_size` | `qdrant.vector_size` | `1024` | Vector dimension |
| `qdrant_prefer_grpc` | `qdrant.prefer_grpc` | `true` | Use gRPC protocol |
| `redis_url` | `redis.url` | `redis://localhost:6379/0` | Redis connection URL |
| `redis_max_connections` | `redis.max_connections` | `20` | Max Redis connections |
| `redis_cache_ttl_seconds` | `redis.cache_ttl_seconds` | `3600` | Cache TTL (1 hour) |
| `redis_rate_limit_requests` | `redis.rate_limit_requests` | `100` | Rate limit per window |
| `redis_rate_limit_window_seconds` | `redis.rate_limit_window_seconds` | `60` | Rate limit window |
| `minio_endpoint` | `minio.endpoint` | `localhost:9000` | MinIO endpoint |
| `minio_access_key` | `minio.access_key` | `minioadmin` | MinIO access key |
| `minio_secret_key` | `minio.secret_key` | `minioadmin` | MinIO secret key |
| `minio_bucket` | `minio.bucket` | `judge-assistant-files` | MinIO bucket name |
| `minio_secure` | `minio.secure` | `false` | Use HTTPS for MinIO |
| `postgresql_url` | `postgresql.url` | `postgresql+asyncpg://...` | PostgreSQL async URL |
| `postgresql_pool_size` | `postgresql.pool_size` | `10` | Connection pool size |
| `postgresql_max_overflow` | `postgresql.max_overflow` | `20` | Max overflow connections |
| `embedding_model` | `embedding.model` | `BAAI/bge-m3` | Embedding model name |

---

## Common Setup Errors

### `ModuleNotFoundError: No module named 'config'`

The project root must be on `sys.path`. When running from the repo root, Python should find the `config/` package. If running tests, the `conftest.py` adds the project root to `sys.path`.

### `RuntimeError: MongoDB is not connected`

MongoDB must be running before the API starts. Check with:
```bash
docker ps | grep mongo
mongosh --eval "db.adminCommand('ping')"
```

### `Qdrant connection failed`

Qdrant must be running at the configured host/port. By default this is non-fatal; the API starts but vector search will fail.

### `GROQ_API_KEY not set`

LLM API keys are not in YAML. Set them in `.env` or as environment variables:
```bash
export GROQ_API_KEY=gsk_your_key_here
```

### HuggingFace model download on first run

The embedding model (`BAAI/bge-m3`) is downloaded on first use. This can take several minutes. In Docker, the download is cached in the `huggingface_cache` volume.

### `jwt_secret` warning

The default JWT secret in `config/settings.yaml` is `123456`. Always override this in production via `JA_API_JWT_SECRET` or `.env`.
