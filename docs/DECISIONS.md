# Architecture Decision Records

## ADR-001: LangGraph for Multi-Agent Orchestration

**Status**: Accepted
**Date**: Project inception

### Context

The system needs to orchestrate multiple AI agents (OCR, RAG, summarization, reasoning) with conditional routing, retry logic, and state management. Options considered:
- LangChain Agents (ReAct pattern)
- Custom Python orchestration
- LangGraph state machines

### Decision

Use **LangGraph** `StateGraph` for the supervisor workflow.

### Rationale

- **Explicit state**: The `TypedDict` state makes data flow visible and debuggable
- **Conditional routing**: Built-in conditional edges with router functions
- **Retry loops**: Natural representation as graph cycles (validate -> dispatch)
- **Streaming**: Built-in `graph.stream()` for real-time progress events
- **Composability**: The Civil Law RAG pipeline is also a LangGraph workflow, demonstrating nesting

### Consequences

- Requires understanding LangGraph concepts (State, Node, Edge)
- Synchronous execution model requires `asyncio.to_thread()` wrapper for the async API
- State must be serializable (TypedDict, no complex objects)

---

## ADR-002: Qdrant Replaces ChromaDB for Vector Storage

**Status**: Accepted
**Date**: Migration completed

### Context

The system originally used ChromaDB for vector storage. As the dataset grew and production requirements emerged, limitations were encountered:
- ChromaDB's in-process mode is not suitable for multi-container deployment
- Limited filtering capabilities for case-scoped document retrieval
- No built-in payload indexing

### Decision

Migrate to **Qdrant** as the vector store.

### Rationale

- **Client-server architecture**: Qdrant runs as a separate service, suitable for production
- **Payload indexes**: Native support for filtering by `case_id` and `doc_type`
- **gRPC support**: Higher throughput for vector operations
- **COSINE distance**: Well-suited for embedding similarity search
- **Horizontal scaling**: Qdrant supports sharding and replication

### Consequences

- `requirements.txt` still includes `chromadb` (needs cleanup)
- `docker-compose.yml` references `chroma_data` volume (legacy)
- The README still references ChromaDB in some places

### Migration Notes

- Collections: `judicial_docs` (civil law), `case_docs` (case documents)
- Vector size: 1024 (BAAI/bge-m3)
- Distance: COSINE
- Payload indexes on `case_id` and `doc_type` are auto-created on startup

---

## ADR-003: Five-Database Architecture

**Status**: Accepted

### Context

The system has diverse storage needs: document metadata, vector search, caching, file storage, and user management.

### Decision

Use five purpose-built databases:

| Database | Purpose |
|---|---|
| MongoDB | Document metadata, cases, conversations, summaries |
| Qdrant | Vector embeddings for similarity search |
| Redis | Caching, rate limiting, session context |
| MinIO | S3-compatible file/object storage |
| PostgreSQL | User management, RBAC, audit logging |

### Rationale

- **MongoDB**: Flexible schema for evolving document structures; Motor async driver for FastAPI
- **Qdrant**: Purpose-built for vector search with filtering
- **Redis**: Sub-millisecond caching for repeated queries; sliding window rate limiting
- **MinIO**: S3-compatible API for durable file storage; presigned URLs for secure access
- **PostgreSQL**: ACID compliance for user data; SQLAlchemy ORM for structured RBAC

### Consequences

- Higher operational complexity (5 services to manage)
- Graceful degradation needed (Redis, MinIO, PostgreSQL are optional)
- Connection lifecycle management in FastAPI lifespan

---

## ADR-004: SSE Streaming for Query Responses

**Status**: Accepted

### Context

The supervisor graph runs multiple LLM calls sequentially (classify, dispatch, merge, validate). Total execution time can be 10-30 seconds. The frontend needs to show progress.

### Decision

Use **Server-Sent Events (SSE)** for the query endpoint.

### Rationale

- **Incremental updates**: Each graph node completion is streamed as a `progress` event
- **Simple protocol**: SSE is simpler than WebSockets for unidirectional server-to-client streaming
- **HTTP/1.1 compatible**: Works with standard reverse proxies
- **Browser native**: `EventSource` API available in all modern browsers

### Alternatives Considered

- **WebSockets**: Bidirectional (unnecessary), more complex connection management
- **Long polling**: Higher latency, more server resources
- **HTTP chunked transfer**: Less structured than SSE

### Consequences

- Requires `X-Accel-Buffering: no` header for nginx/reverse proxy compatibility
- Client must handle reconnection logic (SSE auto-reconnects by default)
- Event format: `event: {type}\ndata: {json}\n\n`

---

## ADR-005: Centralized YAML Configuration with Environment Overrides

**Status**: Accepted

### Context

Configuration was scattered across multiple files and hardcoded values. API keys, database URIs, and model names needed consistent management.

### Decision

Adopt a **centralized YAML configuration** system:
- `config/settings.yaml` as the single source of truth
- `config/settings.local.yaml` for local overrides (gitignored)
- `JA_` prefixed environment variables as the highest-priority override

### Rationale

- **Single source of truth**: All defaults in one readable YAML file
- **Override hierarchy**: YAML defaults < local YAML < environment variables
- **API key separation**: Sensitive keys (`GROQ_API_KEY`, `GOOGLE_API_KEY`) stay in `.env`, not in YAML
- **Type safety**: Environment variable values are auto-cast to match YAML types
- **Discoverability**: `JA_` prefix makes it clear which env vars belong to this project

### Consequences

- Two config loading paths: `config/__init__.py` (global) and `config/api.py` (FastAPI Settings)
- The `JA_` prefix convention must be documented clearly
- Nested YAML keys become long env var names (e.g., `JA_OCR_PREPROCESSING_ENABLE_DESKEW`)

---

## ADR-006: LLM Tier System (High/Medium/Low)

**Status**: Accepted

### Context

Different tasks require different LLM capabilities. Response generation needs strong reasoning; intent classification needs structured output; validation needs basic judgment.

### Decision

Implement a **three-tier LLM system**:
- **High**: Complex reasoning, synthesis, legal analysis (merge_responses, generation)
- **Medium**: Classification, grading, structured extraction (classify_intent)
- **Low**: Simple routing, validation, fallback (validate_output)

### Rationale

- **Cost optimization**: Less capable (cheaper) models can handle simpler tasks
- **Latency optimization**: Simpler tasks can use faster models
- **Flexibility**: Each tier can be independently configured (provider, model, temperature)
- **Future-proofing**: When faster/cheaper models become available, they can be slotted into lower tiers

### Current Configuration

All three tiers currently use the same model (`Groq llama-3.3-70b-versatile`), but the infrastructure supports different models per tier.

### Consequences

- The `get_llm(tier)` factory must be used consistently across all nodes
- Configuration has three sections under `llm:` in YAML
- Monitoring should track per-tier usage and latency

---

## ADR-007: JWT Authentication with Shared Secret

**Status**: Accepted

### Context

The Judge Assistant API is consumed by a separate Express.js backend that handles user registration and login. The API needs to validate user identity without managing its own user sessions.

### Decision

Use **JWT Bearer tokens** with a shared secret between the Express backend and the Python API.

### Rationale

- **Stateless**: No session store needed on the API side
- **Simple**: HS256 with a shared secret is straightforward to implement
- **Standard**: JWT is widely supported across languages and frameworks
- **Claim-based**: The `user_id` claim provides identity without database lookup

### Alternatives Considered

- **OAuth2/OIDC**: Too complex for internal service-to-service auth
- **API keys**: No user identity; harder to revoke per-user
- **mTLS**: Infrastructure-heavy for this use case

### Consequences

- The shared secret must be kept secure and rotated together
- Token expiration must be handled by the Express backend
- No token refresh mechanism on the API side
- Default secret (`123456`) in YAML is a security risk -- must be overridden in production
