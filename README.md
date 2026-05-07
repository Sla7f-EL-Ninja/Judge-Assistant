# Judge Assistant (حكيم)

An AI-powered legal assistant built for Egyptian judicial workflows. It combines Arabic OCR, multi-corpus civil law retrieval, case document reasoning, multi-document summarization, and an adaptive chat reasoner into a unified REST API — orchestrated by a 15-node LangGraph multi-agent supervisor with persistent long-term memory.

> **Language:** All agent outputs, prompts, and responses are in Arabic. The system is purpose-built for Egyptian civil law (القانون المدني المصري).

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Agent Roster](#agent-roster)
- [Infrastructure Stack](#infrastructure-stack)
- [Prerequisites](#prerequisites)
- [Quick Start (Local)](#quick-start-local)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Authentication](#authentication)
- [Streaming Queries (SSE)](#streaming-queries-sse)
- [Report Generation](#report-generation)
- [MCP Transport Layer](#mcp-transport-layer)
- [LLM Tier System](#llm-tier-system)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Diagrams](#diagrams)
- [Troubleshooting](#troubleshooting)
- [Documentation Index](#documentation-index)

---

## Features

| Feature | Description |
|---|---|
| **Arabic OCR** | Surya OCR with Arabic preprocessing: deskew, contrast enhancement, dictionary correction (max Levenshtein distance 2), Arabic-Indic digit normalization, confidence tiers (high ≥0.85 / medium ≥0.60 / low) |
| **Civil Law RAG** | Unified multi-corpus retrieval across Egyptian Civil Law, Evidence Law, and Civil Procedure Law. Corpus is auto-selected per query via LLM scoring. Optional HyDE expansion. TEI-powered BAAI/bge-m3 embeddings + reranker |
| **Case Document RAG** | Per-case document retrieval with parallel sub-question fan-out, three document selection modes (retrieve_specific / restrict_to / search_all), and automatic rephrase-retry loop |
| **Case Reasoner** | Per-issue legal analysis pipeline: issue extraction → law + fact retrieval → evidence classification → legal application → counterarguments → validation → global consistency check → confidence scoring |
| **Chat Reasoner** | Adaptive planner-executor: generates a multi-step tool-use plan, validates it, executes steps in parallel (respecting dependency ordering), and synthesizes a final answer with replan support |
| **Document Summarization** | 7-node sequential pipeline producing a 7-section judge-facing Arabic brief: parallel document intake, role classification, bullet extraction, aggregation (متفق عليه / محل النزاع), thematic clustering, synthesis, and brief generation |
| **Multi-Agent Supervisor** | 15-node LangGraph state machine: intent classification → context enrichment → agent dispatch → response merging → citation verification → output validation with retry loop → long-term memory read/write → history summarization → audit logging |
| **Async Report Generation** | Background pipeline that runs Summarization + Case Reasoning in sequence and returns results via a polling endpoint |
| **REST API + SSE** | FastAPI with JWT authentication, SSE streaming (progress / result / error / done events), file uploads, case management, conversation history |
| **Long-Term Memory** | Per-case semantic facts and per-judge procedural preferences stored in MongoDB, loaded at turn start and updated at turn end |

---

## Architecture Overview

```
Judge (HTTP Client)
        │
        │  POST /api/v1/query  (JWT Bearer)
        ▼
┌───────────────────┐
│  FastAPI  api/    │  SSE stream ──────────────────────────► Judge
│  app.py           │
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Supervisor  (LangGraph 15-node StateGraph)                │
│                                                            │
│  validate_input → load_long_term_memory                    │
│    → classify_intent → enrich_context                      │
│    → dispatch_agents ──────────────────────────────────┐   │
│         │                                              │   │
│         ▼                                              │   │
│  ┌─────────────────────────────────────────────────┐  │   │
│  │  MCP Transport Layer  (JSON-RPC 2.0 / stdio)    │  │   │
│  │                                                 │  │   │
│  │  ┌──────────────────┐  ┌─────────────────────┐ │  │   │
│  │  │  LegalRAG Server │  │  CaseDoc RAG Server  │ │  │   │
│  │  │  (FastMCP)       │  │  (FastMCP)           │ │  │   │
│  │  └──────────────────┘  └─────────────────────┘ │  │   │
│  └─────────────────────────────────────────────────┘  │   │
│         │                                              │   │
│  ChatReasonerAdapter (direct invocation) ──────────────┘   │
│  SummarizeAdapter    (direct invocation)                   │
│                                                            │
│  merge_responses → verify_citations → validate_output      │
│    → [retry loop, max 3] OR fallback                       │
│    → update_memory → write_long_term_memory                │
│    → [summarize_history?] → audit_log                      │
└────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│  Infrastructure                        │
│  MongoDB · Qdrant · Redis              │
│  MinIO · PostgreSQL · TEI              │
└────────────────────────────────────────┘
```

### Supervisor Turn Lifecycle

1. **validate_input** — sanitize and length-check the judge query; off-topic if injection detected
2. **load_long_term_memory** — fetch per-case semantic facts and per-judge preferences from MongoDB
3. **classify_intent** — medium-tier LLM → one of `civil_law_rag | case_doc_rag | reason | multi | off_topic`
4. **enrich_context** — prefetch case summary and document titles from MongoDB
5. **dispatch_agents** — run selected adapters; for `multi` intent, parallel `Send()` fan-out
6. **merge_responses** — high-tier LLM synthesizes all agent outputs
7. **verify_citations** — cross-check article references
8. **validate_output** — low-tier LLM scores 4 criteria: hallucination · relevance · completeness · coherence
9. **prepare_retry** — on failure, append `validation_feedback` and re-dispatch (max 3 retries)
10. **fallback_response** — generic Arabic fallback after retries exhausted
11. **update_memory** — append turn to MongoDB conversation history
12. **write_long_term_memory** — upsert semantic facts and procedural preferences
13. **summarize_history** — if `messages_since_last_summary` exceeds threshold, compress older turns into `running_summary` using high-tier LLM
14. **audit_log** — write structured audit entry to MongoDB
15. **off_topic_response** — direct refusal path (skips dispatch)

---

## Agent Roster

| Agent | Trigger Intent | Transport | Entry Point |
|---|---|---|---|
| **Civil Law RAG** | `civil_law_rag` | MCP (subprocess stdio) | `mcp_servers.legal_rag_server` → `RAG/legal_rag/service.py` |
| **Case Doc RAG** | `case_doc_rag` | MCP (subprocess stdio) | `mcp_servers.case_doc_server` → `RAG/case_doc_rag/graph.py` |
| **Chat Reasoner** | `reason` | Direct invocation | `chat_reasoner/graph.py` |
| **Summarizer** | `summarize` | Direct invocation | `summarize/graph.py` |
| **OCR** | `ocr` | Direct invocation | `OCR/ocr_pipeline.py` |

### Civil Law RAG

- **Corpora:** Civil Law (القانون المدني) · Evidence Law (قانون الإثبات) · Civil Procedure (قانون المرافعات)
- **Corpus routing:** LLM scores the query against all 3 corpora at runtime — no upfront selection needed
- **Scope classification:** narrows to chapter then section before vector search
- **Retrieval:** BAAI/bge-m3 embeddings (1024-dim COSINE) via remote TEI service → Qdrant `civil_law_docs` collection
- **Reranking:** TEI reranker re-scores candidates before grading
- **Grading:** rule_grader (heuristic) → llm_grader (semantic) → generate_answer or refine loop
- **Caching:** `SemanticCache` keyed by (query, corpus_version, prompt_version)

### Case Doc RAG

- **Scope:** per-case documents only, filtered by `case_id` in Qdrant `case_docs` collection
- **Sub-question fan-out:** query is decomposed into sub-questions; each executes in a parallel branch via LangGraph `Send()`
- **Branch modes:** `retrieve_specific_doc` (short-circuit fetch) · `restrict_to_doc` (scoped search) · `no_doc_specified` (full case search)
- **Rephrase loop:** up to 2 rephrases per sub-question on grading failure

### Chat Reasoner

- **Tools available:** `case_doc_rag` · `civil_law_rag` · `fetch_summary_report`
- **Plan validation:** unknown tools, empty queries, circular deps, >5 steps all rejected (up to 3 validator retries)
- **Parallel execution:** steps with empty `depends_on` fan out simultaneously via `Send()`; dependent steps wait
- **Replan cap:** 2 replans before returning best-effort answer
- **Trace:** every run writes a structured trace to `chat_reasoner_traces` in MongoDB

### Case Reasoner

- **Issue extraction:** high-tier LLM identifies distinct legal issues from the case brief
- **Per-issue branch** (sequential within branch, parallel across issues):
  1. Decompose issue into legal elements
  2. Generate retrieval queries for law + facts
  3. Retrieve law via `civil_law_rag` MCP tool
  4. Retrieve facts via `case_doc_rag` MCP tool
  5. Classify evidence: `established | not_established | disputed | insufficient_evidence`
  6. Apply law with Arabic legal reasoning citing specific articles
  7. Generate counterarguments for both parties
  8. Validate analysis (citation + consistency + completeness)
  9. Package result
- **Post-aggregation:** global consistency check → reconciliation paragraphs → per-issue and case-level confidence scoring → final report generation

### Summarizer

- **Parallel intake:** Node 0 processes documents concurrently via `ThreadPoolExecutor`
- **Document types (7):** صحيفة دعوى · مذكرة دفاع · مذكرة رد · حافظة مستندات · محضر جلسة · حكم تمهيدي · غير محدد
- **Party types (6):** المدعي · المدعى عليه · النيابة · المحكمة · خبير · غير محدد
- **Defendant disambiguation:** merges "المدعى عليه" variants into numbered parties automatically
- **Output:** 7-section structured brief persisted to MongoDB `summaries` collection

---

## Infrastructure Stack

| Service | Purpose | Default |
|---|---|---|
| **MongoDB** | Cases, files, conversations, summaries, case_reasonings, chat_reasoner_traces, audit logs, long-term memory | `localhost:27017` |
| **Qdrant** | Vector store — `civil_law_docs` (1024-dim COSINE) and `case_docs` (per-case filtered) | `localhost:6333` (HTTP) / `6334` (gRPC) |
| **Redis** | Caching, rate limiting (100 req/60s), session data | `localhost:6379` |
| **MinIO** | S3-compatible binary file storage (PDF, images) | `localhost:9000` |
| **PostgreSQL** | Users, roles, audit logging | `localhost:5432` |
| **TEI (Embeddings)** | Remote BAAI/bge-m3 embedding service | `localhost:8080` |
| **TEI (Reranker)** | Remote reranker service | `localhost:8081` |

---

## Prerequisites

- **Python 3.11+**
- **Running services:** MongoDB · Qdrant · Redis · MinIO · PostgreSQL
- **TEI services** for embeddings and reranking (or configure fallback in-process)
- **Google API key** for Gemini 2.5 Flash (all LLM tiers default to Google)

```bash
# Minimum required env vars
GOOGLE_API_KEY=your_google_api_key
JA_API_JWT_SECRET=a-strong-random-secret
```

---

## Quick Start (Local)

### 1. Clone and install

```bash
git clone <repo-url>
cd Code
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# Create .env from template
cp .env.example .env
```

Edit `.env`:
```bash
GOOGLE_API_KEY=your_actual_google_api_key
JA_API_JWT_SECRET=your-strong-jwt-secret-here

# Optional overrides
JA_MONGODB_URI=mongodb://localhost:27017/
JA_MONGODB_DATABASE=judge_assistant
JA_QDRANT_HOST=localhost
```

### 3. Start infrastructure services

```bash
# MongoDB
mongod --dbpath ./data/db

# Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Redis
redis-server

# MinIO
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

### 4. Start the API

```bash
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### 5. Verify

```bash
curl http://localhost:8000/api/v1/health
```

Expected:
```json
{"status": "healthy", "mongo": "connected", "qdrant": "connected"}
```

### 6. Open interactive API docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Configuration

Configuration uses a 3-layer precedence system (highest wins):

```
JA_* environment variables  >  config/settings.local.yaml  >  config/settings.yaml
```

`config/settings.local.yaml` is gitignored and deep-merged over `settings.yaml`. Use it for local overrides without modifying committed files.

### Key Configuration Sections (`config/settings.yaml`)

```yaml
llm:
  high:
    provider: google
    model: gemini-2.5-flash        # complex reasoning, synthesis, summarization
    temperature: 0.0
  medium:
    provider: google
    model: gemini-2.5-flash        # intent classification, extraction
    temperature: 0.0
  low:
    provider: google
    model: gemini-2.5-flash-lite   # output validation, simple routing
    temperature: 0.0

embedding:
  model: BAAI/bge-m3

qdrant:
  host: localhost
  port: 6333
  collection: civil_law_docs
  case_collection: case_docs
  vector_size: 1024

tei:
  embedding_url: http://localhost:8080
  reranker_url: http://localhost:8081
  timeout_seconds: 30

rag:
  civil_law:
    hyde_enabled: false
    scope_chapter_threshold: 0.5
    scope_section_threshold: 0.5

supervisor:
  max_retries: 3
  max_conversation_turns: 20
```

### Environment Variable Convention

Nested YAML keys are flattened with `_` and prefixed with `JA_`:

| YAML path | Environment variable | Example |
|---|---|---|
| `llm.high.model` | `JA_LLM_HIGH_MODEL` | `gemini-2.5-pro` |
| `llm.high.provider` | `JA_LLM_HIGH_PROVIDER` | `google` |
| `mongodb.uri` | `JA_MONGODB_URI` | `mongodb://user:pass@host:27017/` |
| `mongodb.database` | `JA_MONGODB_DATABASE` | `production_db` |
| `qdrant.host` | `JA_QDRANT_HOST` | `qdrant.internal` |
| `api.jwt_secret` | `JA_API_JWT_SECRET` | `strong-random-value` |
| `api.cors_origins` | `JA_API_CORS_ORIGINS` | `https://app.example.com` |
| `api.debug` | `JA_API_DEBUG` | `true` |
| `ocr.use_gpu` | `JA_OCR_USE_GPU` | `true` |
| `tei.embedding_url` | `JA_TEI_EMBEDDING_URL` | `http://tei-host:8080` |

---

## API Reference

All routes are prefixed with `/api/v1/`. Full interactive docs at `/docs` when the server is running.

### Endpoints

| Method | Path | Description | Auth |
|---|---|---|---|
| `GET` | `/health` | Service + DB health check | No |
| `POST` | `/cases` | Create a new case | Yes |
| `GET` | `/cases` | List cases (paginated) | Yes |
| `GET` | `/cases/{case_id}` | Get case details | Yes |
| `PATCH` | `/cases/{case_id}` | Update case title / status | Yes |
| `DELETE` | `/cases/{case_id}` | Soft-delete a case | Yes |
| `POST` | `/files/upload` | Upload file (PDF / image, ≤20MB) | Yes |
| `DELETE` | `/files/{file_id}` | Delete uploaded file | Yes |
| `GET` | `/cases/{case_id}/documents` | List documents for a case | Yes |
| `GET` | `/cases/{case_id}/documents/{doc_id}` | Get document metadata | Yes |
| `GET` | `/cases/{case_id}/documents/{doc_id}/ocr` | Get OCR text + confidence | Yes |
| `PATCH` | `/cases/{case_id}/documents/{doc_id}/ocr` | Submit corrected OCR text + re-index | Yes |
| `POST` | `/query` | Run supervisor query — **SSE stream** | Yes |
| `GET` | `/cases/{case_id}/conversations` | List conversations | Yes |
| `GET` | `/conversations/{conversation_id}` | Get full conversation history | Yes |
| `DELETE` | `/conversations/{conversation_id}` | Delete conversation | Yes |
| `GET` | `/cases/{case_id}/summary` | Get generated case summary | Yes |
| `POST` | `/cases/{case_id}/reports/generate` | Kick off async report pipeline | Yes |
| `GET` | `/cases/{case_id}/reports/{report_id}` | Poll report job status + fetch result | Yes |
| `GET` | `/cases/{case_id}/reports` | List historical report jobs | Yes |

### File Upload Constraints

| Constraint | Value |
|---|---|
| Max file size | 20 MB (`max_upload_bytes: 20971520`) |
| Allowed MIME types | `application/pdf`, `image/png`, `image/jpeg`, `image/tiff`, `image/bmp`, `image/webp`, `image/gif`, `image/heic`, `image/heif` |
| Upload field | `file` (multipart/form-data) |

### Standard Error Envelope

All error responses use this shape:

```json
{
  "error": {
    "code": "CASE_NOT_FOUND",
    "message": "Case not found",
    "status": 404
  }
}
```

| Code | HTTP | Meaning |
|---|---|---|
| `UNAUTHORIZED` | 401 | Missing, expired, or invalid JWT |
| `VALIDATION_ERROR` | 422 | Request schema validation failed |
| `CASE_NOT_FOUND` | 404 | Case doesn't exist or belongs to another user |
| `DOCUMENT_NOT_FOUND` | 404 | Document not found |
| `FILE_NOT_FOUND` | 404 | Uploaded file not found |
| `REPORT_NOT_FOUND` | 404 | Report job not found |
| `NO_DOCUMENTS_FOR_CASE` | 404 | No documents uploaded for this case |
| `SUMMARY_NOT_FOUND` | 404 | No summary generated yet |
| `INVALID_MIME_TYPE` | 400 | File type not in allowed list |
| `FILE_TOO_LARGE` | 400 | File exceeds 20 MB |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Authentication

JWT Bearer tokens, HS256 signed.

**Required claims:** `user_id` (string), `exp` (expiration timestamp)  
**Header:** `Authorization: Bearer <token>`

### Generating a Test Token

```python
import jwt
from datetime import datetime, timedelta, timezone

token = jwt.encode(
    {
        "user_id": "judge_001",
        "exp": datetime.now(timezone.utc) + timedelta(hours=24),
    },
    "your-jwt-secret",   # must match JA_API_JWT_SECRET
    algorithm="HS256",
)
print(token)
```

### Example: End-to-End curl Workflow

```bash
export BASE=http://localhost:8000/api/v1
export TOKEN="your-jwt-token"
export H="Authorization: Bearer $TOKEN"

# 1. Create a case
curl -s -X POST $BASE/cases \
  -H "$H" -H "Content-Type: application/json" \
  -d '{"title": "قضية مدنية رقم 2024/1234"}' | jq .

# 2. Upload a scanned document
CASE_ID="<case_id from step 1>"
FILE_ID=$(curl -s -X POST $BASE/files/upload \
  -H "$H" -F "file=@/path/to/document.pdf" | jq -r .file_id)

# 3. Query the case (SSE stream)
curl -N -X POST $BASE/query \
  -H "$H" -H "Content-Type: application/json" \
  -d "{\"case_id\": \"$CASE_ID\", \"query\": \"ما هي المواد القانونية المنطبقة على هذه القضية؟\"}"

# 4. Generate full report (async)
JOB_ID=$(curl -s -X POST $BASE/cases/$CASE_ID/reports/generate \
  -H "$H" | jq -r .job_id)

# 5. Poll report status
curl -s $BASE/cases/$CASE_ID/reports/$JOB_ID -H "$H" | jq .status
```

---

## Streaming Queries (SSE)

`POST /api/v1/query` returns `text/event-stream`. The client receives a sequence of typed events.

### Event Types

**`progress`** — emitted as each supervisor node completes:
```
event: progress
data: {"step": "classify_intent", "status": "done", "intent": "civil_law_rag"}
```

**`result`** — the final validated Arabic answer (emitted once):
```
event: result
data: {
  "final_response": "وفقاً للمادة 163 من القانون المدني ...",
  "sources": [{"article": "163", "title": "المادة 163", "book": "الالتزامات"}],
  "intent": "civil_law_rag",
  "agents_used": ["civil_law_rag"],
  "conversation_id": "conv_abc123",
  "turn_count": 1
}
```

**`error`** — if something goes wrong:
```
event: error
data: {"detail": "An internal error occurred while processing the query"}
```

**`done`** — always the last event:
```
event: done
data: {}
```

### Multi-Turn Conversations

Pass `conversation_id` from a previous `result` event to continue a thread. The supervisor loads prior history from MongoDB and uses `running_summary` to keep context within token limits.

### JavaScript Integration

```javascript
async function queryCase(caseId, question, conversationId = null) {
  const response = await fetch('/api/v1/query', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      case_id: caseId,
      query: question,
      conversation_id: conversationId,
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event: result')) continue;
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (data.final_response) {
          displayAnswer(data.final_response);
          return data.conversation_id;   // pass to next turn
        }
      }
    }
  }
}
```

---

## Report Generation

The report pipeline runs summarization and case reasoning asynchronously as a background job.

```
POST /api/v1/cases/{case_id}/reports/generate
  → HTTP 202 + { "job_id": "..." }

GET  /api/v1/cases/{case_id}/reports/{job_id}
  → { "status": "pending|running|completed|failed", "summary": {...}, "case_reasoning": {...} }
```

**Pipeline sequence (background):**
1. Summarization pipeline (7-node LangGraph) → upsert to `summaries`
2. Case Reasoner pipeline → upsert to `case_reasonings`
3. Job status updated to `completed`

**Polling recommendation:** poll every 5–10 seconds; typical completion time is 60–120 seconds depending on document count and issue complexity.

---

## MCP Transport Layer

The Civil Law RAG and Case Doc RAG agents run as isolated FastMCP subprocess servers, communicating over `stdin`/`stdout` using newline-delimited JSON-RPC 2.0.

```
Supervisor process
  │
  ├─── MCPClient("mcp_servers.legal_rag_server")
  │       │  threading.Lock (serializes calls)
  │       │  auto-respawn on transport failure (max 1 respawn)
  │       └─► subprocess: python -m mcp_servers.legal_rag_server
  │               JSON-RPC: tools/call → search_legal_corpus(query, corpus)
  │
  └─── MCPClient("mcp_servers.case_doc_server")
          │  threading.Lock
          └─► subprocess: python -m mcp_servers.case_doc_server
                  JSON-RPC: tools/call → search_case_docs(query, case_id, ...)
```

**Timeouts:** call timeout 120s · handshake timeout 600s (covers first-run model download)  
**Warmup:** both servers warm up their LangGraph graph + Qdrant vectorstore + reranker at subprocess boot, before accepting any requests.

---

## LLM Tier System

Three tiers map to task complexity:

| Tier | Model | Used For |
|---|---|---|
| **high** | `gemini-2.5-flash` | Legal reasoning, response synthesis, summarization, case briefing, chat reasoner planning/synthesis |
| **medium** | `gemini-2.5-flash` | Intent classification, document classification, query rewriting, relevance grading |
| **low** | `gemini-2.5-flash-lite` | Output validation, off-topic routing, thematic clustering |

```python
from config import get_llm

llm_high   = get_llm("high")
llm_medium = get_llm("medium")
llm_low    = get_llm("low")
```

All tiers use `temperature: 0.0` by default for deterministic legal outputs.

---

## Testing

The test suite has 100+ tests across 6 categories.

### Test Markers

| Marker | Description |
|---|---|
| `unit` | Pure function tests, no external I/O |
| `integration` | Tests against real MongoDB + Qdrant + LLMs |
| `behavioral` | E2E multi-turn conversation and agent dispatch scenarios |
| `performance` | Memory profiling, throughput, cache speedup |
| `regression` | Boundary cases and known failure modes |
| `llm_eval` | LLM-as-judge quality evaluations (RAGAS, custom rubrics) |
| `expensive` | Long-running tests (E2E multi-turn, full pipelines) |

### Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run unit tests only (fast, no LLM calls)
pytest -m unit -v

# Run integration tests (requires all services running)
pytest -m integration -v

# Run the full suite
pytest -v

# Skip expensive LLM-eval tests
pytest -m "not expensive and not llm_eval" -v

# Run a specific subsystem
pytest tests/supervisor/ -v
pytest tests/rag/ -v
pytest tests/summarizer/ -v

# Run with parallel workers
pytest -n auto -m unit
```

### Key Quality Thresholds

| Metric | Threshold | Test |
|---|---|---|
| Supervisor routing accuracy | ≥ 85% | `test_routing_accuracy.py` |
| RAGAS faithfulness | ≥ 0.80 | `test_rag_quality.py` |
| RAGAS context recall | ≥ 0.75 | `test_rag_quality.py` |
| Answer consistency (multi-turn) | ≥ 0.80 | `test_answer_consistency.py` |
| Cache speedup | ≥ 2× | `test_cache_speedup.py` |
| Memory peak (summarizer) | ≤ 500 MB | `test_memory_leak.py` |
| Memory growth ratio | ≤ 1.2× | `test_memory_leak.py` |
| Ingestion pipeline runtime | ≤ 120s | `test_ingestion_pipeline.py` |

### Test Structure

```
tests/
  supervisor/
    unit/           — state reducers, router functions, node logic
    integration/    — full supervisor turns with real LLM + MongoDB
    behavioral/     — multi-turn conversation, intent routing accuracy
    performance/    — memory profiling, throughput
    e2e/            — 3-turn conversation, history trimming, retry count reset
  rag/
    unit/           — corpus config, cache, validation logic
    integration/    — retrieval quality, filter application
    behavioral/     — routing accuracy, boundary cases
    performance/    — cache speedup, vectorstore load
  summarizer/
    unit/           — bullet extraction, aggregation, disambiguation
    integration/    — full 7-node pipeline
    behavioral/     — party classification, theme coverage
    performance/    — memory under load
    llm_eval/       — LLM-as-judge quality scoring (EV-01 to EV-08)
  case_reasoner/
    unit/           — issue extraction, evidence classification
    integration/    — per-issue branch, consistency check
    llm_eval/       — CR-EV-01 to CR-EV-12 quality dimensions
```

---

## Project Structure

```
Code/
├── api/                          # FastAPI application
│   ├── app.py                    #   create_app() factory, lifespan, DB init
│   ├── dependencies.py           #   JWT auth, DB, settings injection
│   ├── errors.py                 #   error code constants
│   ├── routers/                  #   route handlers
│   │   ├── cases.py
│   │   ├── files.py
│   │   ├── documents.py
│   │   ├── query.py              #   SSE streaming endpoint
│   │   ├── conversations.py
│   │   ├── summaries.py
│   │   ├── reports.py            #   async report generation
│   │   └── health.py
│   ├── schemas/                  #   Pydantic request/response models
│   ├── services/                 #   business logic
│   │   ├── query_service.py      #   supervisor graph invocation
│   │   ├── report_service.py     #   background report job
│   │   └── ...
│   └── db/                       #   Motor/Qdrant/Redis/MinIO/PostgreSQL clients
│
├── Supervisor/                   # Multi-agent supervisor
│   ├── graph.py                  #   15-node LangGraph StateGraph
│   ├── state.py                  #   SupervisorState TypedDict + Pydantic schemas
│   ├── prompts.py                #   LLM prompt templates
│   ├── agents/                   #   adapter registry + all adapters
│   │   ├── civil_law_rag_adapter.py
│   │   ├── case_doc_rag_adapter.py
│   │   ├── chat_reasoner_adapter.py
│   │   ├── summarize_adapter.py
│   │   └── ocr_adapter.py
│   └── nodes/                    #   one file per graph node
│       ├── classify_intent.py
│       ├── dispatch_agents.py
│       ├── merge_responses.py
│       ├── validate_output.py
│       ├── update_memory.py
│       ├── load_long_term_memory.py
│       ├── write_long_term_memory.py
│       ├── summarize_history.py
│       ├── audit_log.py
│       ├── enrich_context.py
│       ├── verify_citations.py
│       ├── prepare_retry.py
│       ├── fallback.py
│       ├── off_topic.py
│       └── validate_input.py
│
├── mcp_servers/                  # MCP subprocess servers
│   ├── lifecycle.py              #   start_mcp_servers(), get_client()
│   ├── client.py                 #   MCPClient (JSON-RPC 2.0 / stdio)
│   ├── legal_rag_server.py       #   FastMCP: search_legal_corpus
│   ├── case_doc_server.py        #   FastMCP: search_case_docs
│   └── errors.py                 #   ErrorCode enum, ToolError
│
├── RAG/
│   ├── legal_rag/                # Unified multi-corpus civil law RAG
│   │   ├── graph.py              #   11-node LangGraph (corpus-agnostic)
│   │   ├── service.py            #   ask_question() public entry point
│   │   ├── corpus_config.py      #   CorpusConfig frozen dataclass
│   │   ├── cache.py              #   SemanticCache
│   │   ├── state.py
│   │   ├── nodes/                #   corpus_router, preprocessor, scope_classifier, etc.
│   │   ├── retrieval/            #   embeddings.py, vectorstore.py, reranker.py
│   │   ├── civil_law_rag/        #   CIVIL_LAW_CORPUS config + docs
│   │   ├── evidence_rag/         #   EVIDENCE_CORPUS config + docs
│   │   └── procedures_rag/       #   PROCEDURES_CORPUS config + docs
│   └── case_doc_rag/             # Per-case document RAG
│       ├── graph.py              #   main graph + branch sub-graph
│       ├── routers.py            #   branchDocSelectorRouter, proceedRouter
│       ├── state.py              #   AgentState, SubQuestionState
│       └── nodes/                #   query, selection, retrieval, generation nodes
│
├── summarize/                    # 7-node summarization pipeline
│   ├── graph.py                  #   create_pipeline(), sequential + ThreadPoolExecutor
│   ├── state.py                  #   SummarizationState, disambiguate_defendants()
│   └── nodes/                    #   intake, classifier, extractor, aggregator,
│                                 #   clustering, synthesis, brief
│
├── chat_reasoner/                # Adaptive planner-executor
│   ├── graph.py                  #   build_chat_reasoner_graph()
│   ├── state.py                  #   ChatReasonerState, ALLOWED_TOOLS, Pydantic schemas
│   ├── tools.py                  #   tool implementations
│   ├── prompts.py
│   └── nodes/                    #   planner, plan_validator, executor, step_worker,
│                                 #   collector, synthesizer, replanner, trace_writer
│
├── CR/                           # Case Reasoner
│   ├── graph.py                  #   build_case_reasoner_graph(), build_issue_branch()
│   ├── pipeline.py               #   run_case_reasoning() public entry, CaseReasoningResult
│   ├── state.py                  #   CaseReasonerState, IssueAnalysisState
│   ├── routers.py
│   └── nodes/                    #   extraction, decomposition, retrieval, evidence,
│                                 #   application, counterargument, validation, package,
│                                 #   aggregation, consistency, confidence, report
│
├── OCR/                          # Arabic OCR pipeline
│   ├── ocr_pipeline.py           #   full pipeline orchestration
│   ├── engine.py                 #   Surya OCR wrapper
│   ├── preprocessor.py           #   deskew, denoise, contrast, resolution check
│   ├── postprocessor.py          #   dict correction, digit normalization
│   └── schemas.py
│
├── config/                       # Centralized configuration
│   ├── settings.yaml             #   committed defaults
│   ├── __init__.py               #   AppConfig singleton, get_llm() factory
│   ├── api.py                    #   FastAPI Settings (Pydantic)
│   ├── supervisor.py             #   supervisor constants
│   ├── legal_rag.py              #   RAG constants
│   └── ocr.py                    #   OCR constants
│
├── tests/                        # Full test suite
│   ├── supervisor/
│   ├── rag/
│   ├── summarizer/
│   └── case_reasoner/
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md
│   ├── AGENTS.md
│   ├── API.md
│   ├── DATABASE.md
│   ├── SETUP.md
│   ├── TESTING.md
│   ├── DECISIONS.md
│   ├── TROUBLESHOOTING.md
│   └── diagrams/                 # Mermaid diagrams
│       ├── architecture.mmd
│       ├── supervisor.mmd
│       ├── rag.mmd
│       ├── summarizer.mmd
│       ├── chat_reasoner.mmd
│       ├── case_reasoner.mmd
│       ├── sequences/            # 9 sequence diagrams
│       └── activity/             # 11 activity/flow diagrams
│
├── Methodology.md                # Academic methodology document
├── requirements.txt
└── config/settings.yaml          # Default configuration
```

---

## Diagrams

Mermaid diagrams are in `docs/diagrams/`:

| File | Type | Shows |
|---|---|---|
| `architecture.mmd` | `graph TD` | Full system: FastAPI → Supervisor → agents → MCP → infra |
| `supervisor.mmd` | `classDiagram` | SupervisorState, IntentEnum, ValidationResult, adapters |
| `rag.mmd` | `classDiagram` | LegalRAGGraph, CaseDocRAGGraph, MCPClient, both servers |
| `summarizer.mmd` | `classDiagram` | SummarizationState, 4 enums, SummarizationGraph |
| `chat_reasoner.mmd` | `classDiagram` | ChatReasonerState, Plan/PlanStep/ToolEnum hierarchy |
| `case_reasoner.mmd` | `classDiagram` | CaseReasonerState, IssueAnalysisState, branch + pipeline |
| `sequences/query_lifecycle.mmd` | `sequenceDiagram` | Full turn: POST /query → SSE → audit_log |
| `sequences/chat_reasoner_flow.mmd` | `sequenceDiagram` | planner → fan-out → synthesizer/replan loop |
| `sequences/legal_rag_routing.mmd` | `sequenceDiagram` | corpus_router → scope → embed → rerank → grade → cache |
| `sequences/case_doc_rag_flow.mmd` | `sequenceDiagram` | questionRewriter → parallel branches → mergeAnswers |
| `sequences/memory_lifecycle.mmd` | `sequenceDiagram` | load LTM → turn → summarize_history → write LTM |
| `sequences/report_generation.mmd` | `sequenceDiagram` | 202 → background task → poll |
| `activity/intent_routing.mmd` | `flowchart TD` | validate → classify → 5-way intent dispatch |
| `activity/validation_retry.mmd` | `flowchart TD` | 4-criteria check, partial_pass caveat, retry loop |
| `activity/case_doc_rag_branch.mmd` | `flowchart TD` | 3 branch modes, rephrase loop |
| `activity/legal_rag_retrieval.mmd` | `flowchart TD` | cache hit/miss, HyDE, rule/LLM grader |
| `activity/summarization_activity.mmd` | `flowchart TD` | 7 nodes, ThreadPoolExecutor, disambiguate |
| `activity/chat_reasoner_activity.mmd` | `flowchart TD` | plan → validate → parallel execute → synthesize/replan |
| `activity/case_reasoner_activity.mmd` | `flowchart TD` | issue fan-out, consistency check, confidence |
| `activity/document_ingestion_activity.mmd` | `flowchart TD` | MIME/size gates, OCR tiers, embed → Qdrant |
| `activity/report_generation_activity.mmd` | `flowchart TD` | Background task + polling loop |

---

## Troubleshooting

### MCP server fails to start / handshake timeout

The first spawn of `legal_rag_server` or `case_doc_server` can take 2–10 minutes on a cold start because it downloads and warms up the BAAI/bge-m3 model and the reranker. The handshake timeout is 600s by default. Check subprocess stderr (it inherits the parent's stderr) for progress logs.

```bash
# Confirm TEI services are reachable
curl http://localhost:8080/health
curl http://localhost:8081/health
```

### Qdrant connection refused

Qdrant must be running before the API starts. The API's lifespan hooks call `ensure_indexed()` at startup.

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### MongoDB ObjectId serialization errors

Ensure Motor is version 3.x and that `bson` is installed from the `pymongo` package, not the standalone `bson` package (they conflict).

### OCR confidence always low

- Minimum DPI is 150 (configurable via `JA_OCR_PREPROCESSING_MIN_DPI`)
- Enable GPU: `JA_OCR_USE_GPU=true` (requires CUDA toolkit)
- Check `surya_batch_size` — reduce to 1 if GPU OOM errors appear

### `validator_error` in every response

This usually means the low-tier LLM (`gemini-2.5-flash-lite`) is rate-limited or the `GOOGLE_API_KEY` is missing. Check logs for the specific LLM exception message.

### History grows without bound

The `summarize_history` node fires only when `messages_since_last_summary > threshold`. Check `config/supervisor.py` for the `MAX_CONVERSATION_TURNS` constant. If conversations use the non-persistent graph (`get_app()`), history is not checkpointed between API calls — pass `conversation_id` on every request to use the persistent graph.

### Redis connection errors (non-fatal)

Redis is used for optional caching and rate limiting. If Redis is unavailable, the API continues to function without caching. Errors are logged at WARNING level.

---

## Documentation Index

| Document | Location | Contents |
|---|---|---|
| Architecture | `docs/ARCHITECTURE.md` | Full graph diagrams, ADR pointers, component design |
| Agents | `docs/AGENTS.md` | All 5 agents: triggers, schemas, retrieval strategies |
| API Reference | `docs/API.md` | All endpoints, request/response schemas, SSE guide |
| Database | `docs/DATABASE.md` | MongoDB collections, Qdrant indexes, Redis, MinIO, PostgreSQL |
| Setup | `docs/SETUP.md` | Full env var reference, Docker Compose, local dev |
| Testing | `docs/TESTING.md` | Test suite structure, markers, writing new tests, CI/CD |
| Decisions | `docs/DECISIONS.md` | 7 Architecture Decision Records with rationale |
| Troubleshooting | `docs/TROUBLESHOOTING.md` | 10 known issues with fixes and debugging guide |
| Methodology | `Methodology.md` | Academic system methodology (Arabic) |
| Configuration | `config/settings.yaml` | Annotated default configuration |
