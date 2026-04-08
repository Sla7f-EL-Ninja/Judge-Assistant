# Judge Assistant

An AI-powered legal assistant built for Egyptian judicial workflows. It combines Arabic OCR, document classification, civil law retrieval-augmented generation (RAG), case reasoning, and multi-document summarization into a unified REST API, orchestrated by a LangGraph-based multi-agent supervisor.

## Table of Contents

- [Documentation](#documentation)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start with Docker](#quick-start-with-docker)
- [Local Development Setup](#local-development-setup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Authentication](#authentication)
- [Streaming Queries (SSE)](#streaming-queries-sse)
- [Testing](#testing)
- [Streamlit Testing UI](#streamlit-testing-ui)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Documentation

Comprehensive project documentation lives in the [`docs/`](docs/) directory:

| Document | Description |
|---|---|
| [Glossary](docs/GLOSSARY.md) | Bilingual reference of 30+ terms (Arabic/English) covering system components, legal domain, and technical concepts |
| [Architecture](docs/ARCHITECTURE.md) | System design, component overview, supervisor graph flow, state management, LLM tier system |
| [Agents](docs/AGENTS.md) | All 5 specialist agents: purpose, triggers, retrieval strategies, input/output schemas |
| [Database](docs/DATABASE.md) | MongoDB collections, Qdrant vectors, Redis caching, MinIO storage, PostgreSQL user management |
| [API Reference](docs/API.md) | Every endpoint documented with request/response schemas, SSE streaming guide, error codes |
| [Setup](docs/SETUP.md) | Environment variables, Docker Compose, local development, common setup errors |
| [Testing](docs/TESTING.md) | Test suite structure, running tests, writing new tests, CI/CD |
| [Deployment](docs/DEPLOYMENT.md) | Docker Compose services, Dockerfile details, health checks, scaling, backups |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | 10 known issues with symptoms, root causes, and fixes; debugging guide |
| [Decisions](docs/DECISIONS.md) | 7 Architecture Decision Records (ADRs) with context, rationale, and consequences |
| [Contributing](docs/CONTRIBUTING.md) | Developer onboarding, adding agents/endpoints, code style, PR checklist |

## Features

- **Arabic OCR Pipeline** -- Extract text from scanned legal documents (PDF, PNG, JPEG, TIFF, BMP) using Surya OCR with Arabic-specific preprocessing, deskewing, denoising, confidence scoring, and dictionary-based post-processing.
- **Civil Law RAG** -- Query Egyptian civil law articles using retrieval-augmented generation with ChromaDB vector store and BGE-M3 embeddings. Supports query rewriting, classification, and multi-pass retrieval.
- **Case Document RAG** -- Classify and query case-specific documents uploaded by the user.
- **Case Reasoning** -- Apply legal reasoning to case facts against relevant law articles.
- **Document Summarization** -- Multi-node pipeline that generates structured summaries of legal documents.
- **Multi-Agent Supervisor** -- LangGraph state machine that classifies user intent and dispatches to one or more specialist agents, merges results, validates output, and manages conversation memory.
- **REST API** -- FastAPI application with JWT authentication, SSE streaming, file uploads, case management, and conversation history.
- **Docker Support** -- One-command deployment with Docker Compose (MongoDB + API + optional Streamlit UI).

## Architecture

```
Client (Express frontend / Streamlit)
         |
         v
+--------------------+
|   FastAPI (api/)   |  <-- JWT auth, file upload, SSE streaming
+--------+-----------+
         |
+--------v-----------+
|   Supervisor Agent  |  <-- LangGraph state machine
|   (multi-agent)     |     Intent classification -> dispatch -> merge -> validate
+--------+-----------+
         |
         +-------------------+-------------------+-------------------+
         |                   |                   |                   |
    +----v----+        +-----v-----+       +-----v-----+      +----v----+
    |   OCR   |        | Civil Law |       | Case Doc  |      |Summarize|
    | Pipeline|        |    RAG    |       |    RAG    |      |  Agent  |
    +---------+        +-----------+       +-----------+      +---------+
         |                   |                   |                   |
    +----v----+        +-----v-----+       +-----v-----+      +----v----+
    |  Surya  |        | ChromaDB  |       | ChromaDB  |      |  LLM    |
    |  Engine |        | + BGE-M3  |       | + BGE-M3  |      | (Groq)  |
    +---------+        +-----------+       +-----------+      +---------+

Data stores:
  - MongoDB: cases, files metadata, conversations, summaries
  - ChromaDB: vector embeddings for civil law articles and case documents
```

### How a Query Flows

1. User sends a query via `POST /api/v1/query` with a `case_id`
2. The API streams Server-Sent Events (SSE) back to the client
3. The **Supervisor** classifies the intent (civil law, case doc, OCR, summarize, reasoning, or multi-agent)
4. Relevant agents are dispatched in parallel where possible
5. Results are merged and validated
6. Conversation memory is updated in MongoDB
7. The final response is streamed as an SSE `result` event

## Prerequisites

- **Docker and Docker Compose** (recommended) -- or Python 3.11+ with MongoDB for local dev
- **API keys** for at least one LLM provider:
  - [Groq API key](https://console.groq.com/) (default provider for high/medium tiers)
  - [Google AI API key](https://makersuite.google.com/app/apikey) (default for low tier)

## Quick Start with Docker

This is the fastest way to get everything running. Docker Compose starts MongoDB, the API, and optionally the Streamlit testing UI.

### Step 1: Clone the repository

```bash
git clone https://github.com/hassann16541-create/Judge-Assistant.git
cd Judge-Assistant
```

### Step 2: Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in your editor and set the required values:

```bash
# REQUIRED -- at least one LLM provider key
GROQ_API_KEY=gsk_your_actual_groq_key_here
GOOGLE_API_KEY=your_actual_google_key_here

# REQUIRED -- change this to a strong random string in production
JA_API_JWT_SECRET=my-super-secret-jwt-key
```

### Step 3: Start the services

```bash
# Start MongoDB + API
docker compose up -d
```

This will:
- Pull the MongoDB 7 image
- Build the API Docker image (first time takes a few minutes due to ML dependencies)
- Start MongoDB with a health check
- Start the API once MongoDB is healthy
- Download embedding models on first query (~400MB, cached in a Docker volume)

### Step 4: Verify it's running

```bash
# Check service status
docker compose ps

# Check the health endpoint
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "mongo": "connected",
  "chroma": "connected"
}
```

### Step 5: Open the interactive docs

Navigate to `http://localhost:8000/docs` in your browser for Swagger UI, or `http://localhost:8000/redoc` for ReDoc.

### Step 6 (Optional): Start the Streamlit testing UI

```bash
docker compose --profile testing up -d
```

Open `http://localhost:8501` in your browser. The Streamlit app provides a GUI for testing all API endpoints.

### Docker Management Commands

The included `Makefile` provides shortcuts:

| Command | What it does |
|---------|-------------|
| `make up` | Start MongoDB + API |
| `make up-all` | Start everything including Streamlit |
| `make down` | Stop all services |
| `make down-clean` | Stop all services and delete all data (MongoDB volumes, etc.) |
| `make logs` | Tail the API container logs |
| `make build` | Rebuild Docker images from scratch (no cache) |
| `make shell` | Open a bash shell inside the running API container |
| `make mongo-shell` | Open the MongoDB shell (`mongosh`) |
| `make help` | Show all available make targets |

### Docker Services Overview

| Service | Port | Description |
|---------|------|-------------|
| `mongo` | 27017 | MongoDB 7 with persistent volume |
| `api` | 8000 | FastAPI application |
| `streamlit` | 8501 | Testing UI (only with `--profile testing`) |

### Docker Volumes

| Volume | Purpose |
|--------|---------|
| `mongo_data` | MongoDB data persistence |
| `huggingface_cache` | Cached embedding models (prevents re-download) |
| `./uploads` | Uploaded files (bind mount) |
| `./chroma_data` | ChromaDB vector store data (bind mount) |

## Local Development Setup

For development without Docker, you need Python 3.11+ and a running MongoDB instance.

### Step 1: Install MongoDB

**macOS (Homebrew):**
```bash
brew tap mongodb/brew
brew install mongodb-community@7.0
brew services start mongodb-community@7.0
```

**Ubuntu/Debian:**
```bash
# Follow the official MongoDB installation guide:
# https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
sudo systemctl start mongod
```

**Windows:**
Download and install from https://www.mongodb.com/try/download/community

### Step 2: Set up Python environment

```bash
# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Step 3: Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys (same as Docker setup)
```

Optionally create a local config override:
```yaml
# config/settings.local.yaml (gitignored)
mongodb:
  uri: mongodb://localhost:27017/
api:
  debug: true
llm:
  high:
    provider: google
    model: gemini-1.5-pro  # Use Google if you don't have a Groq key
```

### Step 4: Run the API

```bash
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables hot-reloading on code changes.

### Step 5: Verify

```bash
curl http://localhost:8000/api/v1/health
```

## Configuration

Judge Assistant uses a centralized YAML-based configuration system. See [CONFIG.md](CONFIG.md) for the full reference.

### Config Precedence (highest wins)

```
Environment variables (JA_*)  >  settings.local.yaml  >  settings.yaml
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `config/settings.yaml` | Default configuration (committed to repo) |
| `config/settings.local.yaml` | Local overrides (gitignored) |
| `.env` | API keys and secrets (gitignored) |

### Environment Variable Override Convention

Nested YAML keys are joined with `_` and upper-cased, prefixed with `JA_`:

| YAML Path | Environment Variable | Example Value |
|-----------|---------------------|---------------|
| `llm.high.model` | `JA_LLM_HIGH_MODEL` | `llama-3.3-70b-versatile` |
| `llm.high.provider` | `JA_LLM_HIGH_PROVIDER` | `groq` |
| `mongodb.uri` | `JA_MONGODB_URI` | `mongodb://mongo:27017/` |
| `mongodb.database` | `JA_MONGODB_DATABASE` | `Rag` |
| `api.debug` | `JA_API_DEBUG` | `true` |
| `api.cors_origins` | `JA_API_CORS_ORIGINS` | `http://localhost:3000` |
| `ocr.language` | `JA_OCR_LANGUAGE` | `ar` |
| `ocr.use_gpu` | `JA_OCR_USE_GPU` | `false` |
| `embedding.model` | `JA_EMBEDDING_MODEL` | `BAAI/bge-m3` |

### LLM Tier System

Models are organized into three tiers. Each tier can use a different provider and model:

| Tier | Default Provider | Default Model | Used For |
|------|-----------------|---------------|----------|
| **high** | Groq | llama-3.3-70b-versatile | Legal reasoning, response merging, summarization, RAG answers |
| **medium** | Groq | llama-3.3-70b-versatile | Intent classification, document classification, query rewriting |
| **low** | Google | gemini-1.5-flash | Output validation, off-topic detection, simple routing |

Use the factory function to get a model:

```python
from config import get_llm

llm = get_llm("high")                    # Default high-tier model
llm = get_llm("medium", temperature=0.3) # With override
```

## API Reference

All routes are prefixed with `/api/v1/`. Full interactive documentation is available at `/docs` (Swagger) and `/redoc` when the API is running.

### Endpoints

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| GET | `/health` | Service health check | No |
| POST | `/cases` | Create a new case | Yes |
| GET | `/cases` | List cases (paginated) | Yes |
| GET | `/cases/{case_id}` | Get case details | Yes |
| PATCH | `/cases/{case_id}` | Update case title/status | Yes |
| DELETE | `/cases/{case_id}` | Soft-delete a case | Yes |
| POST | `/files/upload` | Upload a file (PDF/image) | Yes |
| POST | `/cases/{case_id}/documents` | Ingest uploaded files (OCR + classification) | Yes |
| POST | `/query` | Run supervisor query (SSE stream) | Yes |
| GET | `/cases/{case_id}/conversations` | List conversations for a case | Yes |
| GET | `/conversations/{conversation_id}` | Get full conversation history | Yes |
| DELETE | `/conversations/{conversation_id}` | Delete a conversation | Yes |
| GET | `/cases/{case_id}/summary` | Get case summary | Yes |

### Pagination

List endpoints accept `skip` (default: 0) and `limit` (default: 20, max: 100) query parameters:

```bash
GET /api/v1/cases?skip=0&limit=10
```

Response includes a `total` count:
```json
{
  "cases": [...],
  "total": 42
}
```

### File Upload Constraints

| Constraint | Value |
|-----------|-------|
| Max file size | 20 MB |
| Allowed MIME types | `application/pdf`, `image/png`, `image/jpeg`, `image/tiff`, `image/bmp`, `image/webp` |
| Upload field name | `file` (multipart form) |

### Error Format

All errors follow a standard envelope:

```json
{
  "error": {
    "code": "CASE_NOT_FOUND",
    "detail": "Case not found",
    "status": 404
  }
}
```

| Error Code | HTTP Status | Meaning |
|------------|-------------|---------|
| `UNAUTHORIZED` | 401 | Missing, expired, or invalid JWT |
| `VALIDATION_ERROR` | 422 | Request body/query validation failed |
| `CASE_NOT_FOUND` | 404 | Case doesn't exist or belongs to another user |
| `CONVERSATION_NOT_FOUND` | 404 | Conversation doesn't exist |
| `FILE_NOT_FOUND` | 404 | Uploaded file not found |
| `SUMMARY_NOT_FOUND` | 404 | No summary generated for this case |
| `INVALID_MIME_TYPE` | 400 | File type not in allowed list |
| `FILE_TOO_LARGE` | 400 | File exceeds 20 MB limit |
| `NO_FIELDS_TO_UPDATE` | 400 | PATCH body has no updatable fields |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

## Authentication

The API uses **JWT Bearer tokens** with HS256 signing.

### Token Requirements

- **Algorithm:** HS256
- **Required claims:** `user_id` (string), `exp` (expiration timestamp)
- **Shared secret:** Set via `JA_API_JWT_SECRET` environment variable

### Usage

Include the token in the `Authorization` header:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

### Generating a Token (for testing)

```python
from jose import jwt
from datetime import datetime, timedelta

token = jwt.encode(
    {
        "user_id": "test_user_001",
        "exp": datetime.utcnow() + timedelta(hours=24),
    },
    "your-jwt-secret",
    algorithm="HS256",
)
print(token)
```

Or use the Streamlit UI which auto-generates tokens.

### Example: Full Workflow with curl

```bash
# Set your variables
export BASE=http://localhost:8000/api/v1
export TOKEN="your-jwt-token-here"
export AUTH="Authorization: Bearer $TOKEN"

# 1. Check health
curl $BASE/health

# 2. Create a case
curl -X POST $BASE/cases \
  -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{"title": "Civil Dispute Case #2024-1234"}'

# 3. Upload a document (replace case_id and file path)
curl -X POST $BASE/files/upload \
  -H "$AUTH" \
  -F "file=@/path/to/legal-document.pdf"

# 4. Ingest the document into the case
curl -X POST "$BASE/cases/{case_id}/documents" \
  -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{"file_ids": ["{file_id}"]}'

# 5. Query the case (SSE stream)
curl -N -X POST $BASE/query \
  -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d '{"case_id": "{case_id}", "query": "What are the relevant civil law articles?"}'

# 6. Get conversation history
curl "$BASE/cases/{case_id}/conversations" -H "$AUTH"

# 7. Get case summary
curl "$BASE/cases/{case_id}/summary" -H "$AUTH"
```

## Streaming Queries (SSE)

The `POST /api/v1/query` endpoint returns a `text/event-stream` response with these event types:

### Event Types

**`progress`** -- Emitted as each supervisor node completes:
```
event: progress
data: {"step": "classify_intent", "status": "done"}
```

**`result`** -- The final answer (emitted once):
```
event: result
data: {"final_response": "...", "sources": [...], "intent": "civil_law_rag", "agents_used": ["civil_law_rag"], "conversation_id": "conv_xxx"}
```

**`error`** -- If something goes wrong:
```
event: error
data: {"detail": "An internal error occurred while processing the query"}
```

**`done`** -- Always the last event:
```
event: done
data: {}
```

### JavaScript Example (Frontend Integration)

```javascript
const eventSource = new EventSource('/api/v1/query', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    case_id: caseId,
    query: userQuestion,
    conversation_id: existingConversationId, // optional, for follow-ups
  }),
});

eventSource.addEventListener('progress', (e) => {
  const data = JSON.parse(e.data);
  console.log(`Step: ${data.step} - ${data.status}`);
});

eventSource.addEventListener('result', (e) => {
  const data = JSON.parse(e.data);
  displayAnswer(data.final_response);
  saveConversationId(data.conversation_id);
});

eventSource.addEventListener('error', (e) => {
  const data = JSON.parse(e.data);
  showError(data.detail);
});

eventSource.addEventListener('done', () => {
  eventSource.close();
});
```

## Testing

The project includes an integration test suite that tests against real MongoDB and the full Supervisor LangGraph -- no mocks.

### Running Tests

```bash
# Install test dependencies
pip install -r api/tests/requirements-test.txt

# Configure test environment
cp .env.example .env
# Make sure MONGO_URI, JWT_SECRET, GROQ_API_KEY are set

# Run the full suite (requires running MongoDB)
cd api && python -m pytest tests/ -v

# Run only fast tests (skip document ingestion + LLM queries)
cd api && python -m pytest tests/ -v -k "not document and not query and not conversation and not summary"

# Run a single test file
cd api && python -m pytest tests/test_02_health.py -v

# Run with detailed failure output
cd api && python -m pytest tests/ -v --tb=short
```

### Test Coverage

| Area | What's Tested |
|------|--------------|
| **Auth** | Missing header, bad format, tampered token, expired token, missing user_id |
| **Health** | MongoDB connectivity, Chroma connectivity, overall status |
| **Cases** | Create, list, paginate, get, update title, update status, invalid status, empty patch, user isolation, delete |
| **Files** | PDF upload, PNG upload, wrong MIME rejected, oversized rejected |
| **Documents** | Ingest real PDF, case documents updated, nonexistent file error, nonexistent case 404 |
| **Query** | SSE stream structure, all event types present, non-empty response, conversation_id returned, multi-turn conversations |
| **Conversations** | List for case, persistence after query, turn structure, turn count, user isolation, delete |
| **Summaries** | Endpoint reachable, response structure, nonexistent case 404, user isolation |
| **Error Format** | All error responses match the standard envelope |

### Test Execution Order

Tests run in numbered order, with state flowing between them via session-scoped fixtures:

```
test_01_auth.py           -> JWT rejection cases
test_02_health.py         -> MongoDB + Chroma connectivity
test_03_cases.py          -> Create case (stores case_id)
test_04_files.py          -> Upload PDF (stores file_id)
test_05_documents.py      -> Ingest file into case
test_06_query.py          -> Query case (stores conversation_id)
test_07_conversations.py  -> Read + manage conversations
test_08_summaries.py      -> Read generated summary
test_09_cleanup.py        -> Delete test case
test_10_error_format.py   -> Validate error envelopes
```

## Streamlit Testing UI

A Streamlit application is included for manually testing all API endpoints through a web interface.

### Running with Docker

```bash
docker compose --profile testing up -d
# Open http://localhost:8501
```

### Running Locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

### Features

The Streamlit app provides pages for each API area:
- **Health** -- Check API and database status
- **Cases** -- Create, list, update, and delete cases
- **Files** -- Upload files to cases
- **Documents** -- Trigger document ingestion
- **Query** -- Send queries and view SSE stream in real-time
- **Conversations** -- Browse conversation history
- **Summaries** -- View generated summaries

The sidebar lets you configure the API base URL, JWT secret, and user ID. JWT tokens are auto-generated.

## Project Structure

```
Judge-Assistant/
  config/                         # Centralized configuration
    __init__.py                   #   AppConfig singleton, get_llm() factory
    settings.yaml                 #   YAML defaults (committed)
    settings.local.yaml           #   Local overrides (gitignored)
    api.py                        #   FastAPI Pydantic Settings class
    ocr.py                        #   OCR pipeline constants
    supervisor.py                 #   Supervisor agent constants
    rag.py                        #   RAG constants + default state template

  api/                            # FastAPI application
    app.py                        #   Application factory, lifespan, error handlers
    config.py                     #   Shim -> config/api.py
    dependencies.py               #   Dependency injection (auth, DB, settings)
    errors.py                     #   Error code constants
    routers/                      #   Route handlers
      cases.py                    #     CRUD for cases
      files.py                    #     File upload
      documents.py                #     Document ingestion
      query.py                    #     Supervisor query (SSE)
      conversations.py            #     Conversation history
      summaries.py                #     Case summaries
      health.py                   #     Health check
    schemas/                      #   Pydantic request/response models
    services/                     #   Business logic layer
      auth.py                     #     JWT validation
      case_service.py             #     Case CRUD operations
      file_service.py             #     File storage
      document_service.py         #     Document processing
      query_service.py            #     Supervisor graph invocation
      conversation_service.py     #     Conversation management
      summary_service.py          #     Summary generation
    db/                           #   Database layer
      mongodb.py                  #     Motor async MongoDB client
      collections.py              #     Collection name constants
    tests/                        #   Integration test suite

  Supervisor/                     # Multi-agent supervisor (LangGraph)
    graph.py                      #   LangGraph state machine definition
    main.py                       #   Standalone entry point
    state.py                      #   SupervisorState TypedDict
    prompts.py                    #   LLM prompt templates
    config.py                     #   Shim -> config/supervisor.py
    agents/                       #   Agent adapters
      ocr_adapter.py              #     OCR pipeline adapter
      civil_law_rag_adapter.py    #     Civil law RAG adapter
      case_doc_rag_adapter.py     #     Case document RAG adapter
      case_reasoner_adapter.py    #     Legal reasoning adapter
      summarize_adapter.py        #     Summarization adapter
    nodes/                        #   Graph node functions
      classify_intent.py          #     Intent classification
      dispatch_agents.py          #     Agent dispatch
      merge_responses.py          #     Response merging
      validate_output.py          #     Output validation
      update_memory.py            #     Conversation memory
      classify_and_store_document.py  # Document classification + storage
      off_topic.py                #     Off-topic handling
      fallback.py                 #     Fallback responses
    services/
      file_ingestor.py            #   File type detection + ingestion

  OCR/                            # Arabic OCR pipeline
    engine.py                     #   Surya OCR engine wrapper
    preprocessor.py               #   Image preprocessing (deskew, denoise, etc.)
    postprocessor.py              #   Text post-processing (dictionary correction)
    ocr_pipeline.py               #   Full pipeline orchestration
    utils.py                      #   Utility functions
    schemas.py                    #   OCR data models
    config.py                     #   Shim -> config/ocr.py
    dictionaries/
      legal_arabic.txt            #   Arabic legal dictionary for correction

  RAG/                            # Retrieval-Augmented Generation
    Civil Law RAG/                #   Egyptian civil law RAG
      graph.py                    #     LangGraph RAG workflow
      nodes.py                    #     RAG node functions
      vectorstore.py              #     ChromaDB vector store
      indexer.py                  #     Document indexing
      splitter.py                 #     Legal text splitter
      prompts.py                  #     RAG prompt templates
      routers.py                  #     Graph routing logic
      config.py                   #     Shim -> config/rag.py
      docs/
        civil_law_clean.txt       #     Egyptian civil law text
    Case Doc RAG/                 #   Case document RAG
      rag_docs.py                 #     Document Q&A
      document_classifier.py      #     Document type classification

  Case Reasoner/                  #   Legal case reasoning
    case_reasoner.py              #     Reasoning agent

  Summerize/                      #   Document summarization pipeline
    main.py                       #     Pipeline entry point
    graph.py                      #     LangGraph summarization workflow
    schemas.py                    #     Data models
    node_0.py - node_5.py         #     Pipeline stages

  streamlit_app/                  #   Streamlit testing UI
    app.py                        #     Main app + sidebar config
    Dockerfile                    #     Container definition
    requirements.txt              #     Streamlit dependencies
    pages/                        #     One page per API area
    utils/
      api_client.py               #     HTTP client + JWT generation
      display.py                  #     Display helpers

  Dockerfile                      # API container definition
  docker-compose.yml              # Full stack orchestration
  requirements.txt                # Python dependencies
  Makefile                        # Convenience commands
  .env.example                    # Environment variable template
  .dockerignore                   # Docker build context exclusions
  CONFIG.md                       # Configuration system documentation
  docs/
    API_HANDOFF.md                # Express team integration guide
```

## Troubleshooting

### "Connection refused" when calling the API

Make sure the services are running:
```bash
docker compose ps
```
If the API container is restarting, check logs:
```bash
make logs
# or: docker compose logs api
```

### API starts but health check fails on Chroma

On first run, the ChromaDB directory may not exist. The API creates it automatically, but if you see errors, ensure the `chroma_data/` directory is writable:
```bash
mkdir -p chroma_data uploads
```

### Embedding model download is slow

The BGE-M3 embedding model (~400MB) is downloaded on first query. The Docker setup caches it in a named volume (`huggingface_cache`), so subsequent starts are fast. For local dev, models are cached in `~/.cache/huggingface/`.

### "GROQ_API_KEY not set" or LLM errors

Make sure your `.env` file has valid API keys:
```bash
cat .env | grep -E "GROQ|GOOGLE"
```
The high and medium tiers default to Groq; the low tier defaults to Google. You need at least the keys for the providers you're using.

### MongoDB authentication errors

The default Docker setup uses MongoDB without authentication. If you're connecting to an authenticated MongoDB instance, set the full URI:
```bash
JA_MONGODB_URI=mongodb://user:password@host:27017/dbname?authSource=admin
```

### Tests fail with "MongoDB not connected"

Make sure MongoDB is running and accessible. For Docker:
```bash
docker compose up -d mongo
```
For local dev, check that `mongod` is running on port 27017.

### OCR produces poor results

- Ensure input images are at least 150 DPI (configurable via `JA_OCR_PREPROCESSING_MIN_DPI`)
- The OCR pipeline is optimized for Arabic legal documents
- GPU acceleration can be enabled with `JA_OCR_USE_GPU=true` (requires CUDA)

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Run the test suite: `cd api && python -m pytest tests/ -v`
4. Submit a pull request

Follow conventional commit messages (`feat:`, `fix:`, `docs:`, etc.).
