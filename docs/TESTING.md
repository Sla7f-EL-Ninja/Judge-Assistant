# Testing Guide

## Test Suite Structure

The project has two test directories with different purposes:

```
tests/                          # System-level tests (behavioral, integration, performance, eval)
  conftest.py                   # Root fixtures: db connections, JWT helper, pipeline fixture
  pytest.ini                    # Config: asyncio_mode=auto, custom markers
  behavioral/                   # Answer consistency, boundary cases, RAG quality, routing
  integration/                  # API contracts, database state, ingestion pipeline, Redis cache
  performance/                  # Locust load tests, cache speedup, memory leak detection
  eval/                         # Golden dataset evaluation, LLM judge, routing cases

api/tests/                      # API-layer unit tests
  conftest.py                   # API-specific fixtures: httpx AsyncClient, auth headers
  pytest.ini                    # Config for API tests
  test_01_auth.py ... test_10_error_format.py  # Ordered test sequence
  requirements-test.txt         # Test-specific dependencies
```

## Running Tests

### Prerequisites

- Python 3.11+ with dependencies installed (`pip install -r requirements.txt`)
- Infrastructure services running (MongoDB, Qdrant at minimum)
- API keys set (`GROQ_API_KEY` in environment)

### API Unit Tests

```bash
# Run all API tests
cd api && python -m pytest tests/ -v

# Run a specific test file
cd api && python -m pytest tests/test_02_health.py -v

# Run with coverage
cd api && python -m pytest tests/ --cov=api --cov-report=term-missing
```

### System Tests

```bash
# Run all system tests
python -m pytest tests/ -v

# Run by category
python -m pytest tests/behavioral/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/performance/ -v

# Run evaluation tests
python -m pytest tests/eval/ -v
```

### Full Suite

```bash
# Run everything
python -m pytest tests/ api/tests/ -v
```

## Test Categories

### API Unit Tests (`api/tests/`)

Ordered test sequence that exercises the full API surface:

| File | Scope |
|---|---|
| `test_01_auth.py` | JWT validation, missing/invalid tokens |
| `test_02_health.py` | Health endpoint, dependency status |
| `test_03_cases.py` | Case CRUD operations |
| `test_04_files.py` | File upload, MIME type validation |
| `test_05_documents.py` | Document ingestion pipeline |
| `test_06_query.py` | Supervisor query, SSE streaming |
| `test_07_conversations.py` | Conversation history management |
| `test_08_summaries.py` | Summary retrieval |
| `test_09_cleanup.py` | Cleanup / teardown |
| `test_10_error_format.py` | Error envelope structure validation |

### Behavioral Tests (`tests/behavioral/`)

Test the system's behavior at a higher level:

| File | Purpose |
|---|---|
| `test_answer_consistency.py` | Same question should produce consistent answers |
| `test_boundary_cases.py` | Edge cases, empty inputs, very long queries |
| `test_rag_quality.py` | RAG retrieval accuracy and relevance |
| `test_routing_accuracy.py` | Intent classification accuracy against known test cases |

### Integration Tests (`tests/integration/`)

Test interactions between system components:

| File | Purpose |
|---|---|
| `test_api_contracts.py` | API request/response schema compliance |
| `test_database_state.py` | Database consistency after operations |
| `test_ingestion_pipeline.py` | End-to-end document ingestion |
| `test_redis_cache.py` | Cache behavior, TTL, invalidation |

### Performance Tests (`tests/performance/`)

| File | Purpose |
|---|---|
| `locustfile.py` | Load testing with Locust |
| `test_cache_speedup.py` | Verify caching improves response times |
| `test_memory_leak.py` | Memory usage monitoring over time |

### Evaluation Framework (`tests/eval/`)

| File | Purpose |
|---|---|
| `golden_dataset.json` | Known query/answer pairs for regression testing |
| `routing_cases.json` | Intent classification test cases |
| `llm_judge.py` | LLM-based evaluation of response quality |

## Test Configuration

### `tests/pytest.ini`

```ini
[pytest]
asyncio_mode = auto
```

### `api/pytest.ini`

```ini
[pytest]
asyncio_mode = auto
```

### Key Fixtures (from `tests/conftest.py`)

| Fixture | Scope | Description |
|---|---|---|
| `app_client` | session | httpx `AsyncClient` with ASGITransport for testing FastAPI |
| `auth_headers` | session | JWT authorization headers for authenticated requests |
| `pipeline` | session | Compiled supervisor graph for direct invocation |
| `test_document_id` | session | Pre-created document for tests that need existing data |

### JWT Helper

Both conftest files include a JWT helper that generates test tokens:

```python
from jose import jwt

token = jwt.encode(
    {"user_id": "test-user", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
    settings.jwt_secret,
    algorithm="HS256",
)
headers = {"Authorization": f"Bearer {token}"}
```

## Writing New Tests

### Adding an API Test

1. Create a new file in `api/tests/` following the naming convention `test_NN_description.py`
2. Use the `app_client` fixture for HTTP requests
3. Use the `auth_headers` fixture for authenticated endpoints
4. Follow the existing pattern of asserting both status codes and response shapes

### Adding a Behavioral Test

1. Create a new file in `tests/behavioral/`
2. Use the `pipeline` fixture for direct supervisor graph invocation
3. Define test cases as parametrized inputs with expected outputs
4. Use `tests/eval/golden_dataset.json` format for structured test data

### Skip Guards

Tests that require infrastructure services use skip guards:

```python
@pytest.mark.skipif(
    not os.getenv("MONGODB_AVAILABLE"),
    reason="MongoDB not available",
)
def test_something_with_mongo():
    ...
```

## CI/CD

Tests are designed to run in CI with these considerations:

1. Infrastructure services must be available (use Docker Compose or service containers)
2. API keys must be set as CI secrets
3. The `PYTHONPATH` must include the project root
4. Embedding model download is cached between runs (use CI caching for `~/.cache/huggingface`)
