# Integration Test Suite

Real end-to-end tests against live MongoDB, real Chroma, and the full
Supervisor LangGraph. No mocks, no fakes.

--- 

## Setup

1. Install test dependencies:
   ```bash
   pip install -r tests/requirements-test.txt
   ```

2. Copy the environment template and fill in your real values:
   ```bash
   cp tests/.env.test .env
   ```

   The key values to set:
   | Variable | What to put |
   |---|---|
   | `MONGO_URI` | Your running MongoDB URI |
   | `MONGO_DB` | Use a separate DB name like `Rag_test` |
   | `JWT_SECRET` | Must match your Express backend's secret |
   | `CHROMA_PERSIST_DIR` | Path to your Chroma data dir |
   | `TEST_PDF_PATH` | Absolute path to a real Arabic/legal PDF |
   | `TEST_QUERY` | A real Arabic legal question |

3. Make sure MongoDB is running and Chroma persist dir exists.

---

## Running

Run the full suite in order:
```bash
pytest tests/ -v
```

Run only the fast tests (skip ingestion + query):
```bash
pytest tests/ -v -k "not document and not query and not conversation and not summary"
```

Run a single file:
```bash
pytest tests/test_cases.py -v
```

Run with detailed output on failures:
```bash
pytest tests/ -v --tb=short
```

---

## Test execution order

pytest runs files alphabetically, which gives this natural sequence:

```
test_auth.py          ← JWT rejection cases (no DB needed)
test_cases.py         ← Create case → stores case_id in TestState
test_cleanup.py       ← Deletes test case (runs last alphabetically)
test_conversations.py ← Reads + deletes conversation from query tests
test_documents.py     ← Ingests uploaded file into case
test_files.py         ← Uploads PDF → stores file_id in TestState
test_health.py        ← MongoDB + Chroma connectivity
test_query.py         ← Fires real query → stores conversation_id
test_summaries.py     ← Reads generated summary
```

Because `TestState` is a session-scoped fixture, IDs created early
(case_id, file_id, conversation_id) flow through to later tests
automatically.

---

## What gets tested

| Area | Tests |
|---|---|
| Auth | Missing header, bad format, tampered token, expired token, missing user_id |
| Health | MongoDB connected, Chroma connected, overall healthy |
| Cases | Create, list, paginate, get, update title, update status, invalid status, empty patch, wrong-user isolation, delete |
| Files | PDF upload, PNG upload, wrong MIME rejected, oversized rejected |
| Documents | Ingest real PDF, case documents updated after ingest, nonexistent file error, nonexistent case 404 |
| Query | SSE stream structure, all required events present, non-empty response, conversation_id returned, second turn reuses conversation, empty query rejected |
| Conversations | List for case, persisted after query, turn structure, turn count, wrong-user isolation, delete, gone after delete |
| Summaries | Endpoint reachable, structure when present, nonexistent case 404, wrong-user isolation |
