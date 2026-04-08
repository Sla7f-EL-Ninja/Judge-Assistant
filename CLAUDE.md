# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## Project Overview

**Judge Assistant (Hakim)** is an AI-powered legal assistant for Egyptian judicial workflows. It exposes a FastAPI REST API backed by a LangGraph multi-agent supervisor that orchestrates OCR, civil law RAG, case document RAG, legal reasoning, and summarization pipelines.

### Local Development
```bash
pip install -r requirements.txt
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

---

## Architecture

### Request Lifecycle

```
Client → POST /api/v1/query (JWT required)
       → query_service.py invokes Supervisor LangGraph
       → SSE stream: progress / result / error / done events
```

### Supervisor Graph (`Supervisor/graph.py`)

8-node LangGraph state machine (`SupervisorState` in `Supervisor/state.py`):

1. **classify_intent** — medium-tier LLM → one of `ocr | summarize | civil_law_rag | case_doc_rag | reason | multi | off_topic`
2. **dispatch_agents** — runs selected agents sequentially; each sees prior agents' output
3. **classify_and_store_document** — if OCR ran or files were uploaded, classify & embed into Qdrant
4. **merge_responses** — high-tier LLM synthesizes multi-agent output (or passes single result)
5. **validate_output** — low-tier LLM checks for hallucination / relevance / completeness
6. **update_memory** — appends conversation turn to MongoDB
7. **off_topic_response** — direct reply for off-topic intents
8. **fallback_response** — used when retries are exhausted

Retry loop: validation failure → re-dispatch with `validation_feedback` appended, up to `max_retries=3`.

### Agent Adapters (`Supervisor/agents/`)

| Agent | Module | Trigger |
|---|---|---|
| OCR | `OCR/ocr_pipeline.py` | `ocr` |
| Summarizer | `Summerize/main.py` | `summarize` |
| Civil Law RAG | `RAG/Civil Law RAG/graph.py` | `civil_law_rag` |
| Case Doc RAG | `RAG/case_doc_rag/graph.py` | `case_doc_rag` |
| Case Reasoner | `Case Reasoner/case_reasoner.py` | `reason` |

`ADAPTER_REGISTRY` in `Supervisor/nodes/dispatch_agents.py` maps intent names to adapter classes.

### LLM Tier System (`config/__init__.py: get_llm()`)

| Tier | Default Model | Used For |
|---|---|---|
| `high` | gemini-2.5-flash | Reasoning, synthesis, summarization |
| `medium` | gemini-2.5-flash | Intent classification, extraction |
| `low` | gemini-2.5-flash-lite | Output validation |

---

## Data Stores

| Store | Purpose |
|---|---|
| MongoDB | Cases, files, conversations, summaries, documents |
| Qdrant | Vectors — `judicial_docs` (civil law, 1024-dim COSINE) and `case_docs` (filtered by `case_id`) |
| Redis | Optional caching / rate limiting |
| MinIO | S3-compatible file storage (falls back to local disk) |

---

## Configuration

### Precedence (lowest → highest)
1. `config/settings.yaml` — committed defaults (source of truth)
2. `config/settings.local.yaml` — gitignored local overrides, deep-merged
3. `JA_*` environment variables — e.g. `JA_LLM_HIGH_MODEL`, `JA_MONGODB_URI`

API keys (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, etc.) and `MONGO_URI`, `JWT_SECRET` live in `.env` (gitignored).

---

## API Layer (`api/`)

- **Entry point**: `api/app.py` — `create_app()` factory, lifespan connects all DBs
- **Routers**: `health`, `cases`, `files`, `documents`, `query` (SSE), `conversations`, `summaries`
- **Services**: `api/services/` — business logic; `query_service.py` invokes the supervisor graph
- **DB clients**: `api/db/` — Motor (MongoDB), Qdrant, Redis, MinIO, PostgreSQL

All endpoints except `/api/v1/health` require a JWT Bearer token with a `user_id` claim (HS256).

---

## Documentation

Detailed docs live in `docs/`:
- `ARCHITECTURE.md` — full graph diagrams and ADRs pointer
- `AGENTS.md` — agent input/output schemas
- `API.md` — all endpoints with request/response schemas
- `SETUP.md` — environment variables reference
- `TESTING.md` — test writing guide and CI/CD
- `TROUBLESHOOTING.md` — 10 known issues with fixes
- `DECISIONS.md` — 7 Architecture Decision Records

---

## How to Work

### Session Start
Before any non-trivial task:
1. Read `tasks/lessons.md` for patterns from past corrections in this project
2. Identify which agents, nodes, or data stores are touched by this task
3. Enter plan mode if the task is 3+ steps or touches the supervisor graph

### Plan Mode
- Use plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- Write a spec in `tasks/todo.md` with checkable items before touching code
- Check in before starting implementation — plan first, build second
- If something goes sideways mid-task, STOP and re-plan immediately

### Subagent Strategy
- Use subagents to keep the main context window clean
- Offload research, file exploration, and parallel analysis to subagents
- One focused task per subagent — no multipurpose subagents

### Verification Before Done
- Never mark a task complete without proving it works
- Run tests, check logs, demonstrate correctness
- Ask: "Would a staff engineer approve this PR?"
- Diff behavior between main and your changes when relevant

### Autonomous Bug Fixing
- When given a bug report: fix it — no hand-holding needed
- Point at logs, errors, failing tests — then resolve them
- Go fix failing tests without being told how
- Always run pytest via ctx_execute, never raw Bash

### Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: implement the elegant solution from scratch
- Skip this for simple, obvious fixes — don't over-engineer

---

## Task Management

1. **Plan First** — write plan to `tasks/todo.md` with checkable items
2. **Verify Plan** — check in before starting implementation
3. **Track Progress** — mark items complete as you go
4. **Explain Changes** — high-level summary at each step
5. **Document Results** — add a review section to `tasks/todo.md` when done
6. **Capture Lessons** — update `tasks/lessons.md` after any user correction

---

## Communication Style

Responses must be compressed and direct. No exceptions.

**Rules:**
- No filler words. No "the", "is", "am", "are" where removable
- No narration. Never say "I will now...", "Let me...", "Sure, I can..."
- No preamble. Answer first, explain after if needed
- Short sentences. 3–6 words preferred
- Run tools first, show result, then stop
- No restating the question back
- No closing remarks ("Let me know if...", "Hope that helps")

**Bad:** "The issue is that the validate_output node is returning an error because the LLM response is malformed."
**Good:** "validate_output fails — malformed LLM response."

---

## Core Principles

- **Simplicity First** — make every change as simple as possible, minimal code impact
- **No Laziness** — find root causes, no temporary fixes, senior developer standards
- **Minimal Impact** — only touch what's necessary, no side-effect bugs from unrelated changes
- **Self-Correction** — after any correction, write the pattern to `tasks/lessons.md` immediately