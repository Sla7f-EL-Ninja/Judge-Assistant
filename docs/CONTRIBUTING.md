# Contributing Guide

## Getting Started

1. Clone the repository and set up your development environment following [SETUP.md](SETUP.md)
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system design
3. Review [GLOSSARY.md](GLOSSARY.md) for terminology

## Project Structure

```
Judge-Assistant/
  api/                    # FastAPI HTTP layer
    routers/              #   Route definitions (one file per resource)
    schemas/              #   Pydantic request/response models
    services/             #   Business logic
    db/                   #   Database client management
    tests/                #   API unit tests

  Supervisor/             # LangGraph multi-agent orchestrator
    graph.py              #   Graph construction and compilation
    state.py              #   SupervisorState TypedDict + Pydantic schemas
    prompts.py            #   All LLM prompts (Arabic)
    agents/               #   Agent adapter classes
    nodes/                #   Graph node implementations
    services/             #   Support services (file ingestion)

  RAG/                    # Retrieval-Augmented Generation pipelines
    Civil Law RAG/        #   Egyptian civil law article retrieval
    Case Doc RAG/         #   Case-specific document retrieval

  Summerize/              # Document summarization pipeline (7 nodes)
  Case Reasoner/          # Legal reasoning module
  OCR/                    # Arabic OCR pipeline (Surya)

  config/                 # Centralized configuration
    settings.yaml         #   All defaults
    __init__.py           #   Config loader, get_llm() factory
    api.py                #   FastAPI Settings class
    supervisor.py         #   Supervisor constants

  streamlit_app/          # Testing UI
  tests/                  # System-level tests
  docs/                   # Documentation (you are here)
```

## Adding a New Agent

### 1. Create the pipeline

Build your agent's core logic in its own directory (e.g., `MyAgent/`). It should accept a query and context, and return results.

### 2. Create an adapter

Add a new adapter class in `Supervisor/agents/`:

```python
# Supervisor/agents/my_agent_adapter.py
from Supervisor.agents.base import AgentAdapter, AgentResult

class MyAgentAdapter(AgentAdapter):
    def invoke(self, query: str, context: dict) -> AgentResult:
        # Call your pipeline
        result = my_pipeline(query, context)

        return AgentResult(
            response=result["answer"],
            sources=result.get("sources", []),
            raw_output=result,
        )
```

### 3. Register the adapter

Add your agent to the `ADAPTER_REGISTRY` in [`Supervisor/nodes/dispatch_agents.py`](../Supervisor/nodes/dispatch_agents.py:24):

```python
from Supervisor.agents.my_agent_adapter import MyAgentAdapter

ADAPTER_REGISTRY: Dict[str, type] = {
    # ... existing agents ...
    "my_agent": MyAgentAdapter,
}
```

### 4. Update the intent classifier

Add your agent to the valid agent names in [`config/settings.yaml`](../config/settings.yaml:109):

```yaml
supervisor:
  agent_names:
    - ocr
    - summarize
    - civil_law_rag
    - case_doc_rag
    - reason
    - my_agent  # Add here
```

Update the classification prompt in [`Supervisor/prompts.py`](../Supervisor/prompts.py) with a description of when your agent should be triggered.

### 5. Add tests

- Add unit tests in `api/tests/` if the agent has an API endpoint
- Add a routing test case in `tests/eval/routing_cases.json`
- Add behavioral tests in `tests/behavioral/`

---

## Adding a New API Endpoint

### 1. Define schemas

Create or update Pydantic models in `api/schemas/`:

```python
# api/schemas/my_resource.py
from pydantic import BaseModel, Field

class MyResourceCreate(BaseModel):
    name: str = Field(..., min_length=1)

class MyResourceResponse(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
```

### 2. Create the service

Add business logic in `api/services/`:

```python
# api/services/my_resource_service.py
async def create_resource(db, user_id: str, data: dict) -> dict:
    # Database operations
    ...
```

### 3. Create the router

Add a new router file in `api/routers/`:

```python
# api/routers/my_resource.py
from fastapi import APIRouter, Depends
from api.dependencies import get_current_user, get_db

router = APIRouter(prefix="/api/v1/my-resource", tags=["MyResource"])

@router.post("")
async def create(body: MyResourceCreate, user_id: str = Depends(get_current_user), db = Depends(get_db)):
    ...
```

### 4. Mount the router

Add the router to the app in [`api/app.py`](../api/app.py):

```python
from api.routers.my_resource import router as my_resource_router
app.include_router(my_resource_router)
```

### 5. Add tests

Add tests following the naming convention in `api/tests/`.

---

## Code Style and Conventions

### Python

- **Type hints**: Use type hints on all function signatures
- **Docstrings**: Use Google-style or NumPy-style docstrings
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Imports**: Group as stdlib, third-party, local; sorted alphabetically within groups
- **Error handling**: Use the error codes from [`api/errors.py`](../api/errors.py) for API errors

### Pydantic Models

- Use `Field(...)` for required fields with descriptions
- Use `Field(default_factory=...)` for mutable defaults
- Add `model_config` for serialization settings

### LangGraph Nodes

- Each node is a function that takes `SupervisorState` and returns `Dict[str, Any]`
- Return only the state keys that changed (partial update)
- Log the node name and key actions at INFO level
- Handle exceptions gracefully and return error information in state

### Prompts

- All user-facing prompts are in Arabic (in [`Supervisor/prompts.py`](../Supervisor/prompts.py))
- Add English comments above each prompt for developer orientation
- Use `.format()` string templates for variable interpolation

---

## Commit Convention

Use conventional commit messages:

```
feat: add new agent adapter for document comparison
fix: handle empty conversation history in query service
docs: update API reference with new endpoints
test: add behavioral tests for routing accuracy
refactor: extract common database helpers
chore: update dependencies
```

## Pull Request Checklist

Before submitting a PR:

- [ ] Code follows the project's style conventions
- [ ] All new functions have type hints and docstrings
- [ ] New API endpoints have corresponding schema definitions
- [ ] Tests pass locally (`python -m pytest tests/ api/tests/ -v`)
- [ ] New agents are registered in `ADAPTER_REGISTRY` and `config/settings.yaml`
- [ ] New endpoints are mounted in `api/app.py`
- [ ] Documentation is updated if behavior changes
- [ ] No hardcoded secrets or API keys
- [ ] Error handling follows the `ErrorEnvelope` pattern
