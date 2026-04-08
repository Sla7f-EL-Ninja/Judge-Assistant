# Documentation Index

## Project Documentation

| Document | Description |
|---|---|
| [Glossary](GLOSSARY.md) | Bilingual term reference (Arabic/English) |
| [Architecture](ARCHITECTURE.md) | System design, supervisor graph, state management, LLM tiers |
| [Agents](AGENTS.md) | Specialist agents: OCR, Summarize, Civil Law RAG, Case Doc RAG, Reasoner |
| [Database](DATABASE.md) | MongoDB, Qdrant, Redis, MinIO, PostgreSQL schemas and configuration |
| [API Reference](API.md) | Endpoints, request/response schemas, SSE streaming, error codes |
| [Setup](SETUP.md) | Environment setup, configuration, common errors |
| [Testing](TESTING.md) | Test structure, running tests, writing tests |
| [Deployment](DEPLOYMENT.md) | Docker Compose, health checks, scaling, backups |
| [Troubleshooting](TROUBLESHOOTING.md) | Known issues and debugging guide |
| [Decisions](DECISIONS.md) | Architecture Decision Records (ADRs) |
| [Contributing](CONTRIBUTING.md) | Developer guide, adding agents/endpoints, PR checklist |
| [API Handoff](API_HANDOFF.md) | Express team integration guide |

---

## OpenAPI Spec Generation

To generate the OpenAPI spec files (`openapi.json` and `openapi.yaml`), run:

```bash
# From the project root (with dependencies installed)
python api/scripts/export_openapi.py
```

This will write:
- `docs/openapi.json`
- `docs/openapi.yaml` (requires `pyyaml`)

## API Handoff

See [API_HANDOFF.md](./API_HANDOFF.md) for the Express team integration guide.

## Interactive Docs

When the API is running, interactive documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
