# API Documentation

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
