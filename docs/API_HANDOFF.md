# Judge Assistant API -- Express Team Handoff

This document covers everything the Express/frontend team needs to integrate with the Judge Assistant API.

## 0. Getting Started with Docker

The fastest way to run the full stack locally:

1. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   # Edit .env -- set GROQ_API_KEY, GOOGLE_API_KEY, and JA_API_JWT_SECRET
   ```

2. Start MongoDB + API:
   ```bash
   docker compose up -d
   ```

3. The API is now available at `http://localhost:8000/api/v1`

4. Interactive docs at `http://localhost:8000/docs`

5. (Optional) Start the Streamlit testing UI as well:
   ```bash
   docker compose --profile testing up -d
   ```
   Streamlit will be at `http://localhost:8501`.

6. Useful commands (see `Makefile`):
   ```bash
   make logs          # Watch API logs
   make shell         # Bash into the API container
   make mongo-shell   # Open MongoDB shell
   make down          # Stop everything
   make down-clean    # Stop and wipe all data volumes
   ```

## 1. Base URL and Versioning

All routes are prefixed with `/api/v1/`. There is currently no v2 planned.

| Environment | Base URL |
|-------------|----------|
| Local dev | `http://localhost:8000/api/v1` |
| Staging | TBD |
| Production | TBD |

## 2. Authentication

The API uses **JWT Bearer tokens** for authentication.

**Token format:**
- Algorithm: HS256
- Required claim: `user_id` (string) -- identifies the authenticated user
- Required claim: `exp` -- standard JWT expiration timestamp
- Shared secret: configured via `JWT_SECRET` environment variable

**Usage:**
```
Authorization: Bearer <jwt_token>
```

All endpoints except `GET /api/v1/health` require a valid JWT.

**Error responses for auth failures:**
- Missing `Authorization` header: `422 VALIDATION_ERROR`
- Malformed header (not `Bearer ...`): `401 UNAUTHORIZED`
- Expired / tampered / invalid token: `401 UNAUTHORIZED`
- Token missing `user_id` claim: `401 UNAUTHORIZED`

## 3. Standard Error Envelope

Every error response follows this shape:

```json
{
  "error": {
    "code": "CASE_NOT_FOUND",
    "detail": "Case not found",
    "status": 404
  }
}
```

### Error Codes

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `UNAUTHORIZED` | 401 | Missing, expired, or invalid JWT token |
| `VALIDATION_ERROR` | 422 | Request body/query validation failed |
| `CASE_NOT_FOUND` | 404 | Case does not exist or belongs to another user |
| `CONVERSATION_NOT_FOUND` | 404 | Conversation does not exist or belongs to another user |
| `FILE_NOT_FOUND` | 404 | Uploaded file not found |
| `SUMMARY_NOT_FOUND` | 404 | No summary generated for this case yet |
| `INVALID_MIME_TYPE` | 400 | Uploaded file MIME type is not in the allowed list |
| `FILE_TOO_LARGE` | 400 | Uploaded file exceeds the 20 MB limit |
| `NO_FIELDS_TO_UPDATE` | 400 | PATCH request body contains no updatable fields |
| `INTERNAL_ERROR` | 500 | Unexpected server error (details logged server-side only) |

## 4. Endpoint Reference

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| GET | `/api/v1/health` | Service health check | No |
| POST | `/api/v1/cases` | Create a new case | Yes |
| GET | `/api/v1/cases` | List cases (paginated) | Yes |
| GET | `/api/v1/cases/{case_id}` | Get case details | Yes |
| PATCH | `/api/v1/cases/{case_id}` | Update case | Yes |
| DELETE | `/api/v1/cases/{case_id}` | Soft-delete case | Yes |
| POST | `/api/v1/files/upload` | Upload a file | Yes |
| POST | `/api/v1/cases/{case_id}/documents` | Ingest documents into case | Yes |
| POST | `/api/v1/query` | Run supervisor query (SSE) | Yes |
| GET | `/api/v1/cases/{case_id}/conversations` | List conversations for case | Yes |
| GET | `/api/v1/conversations/{conversation_id}` | Get conversation history | Yes |
| DELETE | `/api/v1/conversations/{conversation_id}` | Delete conversation | Yes |
| GET | `/api/v1/cases/{case_id}/summary` | Get case summary | Yes |

## 5. SSE Stream Format for `/api/v1/query`

The query endpoint returns a `text/event-stream` response with these event types:

### Event: `progress`
Emitted as the supervisor graph processes each node.

```
event: progress
data: {"step": "starting", "status": "running"}

event: progress
data: {"step": "classify_intent", "status": "done"}
```

### Event: `result`
Emitted once with the final answer.

```
event: result
data: {"final_response": "...", "sources": [...], "intent": "...", "agents_used": [...], "conversation_id": "conv_xxx"}
```

### Event: `error`
Emitted if the supervisor graph fails. The detail is sanitized (no stack traces).

```
event: error
data: {"detail": "An internal error occurred while processing the query"}
```

If a `conversation_id` was explicitly provided but not found:

```
event: error
data: {"detail": "Conversation not found", "code": "CONVERSATION_NOT_FOUND"}
```

### Event: `done`
Always the last event emitted, even after errors.

```
event: done
data: {}
```

## 6. Pagination Convention

List endpoints accept these query parameters:

| Parameter | Type | Default | Constraints |
|-----------|------|---------|-------------|
| `skip` | int | 0 | >= 0 |
| `limit` | int | 20 | 1-100 |

Response shape for paginated lists:

```json
{
  "cases": [...],
  "total": 42
}
```

The `total` field is the total count of matching records (ignoring skip/limit).

## 7. File Upload Constraints

| Constraint | Value |
|------------|-------|
| Max file size | 20 MB (20,971,520 bytes) |
| Allowed MIME types | `application/pdf`, `image/png`, `image/jpeg`, `image/tiff`, `image/bmp`, `image/webp` |
| Upload field name | `file` (multipart form) |

## 8. Quirks and Gotchas

1. **Case deletion is soft-delete, conversation deletion is hard-delete.** Deleting a case sets `status="deleted"` and the case is excluded from list results but data is retained. Deleting a conversation permanently removes it.

2. **`_id` field in responses, not `id`.** MongoDB convention. The field is aliased from `_id` in the Pydantic models, but the JSON key is `_id`.

3. **`conversation_count` on cases uses N+1 queries.** For a list of 20 cases, 20 extra DB queries are fired. Not a problem at current scale but worth noting for performance-sensitive UIs.

4. **Query endpoint auto-creates conversations.** If `conversation_id` is null/omitted, a new conversation is created automatically. If `conversation_id` is explicitly provided but not found, an error is returned.

5. **Document ingestion is slow.** Ingesting a document (OCR + embedding + classification) typically takes 10-30 seconds per file. Set appropriate timeouts on the client side.

6. **SSE requires no buffering.** The API sets `X-Accel-Buffering: no` and `Cache-Control: no-cache`. If running behind nginx or a CDN, ensure SSE buffering is disabled.

7. **Metadata field on cases is freeform JSON.** The `metadata` field on cases accepts any valid JSON object. There is no schema validation on the contents.
