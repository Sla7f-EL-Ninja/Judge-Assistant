# API Reference

Base URL: `/api/v1`

All endpoints except `/health` require a JWT Bearer token in the `Authorization` header. The API uses structured error responses with machine-readable error codes.

---

## Authentication

Tokens are issued by the Express backend and validated by the API using a shared secret.

| Property | Value |
|---|---|
| Type | JWT Bearer |
| Algorithm | HS256 |
| Required claim | `user_id` |
| Secret | Configurable via `api.jwt_secret` in `config/settings.yaml` or `JWT_SECRET` env var |

### Header Format

```
Authorization: Bearer <JWT_TOKEN>
```

### Example: Generate a test token (Python)

```python
from jose import jwt

token = jwt.encode(
    {"user_id": "test-user-123"},
    "your-jwt-secret",
    algorithm="HS256",
)
```

---

## Error Format

All errors follow the `ErrorEnvelope` schema defined in [`api/schemas/common.py`](../api/schemas/common.py):

```json
{
  "error": {
    "code": "CASE_NOT_FOUND",
    "detail": "Case with id 'abc123' not found",
    "status": 404
  }
}
```

### Error Codes

Defined in [`api/errors.py`](../api/errors.py):

| Code | HTTP Status | Description |
|---|---|---|
| `UNAUTHORIZED` | 401 | Missing or invalid JWT token |
| `VALIDATION_ERROR` | 422 | Request body or query parameter validation failed |
| `CASE_NOT_FOUND` | 404 | Case ID does not exist or not owned by user |
| `CONVERSATION_NOT_FOUND` | 404 | Conversation ID does not exist |
| `FILE_NOT_FOUND` | 404 | File ID does not exist |
| `SUMMARY_NOT_FOUND` | 404 | No summary exists for the given case |
| `INVALID_MIME_TYPE` | 400 | Uploaded file has an unsupported MIME type |
| `FILE_TOO_LARGE` | 400 | Uploaded file exceeds the size limit (default 20 MB) |
| `NO_FIELDS_TO_UPDATE` | 400 | PATCH request body has no fields to update |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Endpoints

### Health

#### `GET /api/v1/health`

Service health and dependency connectivity check. **No authentication required.**

**Response** `200 OK`:
```json
{
  "status": "healthy",
  "mongodb": "connected",
  "qdrant": "connected",
  "redis": "connected",
  "minio": "connected",
  "postgresql": "connected"
}
```

---

### Cases

#### `POST /api/v1/cases`

Create a new case for the authenticated user.

**Request Body** (`CaseCreate`):
```json
{
  "title": "قضية مدنية رقم 123",
  "description": "نزاع عقاري بين المدعي والمدعى عليه",
  "metadata": {}
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `title` | `string` | Yes (min 1 char) | Case title |
| `description` | `string` | No | Case description |
| `metadata` | `object` | No | Arbitrary metadata |

**Response** `201 Created` (`CaseResponse`):
```json
{
  "_id": "case_abc123",
  "user_id": "user-123",
  "title": "قضية مدنية رقم 123",
  "description": "نزاع عقاري بين المدعي والمدعى عليه",
  "status": "active",
  "metadata": {},
  "documents": [],
  "conversation_count": 0,
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:30:00Z"
}
```

#### `GET /api/v1/cases`

List cases for the authenticated user with pagination.

**Query Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skip` | `int` | 0 | Number of records to skip |
| `limit` | `int` | 20 | Max records to return (1-100) |
| `status` | `string` | -- | Filter by status (`active`, `archived`, `closed`) |

**Response** `200 OK` (`CaseListResponse`):
```json
{
  "cases": [ /* CaseResponse objects */ ],
  "total": 42
}
```

#### `GET /api/v1/cases/{case_id}`

Get a single case by ID.

**Response** `200 OK`: `CaseResponse`
**Error** `404`: `CASE_NOT_FOUND`

#### `PATCH /api/v1/cases/{case_id}`

Update a case.

**Request Body** (`CaseUpdate`):
```json
{
  "title": "Updated title",
  "description": "Updated description",
  "status": "archived",
  "metadata": {"key": "value"}
}
```

All fields are optional. At least one must be provided.

**Response** `200 OK`: `CaseResponse`
**Errors**: `404 CASE_NOT_FOUND`, `400 NO_FIELDS_TO_UPDATE`

#### `DELETE /api/v1/cases/{case_id}`

Delete a case and all associated data.

**Response** `200 OK`: `{"message": "Case deleted"}`
**Error** `404`: `CASE_NOT_FOUND`

---

### Files

#### `POST /api/v1/files/upload`

Upload a file. The file is stored in MinIO (or local disk as fallback) and a file record is created in MongoDB.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | Yes | The file to upload |

**Allowed MIME types**: `application/pdf`, `image/png`, `image/jpeg`, `image/tiff`, `image/bmp`, `image/webp`
**Max size**: 20 MB (configurable via `api.max_upload_bytes`)

**Response** `201 Created`:
```json
{
  "file_id": "file_xyz789",
  "filename": "document.pdf",
  "content_type": "application/pdf",
  "size": 1048576
}
```

**Errors**: `400 INVALID_MIME_TYPE`, `400 FILE_TOO_LARGE`

---

### Documents

#### `POST /api/v1/cases/{case_id}/documents`

Ingest a file into a case. Runs the OCR pipeline (if needed), classifies the document, chunks and embeds the text, and stores vectors in Qdrant.

**Request Body**:
```json
{
  "file_id": "file_xyz789"
}
```

**Response** `201 Created`:
```json
{
  "document_id": "doc_abc123",
  "case_id": "case_abc123",
  "file_id": "file_xyz789",
  "doc_type": "صحيفة دعوى",
  "status": "processed"
}
```

---

### Query (SSE Streaming)

#### `POST /api/v1/query`

Submit a judge's question to the multi-agent supervisor graph. Returns a Server-Sent Events stream.

**Request Body** (`QueryRequest`):
```json
{
  "query": "ما هي المادة القانونية المنطبقة على هذه القضية؟",
  "case_id": "case_abc123",
  "conversation_id": null
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | `string` | Yes (min 1 char) | The judge's question |
| `case_id` | `string` | No | Active case identifier |
| `conversation_id` | `string` | No | Existing conversation to continue; null creates a new one |

**Response**: `200 OK` with `Content-Type: text/event-stream`

### SSE Event Types

The stream emits events in this format:

```
event: <type>
data: <json_payload>

```

#### `progress` Event

Emitted as each graph node completes.

```
event: progress
data: {"step": "classify_intent", "status": "done", "detail": {"intent": "civil_law_rag"}}

```

#### `result` Event

The final response from the supervisor.

```
event: result
data: {"final_response": "...", "sources": [...], "intent": "civil_law_rag", "agents_used": ["civil_law_rag"], "conversation_id": "conv_123"}

```

#### `error` Event

Emitted on failures. Error details are sanitized; full details are logged server-side.

```
event: error
data: {"detail": "An internal error occurred while processing the query", "code": "INTERNAL_ERROR"}

```

#### `done` Event

Always the last event in the stream. Signals the client to close the connection.

```
event: done
data: {}

```

### JavaScript Client Example

```javascript
const response = await fetch('/api/v1/query', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'ما هي المادة القانونية المنطبقة؟',
    case_id: 'case_abc123',
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split('\n');
  buffer = lines.pop(); // Keep incomplete line in buffer

  let eventType = '';
  for (const line of lines) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7);
    } else if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      switch (eventType) {
        case 'progress':
          console.log('Progress:', data.step, data.status);
          break;
        case 'result':
          console.log('Final response:', data.final_response);
          break;
        case 'error':
          console.error('Error:', data.detail);
          break;
        case 'done':
          console.log('Stream complete');
          break;
      }
    }
  }
}
```

### curl Example

```bash
curl -N -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هي المادة 148 من القانون المدني؟", "case_id": "case_123"}'
```

---

### Conversations

#### `GET /api/v1/cases/{case_id}/conversations`

List conversations for a case.

**Query Parameters**: `skip` (default 0), `limit` (default 20)

**Response** `200 OK`:
```json
{
  "conversations": [
    {
      "_id": "conv_123",
      "case_id": "case_abc123",
      "turns": [...],
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "total": 5
}
```

#### `GET /api/v1/conversations/{conversation_id}`

Get a single conversation with all turns.

**Response** `200 OK`: Full conversation object
**Error** `404`: `CONVERSATION_NOT_FOUND`

#### `DELETE /api/v1/conversations/{conversation_id}`

Delete a conversation.

**Response** `200 OK`: `{"message": "Conversation deleted"}`
**Error** `404`: `CONVERSATION_NOT_FOUND`

---

### Summaries

#### `GET /api/v1/cases/{case_id}/summary`

Get the auto-generated summary for a case. Summaries are produced by the Summarization agent when it runs during a query.

**Response** `200 OK`: Summary object
**Error** `404`: `SUMMARY_NOT_FOUND`

---

## Rate Limiting

When Redis is available, requests are rate-limited per user using a sliding window counter:

| Setting | Default |
|---|---|
| Max requests per window | 100 |
| Window duration | 60 seconds |
| Redis key pattern | `rate_limit:{user_id}` |

When the limit is exceeded, the API returns `429 Too Many Requests`. If Redis is unavailable, rate limiting is disabled (fail-open).

---

## Pagination

List endpoints support pagination via query parameters:

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `skip` | `int` | 0 | >= 0 | Records to skip |
| `limit` | `int` | 20 | 1-100 | Max records to return |
