# Chapter 3: System Design and Implementation

## 3.1 System Overview

Hakim is a multi-agent AI system designed to assist Egyptian court judges with legal document analysis, case reasoning, and jurisprudential retrieval. The system integrates a FastAPI REST layer with a LangGraph-based orchestration graph that coordinates a constellation of specialized AI agents. These agents collectively handle optical character recognition (OCR), civil law retrieval-augmented generation (RAG), case document RAG, conversational legal reasoning, and document summarization — all operating over Arabic legal texts.

The architecture follows a supervisor-agent paradigm in which a central orchestrator, the Supervisor Graph, classifies incoming judge queries, dispatches relevant agents in a dependency-aware order, validates their outputs against factual and relevance criteria, and maintains multi-tier conversational memory. The system is stateful: each conversation turn is checkpointed via MongoDB, and long-term semantic facts, episodic episodes, and judge behavioral preferences are persisted in a dedicated memory store.

> **[FIGURE NEEDED: End-to-end system architecture diagram showing the FastAPI layer, Supervisor Graph, specialized agents (OCR, Civil Law RAG, Case Doc RAG, Chat Reasoner, Summarizer), and data stores (MongoDB, Qdrant, Redis, MinIO)]**

The primary data stores are:

- **MongoDB** — cases, files, conversations, document storage, checkpoints (`supervisor_checkpoints`), and long-term memory (`supervisor_memory_store`), all housed in the `TESTING` database.
- **Qdrant** — dual vector collections: `judicial_docs` for the Egyptian Civil Code corpus (1024-dimensional COSINE similarity) and `case_docs` for per-case document vectors, accessed preferably via gRPC on port 6334.
- **Redis** — response-level semantic caching in the query service, keyed by SHA-256 of the query and case identifier.
- **MinIO** — S3-compatible binary file storage with local-disk fallback for uploaded case documents.

All endpoints except `/api/v1/health` require a JWT Bearer token (HS256) carrying a `user_id` claim. The system exposes ten router groups: `query` (SSE streaming), `conversations`, `cases`, `documents`, `files`, `case_reasoning`, `legal_search`, `reports`, `summaries`, and `health`.

---

## 3.2 Dataset Description

### 3.2.1 Civil Law Corpus

The civil law knowledge base comprises articles from the Egyptian Civil Code indexed into Qdrant under the collection name `judicial_docs`. Each document is encoded as a 1024-dimensional dense vector produced by the **BAAI/bge-m3** multilingual embedding model, served locally via a Text Embeddings Inference (TEI) server at `http://localhost:8080`. The collection uses COSINE distance for similarity measurement.

Metadata stored alongside each vector encodes the hierarchical structure of Egyptian civil law: book, part, chapter, section, and article number. These metadata fields are exploited by the Civil Law RAG agent's `scope_classifier_node` to restrict retrieval to the legally relevant portion of the code — for example, constraining a query about property easements to Book Two (Real Rights) rather than scanning the entire corpus.

### 3.2.2 Case Document Corpus

Case-specific documents uploaded by court users are processed through the ingestion pipeline and indexed into the `case_docs` Qdrant collection, filtered by `case_id`. Supported document types span the full range of Egyptian judicial proceedings and are defined in a YAML taxonomy (`config/document_taxonomy.yaml`) that includes, among others: صحيفة دعوى (claim petition), مذكرة دفاع (defense memorandum), تقرير خبير (expert report), مذكرة رد (reply memorandum), and حكم ابتدائي (first-instance judgment).

### 3.2.3 Document Schema and Ingestion

Each ingested document is stored in MongoDB's `Document Storage` collection with fields that track: the originating `case_id`, `file_id` references (supporting multi-file document groups), `doc_type` (from taxonomy), OCR-corrected full text, per-page confidence scores, and classification metadata. The `file_ids` field accommodates document groups — a logical multi-file document such as an expert report spanning several PDFs — introduced by the `process_document_group` orchestrator.

> **[FIGURE NEEDED: Entity-relationship diagram of MongoDB collections: cases, files, documents, conversations, summaries, supervisor_checkpoints, supervisor_memory_store]**

---

## 3.3 The Proposed System Pipeline

A judge's interaction follows a fixed lifecycle from query submission to final response delivery.

1. **HTTP Request** — The judge submits a natural-language query (in Arabic) and optional file uploads to `POST /api/v1/query` with a JWT token identifying the case.
2. **Query Service** (`api/services/query_service.py`) — The service computes a cache key as `SHA-256(query + case_id)` and checks Redis. On a cache hit, the stored SSE payload is replayed immediately. On a miss, the service builds an initial `SupervisorState` dictionary and invokes `Supervisor.graph.get_app().stream(state)` via `asyncio.to_thread()`.
3. **Supervisor Graph Execution** — The LangGraph graph executes its 17 nodes across multiple conditional branches, emitting SSE `progress` events per node and a terminal `done` or `error` event.
4. **Response Delivery** — The `final_response` field of the terminal state is extracted, cached in Redis, and streamed to the client.

> **[FIGURE NEEDED: Sequence diagram of the full request lifecycle: client → FastAPI → Redis cache check → Supervisor Graph → SSE stream → client]**

---

## 3.4 Front-End Part

### 3.4.1 Technology Stack

The client-facing application is a multi-page web application served statically by the Node.js back-end. The majority of screens are implemented in plain HTML5, CSS3, and vanilla JavaScript ("use strict" ES6+). The primary AI chat interface (`Case-Chat/CaseChat.jsx`) is implemented as a React functional component, using React hooks (`useState`, `useRef`, `useEffect`) for state management and reactive rendering. The UI framework is Bootstrap (used for modals, toasts, and layout utilities), supplemented by Bootstrap Icons for iconography. Arabic typography is provided by the Cairo and Tajawal typefaces loaded from Google Fonts. The application renders in right-to-left (RTL) mode throughout, using the `dir="rtl"` HTML attribute and CSS mirroring consistent with Arabic script conventions.

### 3.4.2 Application Screens and Navigation Flow

The application comprises thirteen distinct screens, each mapped to a server-side route and corresponding HTML/JavaScript file:

| Route | File | Purpose |
|---|---|---|
| `/login` | `index.html` / `index.js` | Judge authentication (Step 1 — credentials) |
| `/otp` | `otp.html` / `otp.js` | Judge authentication (Step 2 — OTP verification) |
| `/Forget_password` | `forgot.html` / `forgot.js` | Password reset request |
| `/otpForget` | `otpForget.html` / `otpForget.js` | Password reset OTP verification |
| `/resetpassword` | `resetPassword.html` / `resetPassword.js` | New password entry |
| `/dashboard` | `dashboard.html` / `dashboard.js` | Case management dashboard |
| `/chat` | `Case-Chat/case-chat.html` / `CaseChat.jsx` | AI conversational assistant |
| `/OCR` | `ocr.html` / `ocr.js` | OCR text viewer and editor |
| `/legalsearch` | `legalsearch.html` / `legalsearch.js` | Civil law search and article lookup |
| `/Summarize` | `case-summary.html` / `case-summary.js` | Structured case summary viewer |
| `/Settings` | `settings.html` / `settings.js` | Judge profile and account settings |
| `/SupportForm` | `help.html` / `help.js` | Technical support ticket form |
| `/judge-chat` | `judge-chat.html` / `judge-chat.js` | Real-time support chat (token-linked) |

The primary user flow is: **Login → OTP → Dashboard → Case Chat**. All screens except the authentication pages perform a session guard check on load by calling `GET /api/auth`; a non-`200` response immediately redirects to the login screen and clears `sessionStorage`.

> **[FIGURE NEEDED: Navigation flow diagram showing screen transitions: Login → OTP → Dashboard → Chat (primary path), with branches to OCR, Legal Search, Case Summary, and Settings]**

### 3.4.3 Authentication and Login Flow

The login screen (`index.html`) presents a form with three fields: a court-name dropdown (`courtSelect`), a judicial identifier or phone number (`username`), and a password (`password`). On submission, the credentials are sent to `POST /api/login`. On success, a short-lived pre-authentication token (`preAuthToken`, valid for five minutes) is stored in `sessionStorage` and the judge is redirected to the OTP screen (`otp.html`).

The OTP screen displays a masked version of the judge's registered email address and prompts entry of the six-digit code dispatched by the server via email. On successful verification at `POST /api/verify-otp`, an eight-hour JWT session cookie is set and the judge is forwarded to the dashboard. A resend control is available, subject to the same account-lockout limits enforced by the server (five failed attempts lock the account for fifteen minutes).

### 3.4.4 Dashboard — Case Management

The dashboard (`dashboard.html`) displays the judge's full name and judicial number, retrieved from `GET /api/judge/me`, alongside a dual-calendar date widget that renders both the Gregorian date and the corresponding Hijri date (computed via `Intl.DateTimeFormat` with the `islamic-umalqura` calendar extension).

Cases are fetched from `GET /api/ai/cases` with server-side pagination of four cases per page. The case table exposes columns for title, description, document count, status, and creation date. Each row is interactive: clicking a row navigates directly to the Case Chat screen with the selected `case_id` stored in `sessionStorage`. An inline status dropdown per row allows updating case status (`open` / `closed` / `archived`) via `PATCH /api/ai/case/:caseId` without leaving the page. A three-dot action menu per row exposes edit and delete operations through Bootstrap modal dialogs.

Creating a new case composes a title from three structured fields — case number, year, and type (e.g., "١٢٣ / ٢٠٢٤ / مدني") — together with a free-text description, sent via `POST /api/ai/cases/create`.

> **[FIGURE NEEDED: Screenshot of the dashboard case management table with pagination, status filter, and three-dot actions menu]**

### 3.4.5 Case Chat — AI Conversational Interface

The Case Chat screen is the primary AI interaction surface (`CaseChat.jsx`). On load, existing conversation history for the active case is fetched from `GET /api/ai/cases/:caseId/conversations`; the response is parsed for `turns` arrays, each turn containing a `query` (user message) and a `response` (assistant message), which are rendered as a chronological message thread.

When the judge submits a query (Enter key or send button), the text is sent to `POST /api/ai/query` with `{ case_id, query }`. The response is a Server-Sent Events (SSE) stream. The component's `readSSEStream()` function reads the stream incrementally: each `data:` line is parsed as JSON, and the `final_response` field is extracted and appended to the displayed assistant message in real time, producing a typewriter effect. A blinking cursor (`cursorBlink` CSS animation) is rendered at the end of the in-progress message. A three-dot typing indicator (`AILoadingIndicator`) is displayed between the user message and the start of streaming.

A persistent disclaimer banner at the top of the screen reads: "⚠ نظام مساعد ذكي فقط - القاضي يحتفظ بسلطة القرار المطلقة" ("AI advisory system only — the judge retains absolute decision authority"), displayed in amber throughout the session.

The input area supports file attachment via `<input type="file">` and a drag-and-drop drop zone (`onDragOver` / `onDrop` handlers). Accepted file types are PDF, JPEG, PNG, GIF, WebP, SVG, and office documents. Uploaded files are displayed as attachment chips above the message bubble. Files are uploaded to the case via `POST /api/ai/cases/:caseId/upload`, which returns HTTP 202 immediately while OCR and vector ingestion proceed asynchronously in the back-end.

> **[FIGURE NEEDED: Screenshot of the Case Chat interface showing the message thread, streaming response with typing cursor, disclaimer banner, and file attachment area]**

### 3.4.6 OCR Viewer and Editor

The OCR screen (`ocr.js`) operates in two modes, selected via `sessionStorage.getItem("ocr_mode")`:

- **Single mode**: fetches the OCR text of one document via `GET /api/ai/cases/:caseId/documents/:docId/ocr` and renders the extracted text as formatted paragraphs with a word count badge.
- **All mode**: fetches OCR text for multiple documents in parallel (`Promise.allSettled`), presenting a sidebar list of documents. Each sidebar entry shows loading, error, or ready status. Clicking a sidebar entry switches the main panel to that document's text, preserving unsaved edits per document in an `edits` state object.

A floating toolbar at the bottom of the viewport toggles **edit mode**. In edit mode, the read-only paragraph display is replaced with a `<textarea>` pre-filled with the current OCR text. The toolbar tracks the count of edited documents and enables a "Confirm and Send" button that submits all pending edits via parallel `PATCH /api/ai/cases/:caseId/documents/:docId` calls (`saveAllEdits()`). A cancel option discards all in-memory edits after a confirmation dialog.

> **[FIGURE NEEDED: Screenshot of the OCR screen in all-document mode, showing the document sidebar with status indicators and the edit toolbar with confirm/cancel actions]**

### 3.4.7 Legal Search

The legal search screen (`legalsearch.js`) provides two search modes toggled via a sidebar:

- **AI Query mode**: submits a free-text Arabic legal question to `GET /api/ai/legal/search?q=...&corpus=...`, receiving an `{ answer, sources }` response rendered as an AI answer card with a sources block listing the cited legal articles.
- **Direct Article Lookup**: submits an article number to `GET /api/ai/legal/article?article_no=...&corpus=...`, rendering the matching article as a card with article number, law label badge, article text, and structural location (book/chapter).

The sidebar also supports **corpus browsing** via `GET /api/ai/legal/corpus?corpus=...`, which returns a hierarchical tree of books, parts, and articles rendered as section headers and article cards. Three corpora are available: Egyptian Civil Code (`civil`), Evidence Law (`evidence`), and Procedural Law (`procedural`).

An empty state with a branded SVG balance-of-scales illustration (in the system's navy and gold color scheme) is displayed before any search is initiated.

> **[FIGURE NEEDED: Screenshot of the Legal Search screen showing the AI answer card with cited articles and the corpus tree browser in the sidebar]**

### 3.4.8 Case Summary Viewer

The case summary screen (`case-summary.html`) renders the structured Arabic legal brief generated by the Summarization Pipeline. The brief is divided into named sections — facts (`#facts`), parties (`#parties`), timeline (`#timeline`), legal analysis (`#legal`), and contested points (`#contra`) — displayed in a scrollable content pane with a fixed sidebar navigation. A scroll-spy listener on the content container (`contentContainer`) updates the active sidebar item as the judge scrolls, providing visual orientation within long briefs. The judge's name and dual-calendar date are displayed in the header.

### 3.4.9 Settings Screen

The settings screen (`settings.html`) provides two functional sections navigated via a sidebar:

- **Profile section**: displays judge fields (first name, last name, court name, judicial number, email) loaded from `GET /api/judge/me`. A profile avatar supports upload via `POST /api/judge/profile-image` with real-time preview. The avatar can be expanded into a zoomable and pannable lightbox.
- **Security section**: password change form with a six-level strength meter (labels: ضعيفة جداً / ضعيفة / متوسطة / جيدة / قوية / ممتازة) and a match indicator for the confirm-password field. Successful password change calls `PATCH /api/judge/change-password` and then calls `POST /api/logout-all` to invalidate all sessions, forcing re-authentication.

### 3.4.10 Admin Interface

A separate admin interface (`Admin/admin-dashboard.html` / `Admin/admin-dashboard.js`) provides a court administrator panel with its own login and OTP flow. Administrators can search judges by judicial ID (`GET /api/v1/admin/search/:judicialId`), create new judge accounts (`POST /api/v1/admin/signup`), update or delete judge records, add additional administrators, and manage technical support tickets (view unassigned tickets, reply, and open/close). Administrator sessions use the `protectAdmin` middleware, which additionally validates the `role` field against allowed values (`admin`, `superadmin`).

---

## 3.5 Back-End Part

### 3.5.1 General Back-End Architecture

#### 3.5.1.1 Architectural Role and Technology Stack

The web application back-end (`web/Web_app-master/Back-end/`) functions as a **Backend for Frontend (BFF)** layer — a Node.js server that handles all user-facing concerns (authentication, session management, user data, support) while transparently proxying all AI-related requests to the FastAPI AI server running at the address configured by the `MODEL_URL` environment variable.

The server is built on **Express.js v5.2.1** with Node.js. Key dependencies include:

| Dependency | Version | Purpose |
|---|---|---|
| `express` | 5.2.1 | HTTP server and routing |
| `sequelize` | 6.37.8 | PostgreSQL ORM |
| `pg` | 8.20.0 | PostgreSQL driver |
| `jsonwebtoken` | 9.0.3 | JWT issuance and verification |
| `bcrypt` | 6.0.0 | Password hashing (salt rounds = 10) |
| `axios` | 1.15.0 | HTTP proxy to FastAPI AI server |
| `multer` | 2.1.1 | Multipart file upload handling |
| `nodemailer` | 8.0.3 | SMTP email dispatch for OTP codes |
| `node-cron` | 4.2.1 | Scheduled maintenance jobs |
| `express-rate-limit` | 8.3.1 | Request rate limiting |
| `cookie-parser` | 1.4.7 | JWT cookie parsing |
| `cors` | 2.8.6 | Cross-origin resource sharing |

The server is started via `node server.js` (or `nodemon server.js` for development) and listens on port 3000, bound to `0.0.0.0` for local network accessibility.

#### 3.5.1.2 Project Structure

```
Back-end/
├── server.js               # Express app factory, route mounts, static serving
├── db.js                   # Sequelize + PostgreSQL connection
├── Routes/
│   ├── index.js            # Judge + Admin authentication and management routes
│   └── AiRoute.js          # AI proxy routes (cases, query, OCR, legal search)
├── Controller/
│   ├── Judg_Controller/    # authController, JudgeOperation, forget_passwordController
│   ├── Admin_Controller/   # adminAuthController, CRUD_Admin
│   ├── AI/                 # CaseController, ConversationController, FilesUpload,
│   │                       #   query, ocr, LegalSearch, Summary&Reasoner
│   └── SupportController.js
├── Middleware/
│   ├── authMiddleware.js   # JWT + tokenVersion guard (judges)
│   ├── AdminAuth.js        # JWT + role guard (admins)
│   └── uploads.js          # Multer disk-storage for profile images
├── Schema/
│   ├── judge.js            # Sequelize model: judges table
│   ├── admin.js            # Sequelize model: admins table
│   ├── SupportTicket.js    # Sequelize model: support tickets
│   └── SupportMessage.js   # Sequelize model: support messages
└── utils/
    ├── smtp.js             # Nodemailer transporter configuration
    └── cronJobs.js         # Scheduled OTP cleanup + ticket cleanup
```

#### 3.5.1.3 Database Layer — PostgreSQL via Sequelize

The web back-end uses **PostgreSQL** as its relational database, accessed via the Sequelize ORM. This is entirely separate from the MongoDB and Qdrant stores used by the AI layer. Four Sequelize models are defined:

- **`Judge`** (`judges` table): stores `id`, `firstName`, `lastName`, `judicialNumber` (8-char unique), `email`, `phoneNumber`, `courtName`, `nationalId` (14-digit), `password` (bcrypt hash), `otpCode`, `otpExpiry`, `loginAttempts`, `lockUntil`, `tokenVersion`, and `profileImage`. Sequelize lifecycle hooks (`beforeCreate`, `beforeUpdate`) apply bcrypt hashing automatically whenever the `password` field is set.
- **`Admin`** (`admins` table): stores similar fields plus a `role` column defaulting to `'superadmin'`.
- **`SupportTicket`** and **`SupportMessage`**: a one-to-many association (`hasMany` / `belongsTo`) with cascading delete, used for the in-app support system.

The Sequelize connection is initialized with `{ alter: false }` sync, meaning the schema is managed externally; the ORM does not auto-migrate columns on startup.

> **[FIGURE NEEDED: Entity-relationship diagram of the PostgreSQL schema: judges, admins, SupportTickets, SupportMessages with foreign key relationships]**

#### 3.5.1.4 Authentication and Session Management

Authentication follows a **two-factor flow** combining a password check with an email-delivered OTP:

1. **Credential check** (`POST /api/login`): The server queries the `judges` table by `courtName` (case-insensitive, partial match via `Op.iLike`) and `phoneNumber` or `judicialNumber`. Failed password comparisons increment `loginAttempts`; five consecutive failures lock the account for 15 minutes (`lockUntil`). On success, a six-digit OTP is generated, stored as `otpCode` with a five-minute `otpExpiry`, and emailed via Nodemailer. A short-lived pre-authentication JWT (`type: 'pre-auth-judge'`, 5-minute expiry) is returned to the client, identifying the pending session without granting access.

2. **OTP verification** (`POST /api/verify-otp`): The pre-auth token is verified, the OTP matched against the stored value and expiry, and on success the OTP fields are cleared. A full-access JWT (`expiresIn: '8h'`) is issued carrying `{ id, user_id, role: 'judge', version: tokenVersion }` and set as an HTTP-only cookie (`httpOnly: true`, `sameSite: 'Lax'`, `path: '/'`, `domain: 'localhost'`). The cookie is not accessible from JavaScript, preventing XSS token theft.

**Token invalidation** is implemented via a `tokenVersion` integer column on both `Judge` and `Admin` models. The `protect` middleware compares `decoded.version` from the JWT against the current `judge.tokenVersion` in the database. `POST /api/logout-all` increments `tokenVersion` by one, immediately invalidating all previously issued tokens for that judge across all devices without requiring a token blacklist.

The `protect` middleware (`Middleware/authMiddleware.js`) extracts the JWT from `req.cookies.token`, verifies it with `process.env.JWT_SECRET`, fetches the judge record by primary key, and validates `tokenVersion` before calling `next()`. The `protectAdmin` middleware (`Middleware/AdminAuth.js`) performs the same checks and additionally enforces `role ∈ { 'admin', 'superadmin' }`.

> **[FIGURE NEEDED: Sequence diagram of the two-factor authentication flow: browser → POST /api/login → OTP email → POST /api/verify-otp → HTTP-only JWT cookie]**

#### 3.5.1.5 Route Structure

The server mounts two route groups:

**`/api` — User and Admin Routes (`Routes/index.js`):**

| Method | Path | Guard | Purpose |
|---|---|---|---|
| POST | `/login` | — | Initiate judge login; send OTP |
| POST | `/verify-otp` | — | Verify OTP; issue session cookie |
| POST | `/resend-otp` | — | Resend OTP (rate-limited by loginAttempts) |
| POST | `/forgot-password` | — | Trigger password-reset OTP |
| POST | `/verify-reset-otp` | — | Verify reset OTP |
| POST | `/reset-password` | — | Set new password |
| GET | `/auth` | `protect` | Validate active session |
| GET | `/judge/me` | `protect` | Fetch judge profile |
| POST | `/judge/profile-image` | `protect` | Upload profile avatar (disk storage) |
| PATCH | `/judge/change-password` | `protect` | Change password + logout-all |
| POST | `/logout-all` | `protect` | Invalidate all sessions (tokenVersion++) |
| POST | `/v1/admin/login` | — | Admin login |
| POST | `/v1/admin/verify-otp` | — | Admin OTP |
| GET | `/v1/admin/search/:judicialId` | `protectAdmin` | Look up judge |
| POST | `/v1/admin/signup` | `protectAdmin` | Create judge account |
| DELETE | `/v1/admin/judges/:id` | `protectAdmin` | Delete judge |
| PUT | `/v1/admin/judges/:id` | `protectAdmin` | Update judge |
| POST | `/v1/admin/add-admin` | `protectAdmin` | Add administrator |
| POST | `/v1/admin/change-password` | `protectAdmin` | Admin password change |
| POST | `/v1/admin/logout-all` | `protectAdmin` | Admin logout-all |
| GET/POST/PATCH | `/v1/support/admin/*` | `protectAdmin` | Support ticket management |
| POST | `/support/create` | — | Judge creates support ticket |
| GET | `/support/chat/:token` | — | Judge views support chat (token-linked) |
| POST | `/support/chat/:token/reply` | — | Judge replies in support chat |

**`/api/ai` — AI Proxy Routes (`Routes/AiRoute.js`):**

All routes are protected by the `protect` middleware. Each controller method extracts the JWT from `req.cookies.token` and re-issues it as an `Authorization: Bearer <token>` header when calling the FastAPI AI server, bridging the cookie-based web session to the AI server's bearer-token authentication scheme.

| Method | Path | AI Model Endpoint Proxied |
|---|---|---|
| POST | `/cases/create` | `POST /api/v1/cases` |
| GET | `/cases` | `GET /api/v1/cases` |
| GET | `/case/:caseId` | `GET /api/v1/cases/:caseId` |
| PATCH | `/case/:caseId` | `PATCH /api/v1/cases/:caseId` |
| DELETE | `/case/:caseId` | `DELETE /api/v1/cases/:caseId` |
| GET | `/cases/:caseId/conversations` | `GET /api/v1/cases/:caseId/conversations` |
| GET | `/conversations/:conversationId` | `GET /api/v1/conversations/:id` |
| DELETE | `/conversations/:conversationId` | `DELETE /api/v1/conversations/:id` |
| POST | `/query` | `POST /api/v1/query` (SSE stream proxied) |
| POST | `/cases/:caseId/upload` | `POST /api/v1/files/upload` + async ingest |
| GET | `/cases/:caseId/documents` | `GET /api/v1/cases/:caseId/documents` |
| GET | `/cases/:caseId/documents/:docId/ocr` | `GET /api/v1/documents/:docId/ocr` |
| PATCH | `/cases/:caseId/documents/:docId` | `PATCH /api/v1/documents/:docId/ocr` |
| DELETE | `/cases/:caseId/documents/:docId` | `DELETE /api/v1/documents/:docId` |
| GET | `/legal/search` | `GET /api/v1/legal/search` |
| GET | `/legal/article` | `GET /api/v1/legal/article` |
| GET | `/legal/corpus` | `GET /api/v1/legal/corpus/tree` |
| GET | `/cases/:caseId/summary` | `GET /api/v1/cases/:caseId/summary` |
| GET | `/cases/:caseId/case-reasoning` | `GET /api/v1/cases/:caseId/case-reasoning` |

#### 3.5.1.6 File Upload Handling

File uploads follow a two-phase asynchronous pattern implemented in `Controller/AI/FilesUpload.js`:

1. Files arrive at `POST /api/ai/cases/:caseId/upload` via `multer()` with in-memory storage. Up to 10 files per request are accepted; MIME types are validated against an allowlist (`image/jpeg`, `image/png`, `image/gif`, `image/webp`, `application/pdf`, `text/plain`). Files that fail MIME validation are rejected with HTTP 415.

2. Accepted files are uploaded in parallel to the AI server's `/api/v1/files/upload` endpoint via Axios (`Promise.allSettled`, 15-second timeout per file). Each successful upload returns a `file_id`.

3. The controller responds HTTP 202 to the browser immediately with `{ ingest_status: "processing" }`, then triggers the AI ingest call (`POST /api/v1/cases/:caseId/documents`) asynchronously in the background (120-second timeout), decoupling the UI wait time from the heavyweight OCR and Qdrant ingestion pipeline.

The SSE query stream (`POST /api/ai/query`) is proxied differently: Axios is configured with `responseType: 'stream'`, and the response body's `data` events are piped directly to the Express response via `res.write(chunk)`. The `X-Conversation-Id` header returned by the AI server is forwarded to the browser via the `exposedHeaders` CORS configuration.

#### 3.5.1.7 Scheduled Maintenance Jobs

Two scheduled tasks run continuously via `node-cron` (`utils/cronJobs.js`):

- **Hourly OTP cleanup** (`0 * * * *`): Sets `otpCode` and `otpExpiry` to `null` for all judge records whose OTP has expired (`otpExpiry < now`). This prevents stale one-time codes from persisting in the database.
- **Support ticket cleanup** (every 10 minutes, `*/10 * * * *`): Permanently deletes `SupportTicket` records that have been in `closed` status for more than 30 minutes, along with their associated `SupportMessage` records (cascade delete).

#### 3.5.1.8 CORS and Static File Serving

CORS is configured to allow credentials from three origins: `http://localhost:3000`, `http://127.0.0.1:3000`, and `http://localhost:4000`, with an additional dynamic allow-list for `192.168.x.x` addresses to support local network access from mobile or tablet devices. The `X-Conversation-Id` header is listed in `exposedHeaders` so that the front-end JavaScript can read it from SSE responses.

The Express server statically serves the entire `Front-end/` directory, extending file resolution to `.html` and `.htm` suffixes. The `Uploads/` directory (profile images) is separately mounted at the `/uploads` URL prefix.

### 3.5.2 AI Layer

#### 3.5.2.1 OCR Pipeline

Uploaded image files and PDF scans are processed by a five-stage OCR pipeline (`OCR/ocr_pipeline.py`) whose entry point is `run_ocr(file_path, doc_id, config) → OCRDocumentResult`.

**Stage 1 — Ingestion** (`ingestion.py`): PDF pages are rasterized at 400 DPI to produce high-fidelity images suitable for Arabic script recognition.

**Stage 2 — Restoration** (`restoration.py`): CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to compensate for faded ink and uneven illumination common in archival legal documents.

**Stage 3 — Perspective Correction** (`perspective_correction.py`): Canny edge detection combined with an adaptive-threshold fallback corrects document skew. Safety guards enforce that the corrected output maintains a crop ratio of at least 0.65 and occupies at least 35% of the original image area; otherwise the original image is used unchanged.

**Stage 4 — OCR Engine** (`ocr_engine.py`): Text is extracted by **NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct**, a Qwen2VL-based vision-language model specialized for Arabic legal document transcription. The model is loaded in 8-bit quantization with `float16` precision on GPU (`max_new_tokens=4000`). The model is instructed via a strict prompt to transcribe text character-for-character without correction, preserving every error and archaic spelling as written. Per-page confidence scores are derived from token-level softmax probabilities.

**Stage 5 — Text Reconstruction** (`text_reconstruction.py`): Eastern Arabic-Indic numerals are normalized to Western Arabic numerals for downstream compatibility.

The pipeline returns an `OCRDocumentResult` containing a list of `OCRPageResult` objects, each carrying `page_num`, `raw_text`, `confidence` (0.0–1.0), and any `error`.

> **[FIGURE NEEDED: Five-stage OCR pipeline flowchart: Ingestion → Restoration → Perspective Correction → OCR Engine (Qari-OCR) → Text Reconstruction → OCRDocumentResult]**

#### 3.5.2.2 Document Classification

Following OCR, documents are classified by the `DocumentProcessor/classifier.py` module through a two-stage pipeline.

**Stage 1 — Weighted Keyword Heuristic**: The Arabic text (after diacritic removal and Unicode normalization via `arabic_norm.py`) is scored against every document type defined in `config/document_taxonomy.yaml`. Each document type carries three keyword lists — strong, weak, and anti — with weights of +60, +10, and −30 points respectively. The heuristic short-circuits and returns the top candidate if that candidate contains at least one strong keyword hit and its score exceeds the second-place candidate by a margin of ≥ 30 points (`_AMBIGUITY_MARGIN`).

**Stage 2 — LLM Classification**: When the heuristic produces an ambiguous result, the top-3 candidates are passed as hints to a Gemini-2.5-Flash LLM call, which returns a structured `_ClassificationResult` containing `doc_type`, `confidence` (0–100), and `reasons` in Arabic. The classified document type is then used to select the appropriate party-role mapping for downstream summarization.

#### 3.5.2.3 Civil Law RAG Agent

The Civil Law RAG agent (`RAG/civil_law_rag/graph.py`) is a 10-node LangGraph graph responsible for answering questions grounded in the Egyptian Civil Code.

**Graph Topology:**

```
START
  → preprocessor_node
  → [classification router]
      ├─→ off_topic_node → END
      ├─→ textual_node   → END
      └─→ scope_classifier_node
            → retrieve_node
            → rule_grader_node
            → [grade router]
                ├─→ generate_answer_node → END
                ├─→ refine_node → retrieve_node   (retry loop)
                ├─→ llm_grader_node
                │     → [llm grade router]
                │         ├─→ generate_answer_node → END
                │         └─→ cannot_answer_node  → END
                └─→ cannot_answer_node → END
```

**Key mechanisms:**

- *Preprocessing* (`preprocessor_node`): validates query length (3–2,000 characters), enforces a minimum Arabic character ratio of 0.30, and normalizes Unicode.
- *Scope classification* (`scope_classifier_node`): an LLM call emitting a `scope_confidence` (0–1) and a `scope_filter` metadata dictionary that restricts subsequent Qdrant queries to the relevant book/chapter of the Civil Code.
- *Retrieval* (`retrieve_node`): performs a filtered Qdrant similarity search over `judicial_docs` using `BAAI/bge-m3` embeddings from the TEI server at `http://localhost:8080`, returning `k=8` top documents by default.
- *Grading* (`rule_grader_node`, `llm_grader_node`): a two-pass relevance check; the rule-based grader runs first to avoid LLM calls on clearly irrelevant retrievals. The `llm_call_count` budget is capped at `MAX_LLM_CALLS=5`.
- *Semantic caching*: repeated queries with cosine similarity ≥ 0.97 (`CACHE_SIMILARITY_THRESHOLD`) to a cached query are served directly from a local LRU cache (max 500 entries) without a Qdrant round-trip.
- *Query refinement* (`refine_node`): on a failed grade, the query is reformulated and retrieval retried up to `max_retries=3`.

> **[FIGURE NEEDED: Civil Law RAG graph node flow: preprocessing → scope classification → retrieval → dual-pass grading → answer generation or query refinement loop]**

#### 3.5.2.4 Case Document RAG Agent

The Case Document RAG agent (`RAG/case_doc_rag/graph.py`) retrieves information from documents belonging to a specific case. Its defining architectural feature is **parallel fan-out via LangGraph's `Send()` primitive**, which spawns one independent branch per decomposed sub-question.

**Main graph:**

```
START
  → questionRewriter
  → questionClassifier
  → [onTopicRouter]
      ├─→ offTopicResponse → END
      ├─→ errorResponse   → END
      └─→ documentSelector
            → [docSelectorDispatchRouter]
                ├─→ errorResponse → END
                └─→ Send([SubQuestionState, ...]) → retrieve_branch × N
                                                  → mergeAnswers → END
```

**Branch sub-graph (per `SubQuestionState`):**

```
START
  → branchDocSelector
  → [branchDocSelectorRouter]
      ├─→ BranchDocumentFinalizer → END
      └─→ retrieve
            → retrievalGrader
            → [proceedRouter]
                ├─→ generateAnswer  → END
                ├─→ refineQuestion  → retrieve   (retry)
                └─→ cannotAnswer    → END
```

Each branch operates on a `SubQuestionState` carrying its own `sub_question`, `case_id`, `doc_selection_mode` (one of `retrieve_specific_doc`, `restrict_to_doc`, or `no_doc_specified`), and a per-branch `retrieved_docs` list with an `operator.add` reducer enabling safe concurrent writes. The `AgentState.sub_answers` field also uses an `operator.add` reducer to collect results from all parallel branches without race conditions.

The `documentSelector` node fetches document titles for the case from MongoDB and passes them to a classification step that determines whether the query targets a specific named document or should retrieve broadly. The `branchDocSelector` further refines this decision at the per-branch level.

Vector retrieval hits the `case_docs` Qdrant collection, filtered by `case_id` to ensure strict isolation between cases.

> **[FIGURE NEEDED: Case Doc RAG fan-out architecture: main graph spawning N parallel branch sub-graphs via Send(), each independently retrieving and grading, converging at mergeAnswers]**

#### 3.5.2.5 Supervisor Orchestration

The Supervisor (`Supervisor/graph.py`) is a 17-node LangGraph state machine that coordinates all agents and enforces a validation-retry loop. Its shared state is the `SupervisorState` TypedDict.

**SupervisorState (selected fields):**

| Field | Type | Description |
|---|---|---|
| `judge_query` | `str` | Raw Arabic query from judge |
| `intent` | `str` | Classified intent: `civil_law_rag \| case_doc_rag \| reason \| multi \| off_topic` |
| `target_agents` | `List[str]` | Agents selected by classifier |
| `agent_results` | `Dict[str, Any]` | Shape: `{agent_name: {response, sources, raw_output}}` |
| `agent_errors` | `Dict[str, str]` | Per-agent error messages |
| `merged_response` | `str` | LLM-synthesized multi-agent output |
| `final_response` | `str` | Validated, formatted answer |
| `validation_status` | `str` | `pass \| partial_pass \| fail_* \| fallback` |
| `retry_count` | `int` | Current retry; max = `MAX_RETRIES` (2) |
| `running_summary` | `Optional[str]` | Compressed older turns (Arabic) |
| `semantic_facts` | `List[Dict]` | Loaded long-term case facts |
| `procedural_prefs` | `Optional[str]` | Judge behavioral preferences |
| `correlation_id` | `Optional[str]` | Per-turn UUID for distributed tracing |

**Graph topology (abbreviated):**

```
START → validate_input → [router]
  ├─→ off_topic_response → END
  └─→ load_long_term_memory → classify_intent → [router]
        ├─→ off_topic_response → END
        └─→ enrich_context → dispatch_agents → [router]
              ├─→ classify_and_store_document → merge_responses
              └─→ merge_responses
                    → verify_citations → validate_output → [router]
                          ├─→ update_memory (pass / partial_pass)
                          ├─→ prepare_retry → dispatch_agents   (retry loop)
                          └─→ fallback_response → END

update_memory → write_long_term_memory → summarize_history → audit_log → END
```

**ADAPTER_REGISTRY and Execution Tiers:**

Agent dispatch (`dispatch_agents.py`) uses an `ADAPTER_REGISTRY` mapping intent names to adapter classes:

```python
ADAPTER_REGISTRY = {
    "civil_law_rag": CivilLawRAGAdapter,
    "case_doc_rag":  CaseDocRAGAdapter,
    "reason":        ChatReasonerAdapter,
}
```

Adapters are partitioned into execution tiers based on inter-agent dependencies:
- **Tier 0 (parallel)**: `civil_law_rag` and `case_doc_rag` — no dependencies, executed concurrently via `ThreadPoolExecutor`.
- **Tier 1 (sequential)**: `reason` — executes after Tier 0 and can read Tier 0 results from `state.agent_results`.

**Validation Pipeline** (`validate_output.py`): The `validate_output_node` invokes a low-tier LLM (Gemini-2.5-Flash-Lite) to evaluate the merged response on four boolean dimensions:

1. `hallucination_pass` — response is grounded in retrieved sources
2. `relevance_pass` — response addresses the judge's query
3. `completeness_pass` — all aspects of the query are addressed
4. `coherence_pass` — response does not contradict the prior conversation turn

`overall_pass` is `True` only when hallucination, relevance, and coherence all pass. When completeness alone fails while the other three pass, the system assigns `partial_pass` status and appends a disclosure caveat rather than retrying. Validation failures that do not exceed `MAX_RETRIES` trigger `prepare_retry_node`, which resets agent results and re-dispatches with the `validation_feedback` string appended to the prompt context.

> **[FIGURE NEEDED: Supervisor LangGraph node diagram showing all 17 nodes, conditional edges, retry loop (dispatch_agents ↔ prepare_retry), and terminal paths (update_memory chain, fallback, off_topic)]**

#### 3.5.2.6 Chat Reasoner Agent

The Chat Reasoner (`chat_reasoner/`) implements a **plan-then-execute-replan** loop for complex multi-hop legal queries that require chaining retrieval steps. It is invoked when the Supervisor classifies intent as `reason`.

**ChatReasonerState (selected fields):**

| Field | Type | Description |
|---|---|---|
| `plan` | `List[dict]` | Serialized `PlanStep` objects |
| `step_results` | `Annotated[List[dict], add_or_reset]` | Per-step outputs (concurrent-safe reducer) |
| `step_failures` | `Annotated[Dict[str, int], merge_max]` | Failure count per step |
| `replan_count` | `int` | Replan iterations; capped at 2 |
| `run_count` | `int` | Synthesis attempts; capped at 2 |
| `final_answer` | `str` | Synthesized Arabic legal answer |
| `tool_calls_log` | `Annotated[List[dict], add]` | Append-only trace of all tool invocations |
| `replan_events` | `Annotated[List[dict], add]` | Append-only replan trigger history |

**PlanStep schema:**

```python
class PlanStep(BaseModel):
    step_id: str
    tool: Literal["case_doc_rag", "civil_law_rag", "fetch_summary_report"]
    query: str
    depends_on: List[str]
```

**Graph topology:**

```
START
  → planner
  → plan_validator → [validator_router]
        ├─→ planner        (invalid plan, retry)
        ├─→ replanner      (structurally valid but logically flawed)
        └─→ executor_fanout

executor_fanout ─(Send list)─→ step_worker × N
                                   → collector → [collector_router]
                                       ├─→ executor_fanout  (step retry)
                                       ├─→ synthesizer
                                       └─→ replanner

synthesizer → [synth_router]
  ├─→ trace_writer → END
  └─→ replanner → plan_validator

replanner → [replanner_router]
  ├─→ trace_writer  (status = "failed")
  └─→ plan_validator

trace_writer → END
```

The `planner` node uses a high-tier LLM to decompose the judge's query into a directed acyclic graph of `PlanStep` objects, each targeting one of three tools. The `executor_fanout` node uses LangGraph's `Send()` API to spawn one `step_worker` coroutine per plan step whose dependencies are satisfied, enabling parallel execution of independent retrieval steps. The `collector` aggregates step results and determines whether to proceed to synthesis, retry failed steps, or invoke the replanner. The `replanner` is constrained to `replan_count ≤ 2` replanning iterations; exceeding this limit sets `status = "failed"` and routes to `trace_writer`. The `trace_writer` node persists the complete tool call log and replan event history for auditability.

> **[FIGURE NEEDED: Chat Reasoner LangGraph node flow showing planner → plan_validator → executor_fanout → parallel step_workers → collector → synthesizer/replanner loop with replan_count cap]**

#### 3.5.2.7 Summarization Pipeline

The Summarization Pipeline (`summarize/`) produces a structured seven-section Arabic legal brief from a collection of case documents. The pipeline is a 7-node LangGraph graph with entry point `run_summarization()`.

**Node sequence:**

| Node | Class | Function |
|---|---|---|
| 0 | `node_0_intake` | Text cleaning (remove RTL markers, diacritics `ـ`), metadata extraction, document segmentation into `NormalizedChunk` objects |
| 1 | `node_1_classify` | Role classification of each chunk: plaintiff, defendant (ordinal-ranked with gender agreement), expert, court |
| 2 | `node_2_extract` | Atomic bullet-point extraction of legal claims, facts, and arguments per chunk into `LegalBullet` objects |
| 3 | `node_3_aggregate` | Aggregation of bullets by party role; categorization into agreed facts, disputed facts, and party-specific positions |
| 4a | `node_4a_cluster` | Thematic clustering of role-aggregated bullets into coherent legal topics |
| 4b | `node_4b_synthesize` | 2–3 paragraph Arabic narrative summaries per theme |
| 5 | `node_5_brief` | Final seven-section judge-facing Arabic brief generation: stored in `CaseBrief` object, rendered as `rendered_brief` (Arabic Markdown) |

**Arabic defendant disambiguation**: Documents typed as مذكرة دفاع or مذكرة رد trigger ordinal-rank detection from the `doc_id`. Defendants are labelled with gender-correct Arabic ordinals (e.g., "المدعى عليه الثالث" for masculine third defendant) using a predefined map for ranks 1–10.

**Persistence**: The `rendered_brief` and `party_manifest` are written to MongoDB via Motor (async driver) upon pipeline completion, associated with the relevant `case_id`.

> **[FIGURE NEEDED: Summarization pipeline node flow: intake → role classification → bullet extraction → role aggregation → thematic clustering → theme synthesis → Arabic brief generation → MongoDB persistence]**

#### 3.5.2.8 Memory Architecture

Hakim implements a three-tier memory architecture that mirrors the working, short-term, and long-term memory distinctions common in cognitive architectures.

**Tier 1 — Working Memory (In-Flight State)**

`SupervisorState` functions as working memory for the duration of a single conversation turn. It holds the active `judge_query`, `agent_results`, `merged_response`, and all intermediate routing flags. State is passed by value through the LangGraph node chain and is discarded at turn completion, having been durably checkpointed in Tier 2.

**Tier 2 — Short-Term Memory (Conversation Persistence)**

Two mechanisms constitute short-term memory:

- *LangGraph MongoDBSaver checkpointing*: After each turn, LangGraph serializes the full `SupervisorState` to the `supervisor_checkpoints` MongoDB collection (database: `TESTING`), keyed by `(thread_id, checkpoint_id)`. This enables exact conversation replay and crash recovery.
- *Conversation compression* (`summarize_history_node`): When `messages_since_last_summary` reaches the `SUMMARIZE_EVERY_N_MESSAGES` threshold (trigger at `SUMMARIZE_TRIGGER_TOKENS=4000` tokens equivalent), the `summarize_history_node` compresses older turns into a rolling Arabic-language `running_summary`. The compressed summary is stored in `SupervisorState.running_summary` via a `_keep_non_none` custom reducer that never overwrites a non-null value with null. Only the most recent `SHORT_TERM_KEEP_TURNS=6` turns are retained in verbatim `conversation_history`; the remainder is represented by `running_summary`.

**Tier 3 — Long-Term Memory (Cross-Session Persistence)**

Long-term memory is managed by `write_long_term_memory_node` and `load_long_term_memory_node`, operating over a LangGraph `MongoDBStore` instance persisted to the `supervisor_memory_store` collection.

Three namespaced sub-stores are maintained:

| Sub-store | Namespace | Content | Persistence mode |
|---|---|---|---|
| Semantic facts | `("case", case_id, "facts")` | Factual case knowledge extracted per turn | Synchronous |
| Episodic memories | `("case", case_id, "episodes")` | Session episode summaries | Async, delay = 300 s |
| Procedural preferences | `("user", user_id, "prefs")` | Judge query and interaction preferences | Async |

On each turn start, `load_long_term_memory_node` retrieves the top `SEMANTIC_FACTS_TOP_K=10` semantic facts and up to `PROCEDURAL_INJECT_MAX_CHARS=2000` characters of procedural preferences, injecting them into `SupervisorState` to condition agent responses. Episodic memories are written by a background `ReflectionExecutor` after a 300-second delay (`EPISODIC_REFLECT_DELAY_S`) to avoid blocking the response path.

> **[FIGURE NEEDED: Three-tier memory architecture diagram: Working Memory (SupervisorState) → Short-Term (MongoDBSaver checkpoints + running_summary compression) → Long-Term (MongoDBStore: semantic_facts, episodic_memories, procedural_prefs with async reflection)]**

---

## 3.6 Evaluation Phase

### 3.6.1 Evaluation Metrics

System performance is evaluated across four quality dimensions aligned with the requirements of Arabic legal document processing:

**Retrieval Quality**
- *Precision@k* and *Recall@k*: proportion of retrieved documents that are relevant, and proportion of relevant documents retrieved, at cutoff k=8 for both RAG agents.
- *Semantic Similarity*: cosine similarity between query and retrieved document embeddings in the `BAAI/bge-m3` embedding space.

**Generation Quality (RAGAS-aligned)**
- *Faithfulness*: proportion of claims in the generated response that are entailed by the retrieved documents. Measured by the `validate_output_node`'s `hallucination_pass` check at inference time and by offline RAGAS evaluation over golden-set queries.
- *Answer Relevance*: semantic alignment between the judge query and the generated answer, corresponding to `relevance_pass`.
- *Context Recall*: proportion of ground-truth answer elements attributable to the retrieved context.

**Summarization Quality**
- *Coverage*: proportion of key legal claims from source documents that appear in the generated `rendered_brief`.
- *Arabic fluency*: assessed via automated readability measures and human expert review (not automated).

**End-to-End System Quality**
- *Validation pass rate*: proportion of turns achieving `validation_status = "pass"` without retry.
- *Partial-pass rate*: proportion reaching `partial_pass`.
- *Retry rate*: proportion requiring at least one `prepare_retry` cycle.
- *Fallback rate*: proportion reaching `fallback_response`.

### 3.6.2 Test Suite

The test suite (`tests/`) is organized into four groups:

**Supervisor tests** (`tests/supervisor/`):
- `memory/test_01_short_term.py` — MongoDBSaver checkpoint round-trip correctness.
- `memory/test_02_long_term.py` — MongoDBStore write/load for all three sub-stores.
- `memory/test_03_crash_safety.py` — State recovery after simulated process interruption.
- `memory/test_04_episodic_procedural.py` — Async episodic and procedural memory reflection.
- `unit_nodes/test_*.py` — Individual node unit tests covering classify_intent, merge_responses, validate_output, and dispatch_agents.
- `e2e/test_e2e_*.py` — End-to-end integration tests exercising full turn cycles.
- `failure/test_*.py` — Fault injection tests: database outage, LLM timeout, and adapter crash scenarios.

**Summarizer tests** (`tests/summarizer/`):
- `test_node_0.py` through `test_node_5.py` — Per-node unit tests validating text cleaning (assertion: no RTL marker `‏`/`‎`, no tatweel `ـ`), metadata extraction, role classification, bullet extraction, aggregation, clustering, and brief generation.
- `test_graph.py` — Full graph execution test over synthetic Arabic legal document sets.
- `test_data_contracts.py` — Pydantic schema validation for all intermediate state objects.
- `test_eval_quality.py` — LLM-graded output quality checks against reference briefs.

**RAG tests** (`tests/legal_rag/`, `tests/CASE_RAG/`):
- Graph path validation ensuring correct conditional routing.
- Node-level execution tests for retrieval, grading, and generation.
- Golden-set evaluation: known queries with reference answers for retrieval quality measurement.

**API tests** (`tests/api/`):
- Router-level integration tests for all ten endpoint groups.
- Bulk OCR correction endpoint (`POST /api/v1/documents/ocr/bulk`) tested for per-item error isolation and 207 Multi-Status response structure.

### 3.6.3 Quantitative Results

> **[EVALUATION DATA NEEDED: Insert measured values for the following metrics after running the full evaluation suite:]**
>
> - Civil Law RAG: Precision@8, Recall@8, Faithfulness, Answer Relevance (RAGAS)
> - Case Doc RAG: Precision@8, Recall@8, Faithfulness, Answer Relevance (RAGAS)
> - Supervisor: Validation pass rate, partial-pass rate, retry rate, fallback rate
> - Summarization: Coverage score, fluency score
> - OCR: Character Error Rate (CER) on Arabic legal document test set
> - Document Classifier: Accuracy, macro-F1 across taxonomy categories
> - End-to-end: Mean response latency (ms), P95 latency, cache hit rate

---

*All class names, node names, collection names, configuration constants, and data-flow descriptions in this chapter are derived directly from the Hakim codebase as implemented. No values have been inferred or approximated.*
