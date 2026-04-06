# Glossary

Bilingual reference of terms used throughout the Judge Assistant (Hakim) codebase and documentation.

## System Components

| Term | Arabic | Definition |
|---|---|---|
| **Supervisor** | المشرف | The LangGraph state machine that orchestrates intent classification, agent dispatch, response merging, output validation, and conversation memory. Entry point: `Supervisor/graph.py`. |
| **Agent** | وكيل | A specialist worker that handles one type of task (OCR, summarization, RAG, reasoning). Each agent is wrapped in an adapter that implements `AgentAdapter`. |
| **Adapter** | محوّل | A thin wrapper class (subclass of `AgentAdapter` from `Supervisor/agents/base.py`) that translates between the Supervisor's uniform interface and the underlying pipeline. |
| **Pipeline** | خط أنابيب | A multi-step processing workflow, typically implemented as a LangGraph graph or a chain of function calls. Examples: OCR pipeline, Summarization pipeline, Civil Law RAG pipeline. |
| **RAG** | الاسترجاع المعزز بالتوليد | Retrieval-Augmented Generation. A pattern that retrieves relevant documents from a vector store before generating an LLM response grounded in those documents. |

## LangGraph Concepts

| Term | Arabic | Definition |
|---|---|---|
| **State** | الحالة | A `TypedDict` (`SupervisorState` in `Supervisor/state.py`) that flows through every node in the graph. Each node reads from and writes to this shared state. |
| **Node** | عقدة | A Python function registered in the `StateGraph`. Each node receives the current state, performs work (e.g., LLM call, database query), and returns a partial state update. |
| **Edge** | حافة | A connection between two nodes in the graph. Can be unconditional (always follows) or conditional (dispatched by a router function). |
| **Conditional Edge** | حافة شرطية | An edge that evaluates a router function against the current state to decide which node to visit next. Used for intent routing, validation retry logic, and fallback handling. |
| **Graph** | رسم بياني | The compiled `StateGraph` object. Built in `Supervisor/graph.py` via `build_supervisor_graph()` and invoked with `graph.stream(state)`. |

## Legal Domain Terms

| Term (Arabic) | English | Definition |
|---|---|---|
| **صحيفة دعوى** | Statement of Claim | The initial document filed by the plaintiff to start a lawsuit. |
| **مذكرة دفاع** | Defense Memo | A written response from the defendant addressing the plaintiff's claims. |
| **مذكرة رد** | Reply Memo | A follow-up memo responding to the defense memo. |
| **حافظة مستندات** | Evidence Portfolio | A collection of supporting documents submitted as evidence. |
| **محضر جلسة** | Hearing Minutes | Official record of proceedings during a court session. |
| **حكم تمهيدي** | Preliminary Judgment | An interim ruling issued before the final judgment. |
| **المدعي** | Plaintiff | The party who initiates the lawsuit. |
| **المدعى عليه** | Defendant | The party against whom the lawsuit is filed. |
| **النيابة** | Prosecution | The public prosecution authority. |
| **خبير** | Expert | A court-appointed expert witness. |

## Legal Role Classifications

These roles are used by the Summarization pipeline (`Summerize/schemas.py`) to classify text chunks:

| Role (Arabic) | English | Definition |
|---|---|---|
| **الوقائع** | Facts | The narrative of events, timeline, and circumstances of the case. |
| **الطلبات** | Requests | What the party is asking the court to rule or order. |
| **الدفوع** | Defenses | Legal or procedural arguments raised against the opposing party. |
| **المستندات** | Evidence | References to attached documents and exhibits. |
| **الأساس القانوني** | Legal Basis | Citations of specific law articles or legal principles. |
| **الإجراءات** | Procedures | Case history, previous hearings, and procedural steps taken. |
| **غير محدد** | Unspecified | Text that does not fit into any of the above categories. |

## Database Concepts

| Term | Arabic | Definition |
|---|---|---|
| **Vector Store** | مخزن المتجهات | A database (Qdrant) that stores document embeddings and supports similarity search. |
| **Embedding** | تضمين | A dense numerical vector representation of text, produced by the `BAAI/bge-m3` model. Dimension: 1024. |
| **Chunk** | جزء | A segment of a larger document, split for embedding and retrieval. Each chunk has metadata (page number, paragraph, source document). |
| **Collection** | مجموعة | A named group of vectors in Qdrant. The system uses `judicial_docs` for civil law articles and `case_docs` for case-specific documents. |
| **Payload Index** | فهرس الحمولة | A Qdrant index on non-vector metadata fields (`case_id`, `doc_type`) to enable filtered similarity search. |

## Quality & Validation

| Term | Arabic | Definition |
|---|---|---|
| **Hallucination** | هلوسة | When the LLM generates information not grounded in the retrieved source material. Checked by `hallucination_pass` in `ValidationResult`. |
| **Relevance** | صلة | Whether the generated response actually addresses the judge's query. Checked by `relevance_pass`. |
| **Completeness** | اكتمال | Whether all aspects of the query are covered in the response. Checked by `completeness_pass`. |
| **Faithfulness** | أمانة | The degree to which the response accurately reflects the source documents without distortion. |
| **Context Recall** | استدعاء السياق | A RAG evaluation metric measuring how much of the relevant context was retrieved from the vector store. |

## API & Infrastructure Concepts

| Term | Arabic | Definition |
|---|---|---|
| **SSE** | أحداث مرسلة من الخادم | Server-Sent Events. A streaming protocol used by the `/query` endpoint to push progress updates and the final response to the client in real time. Event types: `progress`, `result`, `error`, `done`. |
| **JWT** | رمز ويب JSON | JSON Web Token. Used for authentication. Issued by the Express backend, validated by the API using a shared secret (`HS256` algorithm). Must contain a `user_id` claim. |
| **Session** | جلسة | A conversation session tied to a case, stored in MongoDB's `conversations` collection. Each session contains an ordered list of query/response turns. |
| **Intent** | نية | The classified purpose of a judge's query. One of: `ocr`, `summarize`, `civil_law_rag`, `case_doc_rag`, `reason`, `multi`, `off_topic`. Determined by `classify_intent` node. |
| **Tier** | مستوى | LLM tier system (`high`, `medium`, `low`) mapping task complexity to model configuration. All tiers currently use Groq `llama-3.3-70b-versatile`. |
