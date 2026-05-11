# Methodology

## 1. Methodological Overview

### 1.1 Overview
This project develops an AI-based judicial assistant to support judges in Egyptian civil courts during the pre-hearing stage. The system improves access to case information by organizing documents, retrieving relevant facts, and generating structured summaries—without participating in legal decision-making.

The architecture coordinates four specialized agents—civil law retrieval, case document retrieval, case reasoning, and conversational reasoning—through a central supervisory component. Each retrieval agent operates as an isolated subprocess; the supervisor communicates with it via JSON-RPC messages over standard input/output streams, a pattern referred to as the MCP transport layer. The system is exposed through a REST API that requires JWT authentication and delivers query responses as a real-time event stream.

### 1.2 Objectives of the Methodology
The methodology aims to enable efficient review of large, complex case files; ground all outputs exclusively in official case documents or authorized legal texts; ensure complete traceability of information to original sources; and provide structured, neutral interaction with case materials. The focus is on reliability and transparency rather than technical complexity.

### 1.3 Project Scope
The system supports judges during pre-hearing preparation for civil cases. Its role is strictly limited to **informational support**: organizing documents, retrieving facts, and summarizing content. It explicitly excludes legal interpretation, evidence evaluation, or decision-making.

---

## 2. Data Acquisition and Preparation

This chapter explains how case documents and legal texts are collected and prepared for use in the system.

### 2.1 Case Document Ingestion
Civil case documents are uploaded in their original form through a file upload endpoint that accepts PDF documents and common image formats up to twenty megabytes per file. Uploaded files are stored in an S3-compatible object storage system. Image-based documents are processed through an OCR pipeline using the Surya engine configured for Arabic; each result is assigned a confidence tier—high (at or above 0.85), medium (at or above 0.60), or low—and undergoes post-processing including dictionary correction, Arabic-Indic digit normalization, deskewing, and contrast enhancement. The extracted text is segmented and indexed into a vector store collection dedicated to the active case, and document metadata is recorded in MongoDB.

### 2.2 Incremental Document Upload Handling
New documents may be added at any time as cases evolve. These are processed and integrated without reprocessing existing files. When OCR output requires correction, a dedicated endpoint accepts corrected text and re-indexes the document in the vector store, ensuring retrieval reflects the corrected content. Submission order is preserved to maintain chronological clarity.

### 2.3 Legal Corpus Collection and Structuring
Official Egyptian legal texts are collected independently from case files and organized into three bodies of law: the Egyptian Civil Code, the Law of Evidence, and the Civil and Commercial Procedure Code. Each corpus is indexed into its own vector store collection. Vector embeddings are produced by a remote Text Embeddings Inference (TEI) service using the BAAI/bge-m3 model; a separate TEI reranker service re-scores retrieval results for precision. An in-process embedding model serves as a fallback when the remote service is unavailable.

### 2.4 Text Preprocessing and Metadata Annotation
All texts undergo basic cleaning and segmentation into logical sections. For legal corpus chunks, metadata fields indexed alongside each segment include source identifier, content type, article number, chapter, section, book, and part. Payload indexes are created on all these fields in the vector store to support filtered retrieval during legal article lookup.

---

## 3. Retrieval-Augmented Generation (RAG) Architecture

The system retrieves information through two independent channels.

### 3.1 Case-Specific RAG System
This component works exclusively with documents related to the active civil case. The system first rewrites the judge's query, then decomposes it into sub-questions and processes each sub-question in a separate parallel branch. Within each branch, a document selector classifies the sub-question into one of three retrieval modes: return a specific document in full, restrict retrieval to within a single named document, or search across all case documents without restriction. For sub-questions entering retrieval, a grader evaluates retrieved passages and permits up to two rephrase-and-retry cycles before returning an unavailability response. Answers from all branches are merged into a single response.

### 3.2 Legal Articles RAG System
A separate retrieval component handles Egyptian legal texts across the three corpora. A corpus router node scores the query against all three corpora using an LLM call and selects the best match before any retrieval takes place. Retrieval is preceded by scope classification, which identifies the relevant chapter and section to narrow the search space. A two-stage grading pipeline—heuristic rule grading followed by semantic LLM grading—evaluates retrieved passages before answer generation. When HyDE is enabled in configuration, the system generates a hypothetical legal article and paraphrases to expand the query before retrieval. A semantic cache per corpus stores results keyed by query, corpus version, and prompt version to avoid redundant LLM calls for repeated questions.

### 3.3 MCP Transport Layer
Each retrieval agent runs as a child process managed by an MCP client. Communication uses newline-delimited JSON-RPC 2.0 over subprocess standard input/output. On transport failure, the client terminates the child process, spawns a replacement, and retries the call once; timeout errors are not retried, as the server remains alive and processing. Both servers complete graph compilation, vector store initialization, and reranker probing at process startup rather than on the first request.

### 3.4 Architectural Separation and Source Traceability
The two retrieval components operate independently. Case facts and legal texts are handled separately, each with a defined role, to prevent conflation of facts with legal rules and ensure the system does not generate hybrid conclusions. Each server returns a structured response containing the answer, source references, and a citation integrity assessment; the supervisor performs an additional citation verification pass before output validation.

---

## 4. Large-Scale Case Summarization Methodology

Civil case files often contain numerous documents submitted over extended periods. This chapter describes how the system converts fragmented case materials into clear, structured summaries that support judicial preparation without replacing judicial judgment.

The summarization process preserves legally relevant facts, maintains neutrality, and enables traceability to original documents.

### 4.1 Handling Different Case Sizes and Document Types
The pipeline accepts documents in any combination of types and processes them in parallel at the intake stage using a thread pool. Documents are classified by type—صحيفة دعوى, مذكرة دفاع, مذكرة رد, حافظة مستندات, محضر جلسة, حكم تمهيدي, or غير محدد—and by party—المدعي, المدعى عليه, النيابة, المحكمة, خبير, or غير محدد. When multiple defendants are present, party labels are disambiguated using Arabic ordinal suffixes with gender agreement, and a party manifest mapping each party to their submitted document types is passed through all downstream stages.

### 4.2 Summarization Workflow

#### 4.2.1 Role Classification
Each document chunk is classified into one of seven legal roles: الوقائع, الطلبات, الدفوع, المستندات, الأساس القانوني, الإجراءات, or غير محدد. Classification uses the party and document type as context to guide the LLM call.

#### 4.2.2 Bullet-Point Extraction
Each classified chunk is converted into atomic legal bullet points, with each bullet representing a single legal idea. Role-specific instructions guide the extraction: for الدفوع, for example, each distinct defense and its legal basis must be captured separately. A strict prohibition prevents adding any number, date, or amount not present verbatim in the source text.

#### 4.2.3 Aggregation
Bullets are grouped by legal role across all documents and classified as متفق عليه (undisputed), محل النزاع (disputed), or خاص بطرف (party-specific). Expert report findings are not classified as undisputed unless explicitly accepted by all parties.

#### 4.2.4 Thematic Clustering and Paragraph Construction
Within each role group, bullets are organized into three to seven thematic sub-groups with suggested category names per role—for الدفوع, these include دفوع شكلية, دفوع موضوعية, دفوع بالتقادم, and دفوع بعدم القبول. Every bullet must be assigned to exactly one theme. Each theme is then rendered as a two-to-three-paragraph Arabic prose summary; every sentence must cite a specific bullet identifier, and disputed points must use the يتمسك/يدفع framing to represent both positions.

#### 4.2.5 Final Case Summary Structure
The output is a seven-section judge-facing brief: a dispute summary, undisputed facts (maximum one hundred words), key disputes, party requests, party defenses classified as شكلية, بعدم القبول, or موضوعية, submitted documents, and legal questions for the court. If a section has no source data, the brief states that no information is available; no content is generated without a traceable source.

### 4.3 Information Preservation and Significance-Aware Compression
Priority is given to core factual statements by parties, points of disagreement, dates, names, amounts, procedural actions, and legal context necessary to understand case flow. Repetitive wording is reduced, but substantively different information is preserved.

### 4.4 Neutrality, Contradictions, and Traceability
The system maintains judicial neutrality through three mechanisms. **Non-interpretation**: the system does not interpret legal arguments, assess credibility, or resolve conflicting claims; all language is descriptive. **Contradiction preservation**: when case files contain conflicting statements, each version is clearly presented with explicit source identification; no attempt is made to resolve or favor any position. **Full traceability**: each summary section can be traced to supporting bullet identifiers and source documents, with metadata tracking submission dates, procedural phase, and document type.

---

## 5. Conversational Judicial Assistant Methodology

### 5.1 Conversational Scope and Session Constraints
Each session is strictly linked to a single civil case. When a session begins, the system loads the structured case representation, including the global summary, indexed document segments, and identified entities. Session state is persisted using a MongoDB checkpointer so that conversations survive server restarts; each session is identified by a conversation identifier and a user identifier. Switching cases during an active session is prohibited.

### 5.2 Query Intake and Intent Classification
Judges submit questions in natural language. The system classifies intent into one of five categories: civil law retrieval, case document retrieval, analytical reasoning, multi-agent, or off-topic. The analytical reasoning intent dispatches queries to the Chat Reasoner for multi-step decomposition; the multi-agent intent triggers parallel dispatch to more than one agent. Classification includes a prompt injection defense: user content is enclosed in trust boundary markers, and the classifier is instructed to ignore any commands embedded within the query.

### 5.3 Conversational Rephrasing and Clarification Control
To improve retrieval accuracy, the system internally reformulates questions while preserving meaning, storing the rewritten version as the classified query passed to all dispatched agents. Vague references are resolved using conversation history and structured entity memory. Clarification is requested only when automatic resolution fails.

### 5.4 Context Enrichment
Before agents are dispatched, a context enrichment step pre-fetches the latest case summary and all document titles for the active case from MongoDB. This data is injected into every agent call as shared context, so no agent needs to perform redundant database lookups during execution.

### 5.5 Case-Aware Conversational Memory Architecture
The system maintains three levels of memory. Short-term memory consists of recent conversation turns stored as role/content pairs. A running summary compresses older turns into Arabic prose when the turn count since the last summary exceeds a threshold; this summary is injected as context but is never treated as a factual source. Long-term memory persists across sessions in two forms: semantic facts extracted from case materials keyed by case identifier, and procedural preferences extracted from past sessions and injected into the intent classification prompt. Memory is loaded at the start of each turn and written back after each successful response.

### 5.6 Controlled Response Structure and Content Rules
All responses must be fully grounded in retrieved sources with explicit citations for each sentence, using neutral, non-evaluative language. Answers are organized into two clearly separated sections: facts extracted from the case file, and relevant legal texts if applicable. Factual content and legal articles are never merged into a single analytical narrative.

### 5.7 Safety Controls and Prohibited Queries
The assistant is restricted from answering questions involving legal judgment or evaluation, including predicting case outcomes, assessing argument strength, evaluating witness credibility, and interpreting or applying legal rules. Misuse detection also covers prompt injection patterns such as instructions to act as a different system, ignore prior context, or override operational rules; these are detected at classification and routed as off-topic. When prohibited queries are detected, the system returns a standardized refusal response.

---

## 6. Execution Control and Orchestration Methodology

This chapter describes how the system coordinates internal components to ensure correctness, neutrality, and compliance with judicial constraints.

### 6.1 Rationale for Controlled Orchestration
Judicial queries vary in complexity—some require simple fact retrieval, others involve summarization, document comparison, or statutory lookup. A fixed execution sequence is insufficient. The methodology requires enforcing judicial constraints before producing output, verifying grounded generation, and handling failures safely. An orchestration layer manages component activation and result combination.

### 6.2 Central Supervisory Control
A supervisory component governs every conversational request. Its responsibilities include validating input, loading long-term memory, classifying intent, pre-fetching case context, selecting appropriate internal components, enforcing scope limitations and prohibited-query rules, verifying citations, validating generated responses, and blocking outputs that violate methodological constraints. Each turn is assigned a correlation identifier used to link log entries across all nodes. The supervisor coordinates execution but does not generate answers directly.

### 6.3 Specialized Processing Components
The system employs three specialized adapters with restricted roles: a case document retrieval adapter that delegates to the case document MCP server; a civil law retrieval adapter that delegates to the legal RAG MCP server; and a chat reasoner adapter that invokes the analytical reasoning graph directly without MCP. Each adapter returns a standardized result structure containing the response text, source references, and raw output. Outputs are returned to the supervisor for citation verification and validation.

### 6.4 Chat Reasoner
The Chat Reasoner handles queries classified as analytical reasoning, which require decomposing a question into multiple steps across different information sources. The planner produces an execution plan in which each step specifies a tool, a query, and a list of prerequisite steps; steps without prerequisites execute in parallel. A plan validator checks that all tools are recognized, all queries are non-empty, all dependency references are valid, and the plan does not exceed five steps; invalid plans are returned to the planner for revision. Each step calls one of three tools: case document retrieval, civil law retrieval, or case summary fetch. If the synthesizer determines results are insufficient, a replanner rewrites the plan for up to two additional rounds before producing a final response. All traces are recorded in MongoDB.

### 6.5 Validation Before Response Delivery
All responses undergo two sequential validation stages. First, a citation verification step checks citation integrity at the raw output level before merging. Second, the output validator checks four independent criteria: absence of hallucination, relevance to the query, completeness, and coherence with prior turns. When hallucination, relevance, and coherence all pass but completeness fails, the response is classified as a partial pass and delivered with an automatically appended disclosure caveat rather than triggering a retry.

### 6.6 Error Handling and Safe Failure Behavior
Errors are handled in two categories: absence of information in the case file or legal corpus, and technical failures in retrieval or processing. A validator error is a distinct status that triggers a retry rather than an immediate fallback; retries continue until the maximum retry count is reached, at which point the system delivers a standardized fallback response. The system does not guess missing information or compensate through inference.

### 6.7 Audit Logging
Every turn produces an audit log record written by a dedicated node at the end of the workflow. Each record includes the request type, the data sources accessed, validation outcomes, and any citation issues identified. Logs are append-only and accessible only to authorized administrators.

---

## 7. Trust, Accuracy, and Compliance Mechanisms

This chapter describes operational rules and safeguards that ensure reliable, compliant system behavior. These mechanisms keep the assistant a neutral support tool and prevent evolution into an authoritative or decision-making system.

### 7.1 Judicial Neutrality and Output Constraints
The system restricts outputs to descriptive, source-based presentation. It organizes and presents case facts and statutory references without interpretation, evaluation, or persuasive framing. A fabrication ban is enforced at every pipeline stage, including bullet extraction, theme synthesis, and case brief generation, where any number, date, or detail not present verbatim in the source triggers a prompt-level refusal.

### 7.2 Accuracy and Faithfulness Enforcement
The methodology prioritizes faithfulness to original documents over stylistic quality or completeness. All outputs must originate from case documents or the legal corpus, reflect sources accurately, and exclude inference and extrapolation.

### 7.3 Judicial Ethics and Legal Compliance Controls
Judicial ethics are enforced through system design. The assistant is constrained to operate as a neutral informational tool, excluding legal reasoning, evaluation, or judgment. Legal interpretation and decision-making authority remain exclusively with the judge.

### 7.4 Operational Governance
**Auditability and logging.** Internal logs record request types, accessed data sources, validation outcomes, and missing or weak citations. Logs are append-only and accessible only to authorized administrators, supporting institutional oversight without exposing internal behavior to judges.

**Misuse detection.** The system monitors for attempts to bypass judicial restrictions, reformulate prohibited queries, or inject instructions to override rules, including prompt injection patterns detected at the classifier level. When detected, the system refuses to respond and records the event for administrative review.

**Rate limiting and session isolation.** Request rate limiting is enforced through Redis at one hundred requests per sixty seconds per user. Each conversation is bound to exactly one case identifier; switching cases during an active session is prohibited by the session management layer.

**Data governance.** Updates to case documents, legal corpora, and retrieval indices are logged and traceable. The system distinguishes data-related errors from system-level errors, enabling appropriate corrective action.

---

## 8. Methodological Evaluation Criteria

This chapter describes how the system is evaluated to verify behavior as a neutral judicial support tool. Evaluation focuses on system behavior, accuracy, and compliance with methodological constraints—not on legal correctness or decision quality.

### 8.1 Evaluation Scope and Principles
The system is evaluated as an assistive tool, not a legal authority. Tests are organized into five categories: unit tests verify pure routing and state logic without LLM calls; integration tests verify API contracts and infrastructure connectivity; behavioral tests verify end-to-end accuracy against golden datasets; performance tests verify resource bounds; and regression tests guard against recurrence of previously fixed defects. LLM-dependent tests are separated under an explicit marker and skipped when API credentials are unavailable. Legal reasoning, judgment quality, and case outcomes are outside evaluation scope.

### 8.2 Groundedness and Hallucination Evaluation
All outputs must be grounded in retrieved source documents, with no statement generated without an identifiable source. For the summarization pipeline, source traceability is verified by checking that every citation in the rendered brief references a document identifier present in the input set; any fabricated citation triggers an immediate test failure (EV-03). For the case reasoner, at least eighty percent of applied legal elements must cite a specific article (CR-EV-02), and overall factual faithfulness is scored by an LLM judge against the retrieved facts with a passing threshold of eleven out of fifteen (CR-EV-10).

### 8.3 Citation Quality Evaluation
Evaluation checks that responses include references to source documents or legal articles, and that citations point to correct material. Citation quality is reviewed qualitatively; minor weaknesses may be flagged without invalidating system behavior.

### 8.4 Large-Scale Summarization Evaluation
Summarization quality is evaluated using eight structured dimensions (EV-01 through EV-08) applied to pipeline output on a set of seven Arabic fixture documents spanning all party roles and document types. Structural completeness requires all seven case brief sections to be non-empty and all seven Arabic ordinal headings to appear in the rendered document (EV-01). Bullet coverage, scored by an LLM judge on a sampled subset of extracted bullets, must reach at least eighty percent (EV-02). Linguistic quality is scored by an LLM judge on a ten-point scale with a passing threshold of seven (EV-05), and factual faithfulness on a fifteen-point scale with a passing threshold of eleven (EV-06). Neutrality is checked by verifying the absence of bias keywords and by confirming that both plaintiff and defendant are present in the brief; verb framing ratios are monitored as a soft warning rather than a hard assertion (EV-04). Multi-party balance requires all fixture parties to appear in the brief (EV-07). Pipeline timing for seven documents is monitored against a threshold of one hundred and twenty seconds; exceeding this is recorded as an expected failure rather than a hard test failure (EV-08).

### 8.5 Conversational Evaluation
Multi-turn test sessions verify that turn count increments correctly across three or more consecutive turns, that conversation history carries forward to subsequent turns, and that history is trimmed to the configured maximum once the limit is exceeded. Retry count is verified to reset to zero at the start of each new turn rather than carrying over from the previous one. Each turn is also verified to receive a unique correlation identifier. The system must retrieve sources for each response regardless of conversational history.

### 8.6 Constraint Enforcement and Refusal Evaluation
Testing against out-of-scope queries confirms consistent refusal of prohibited queries, standardized neutral refusal format, and no attempt to bypass restrictions. Adversarial tests submit Arabic prompt injection attempts and verify that the classifier routes them to the off-topic path without producing an uncontrolled response. Input boundary tests confirm that empty queries are rejected with a validation error, that queries exceeding ten thousand characters do not produce server errors, and that non-Arabic input and injection-style strings are handled without crashing.

### 8.7 Performance and Scalability Evaluation
Memory stability is evaluated by running the pipeline thirty-five times in sequence and verifying that peak memory remains below five hundred megabytes and that total memory growth does not exceed 1.2 times the baseline measured after the first run. Cache effectiveness is verified by confirming that a warm cached query completes in at most half the wall-clock time of the corresponding cold query. Performance is evaluated for stability and bounded resource usage, not absolute speed.

### 8.8 Robustness and Failure Handling Evaluation
Failure scenario tests cover all-agent failure, database outage during a turn, LLM timeout, partial agent failure where only some agents in a multi-agent dispatch fail, and validator instability across repeated retries. Each scenario is verified to produce a standardized safe response rather than a crash or uncontrolled output. Regression tests additionally guard against RAG cross-case contamination, synchronous blocking in async paths, and incorrect structured output field shapes.

### 8.9 Human Expert Review
Human experts, including judges, conduct structured reviews of clarity of legal language, neutral and non-persuasive tone, and practical usefulness during case preparation. Judges assess usability without influencing system design.

### 8.10 Intent Classification Accuracy
Intent classification accuracy is evaluated against a curated golden dataset of labeled routing cases. The overall accuracy threshold is eighty-five percent; accuracy is also reported per agent category to identify systematic misclassification patterns. A query is counted as correct if either the classified intent or the target agent list matches the expected label.

### 8.11 RAG Retrieval Quality
Retrieval quality for the civil law RAG agent is measured using RAGAS metrics against a golden dataset: faithfulness must reach at least 0.80 and context recall at least 0.75. Answer consistency is evaluated separately by running five representative Arabic legal queries three times each and verifying that the pairwise cosine similarity of the resulting answers, computed using the BAAI/bge-m3 embedding model, is at least 0.80. A regression test additionally verifies that document-filtered queries return results only from the specified document with no cross-case contamination.

### 8.12 Case Reasoner Evaluation
Case reasoner quality is evaluated using twelve structured dimensions (CR-EV-01 through CR-EV-12) applied to a golden set prepared by a legal domain expert. Branch coverage requires all extracted issues to complete with a passed validation status (CR-EV-01). Confidence level is verified to match the expert-labeled expected level from the golden set (CR-EV-04). Reconciliation is verified to trigger if and only if cross-issue conflicts are expected (CR-EV-06), and the number of report sections is verified against the expected count (CR-EV-07). LLM-judge dimensions evaluate Arabic register quality with a threshold of seven out of ten (CR-EV-08), full neutrality of the counterargument section (CR-EV-09), factual faithfulness with a threshold of eleven out of fifteen (CR-EV-10), counterargument balance with a threshold of seven out of ten (CR-EV-11), and reconciliation clarity with a threshold of seven out of ten (CR-EV-12).

### 8.13 Chat Reasoner Evaluation
Chat Reasoner quality is evaluated through six end-to-end scenarios run against a seeded case involving a forged apartment sale contract, with all seven Arabic fixture documents indexed under a fixed case identifier. Each scenario verifies that the synthesizer marks the result as sufficient, that the execution plan includes the expected tools, and that the Arabic response meets a minimum character count. Scenarios cover comparison between forensic and engineering expert reports, civil law cross-referencing grounded in case facts, multi-party defense analysis covering both defendants, legal consequence reasoning from forensic findings, integration of a pre-stored case summary via the fetch-summary tool, and detection of contradictions between a party's defense and forensic evidence.

### 8.14 API and Integration Testing
A health endpoint verifies connectivity to all five infrastructure dependencies—MongoDB, Qdrant, Redis, MinIO, and PostgreSQL—and is evaluated as part of deployment verification. The query endpoint is verified to return a server-sent event stream content type, to reject unauthenticated requests, and to never return a server error across a set of ten valid Arabic queries. The OCR correction workflow is tested end-to-end from document upload through OCR output, manual correction, re-indexing, and retrieval verification.

---

## 9. Case Reasoner: Issue-Based Legal Analysis

### 9.1 Purpose
The Case Reasoner is a standalone pipeline that accepts a structured case brief and generates a comprehensive legal analysis report. It identifies distinct legal issues in the brief and analyzes each independently, producing issue-level reasoning that can be reviewed separately before a unified report is assembled.

### 9.2 Workflow
The main graph begins by extracting distinct legal issues from the case brief. Each issue is dispatched to a parallel analysis branch; all branches execute simultaneously and their results are collected before aggregation proceeds. After all branches complete, a consistency check identifies cross-issue relationships and detects logical conflicts, followed by per-issue and case-level confidence scoring and final report generation. When no issues are identified, the graph routes directly to an empty report node.

### 9.3 Issue Analysis Branch
Each branch receives an issue identifier, title, legal domain, and the exact source text excerpt from the brief. The branch begins by decomposing the issue into its required legal elements—for a contract breach claim, these include وجود العقد, الإخلال, الضرر, and العلاقة السببية. Retrieval queries are generated per element and submitted separately to the civil law retrieval and case document retrieval agents. Each element is then classified as established, not established, disputed, or insufficient in evidence. Legal reasoning prose is generated in Arabic citing specific articles, followed by analysis of the arguments available to each party. A validation step checks citation integrity, logical consistency, and completeness before the branch result is packaged.

### 9.4 Aggregation and Consistency
After all issue branches complete, the aggregation stage identifies relationships between issues and detects logical conflicts in their outcomes, generating reconciliation text where needed. Confidence scores are computed at the issue level and at the case level before the final report is generated. The completed report is stored in MongoDB and accessible through a dedicated API endpoint.

---

## 10. Report Generation Pipeline

### 10.1 Purpose
The report pipeline combines summarization and case reasoning into a single asynchronous job, producing a complete pre-hearing package from one API request. The request returns a job identifier immediately; the pipeline executes in the background.

### 10.2 Workflow
The background task runs the summarization pipeline followed by the case reasoning pipeline. Results from both stages are upserted to MongoDB under the case identifier. The job status can be polled through a report retrieval endpoint, which returns one of four states: pending, running, completed, or failed. When completed, the response includes both the case summary and the case reasoning report.

### 10.3 Standalone Endpoints
Summary generation, case reasoning retrieval, and structured brief retrieval are each available as independent endpoints. This allows judges or administrators to request only the component needed—for example, retrieving a previously generated case brief without triggering a full report regeneration.

---

## 11. System Architecture and Infrastructure

### 11.1 Infrastructure Components
The system depends on six infrastructure services. MongoDB stores conversations, documents, case summaries, case reasoning reports, and audit logs. Qdrant stores vector indexes for both case documents and the legal corpora. Redis provides the semantic cache for RAG queries, request rate limiting, and session data. MinIO stores uploaded binary files. PostgreSQL stores user accounts, roles, and the audit trail. Two TEI service instances provide remote text embeddings using the BAAI/bge-m3 model and reranking, respectively.

### 11.2 API Design
The API is implemented with FastAPI and uses an asynchronous MongoDB driver. All endpoints except the health check require a JWT bearer token carrying a user identifier. The query endpoint delivers responses as a server-sent event stream with four event types: progress, result, error, and done. File uploads are accepted up to twenty megabytes for PDF documents and common image formats.

### 11.3 LLM Tier Strategy
Two LLM tiers are used to align model capability with task requirements. The high tier, using Gemini 2.5 Flash, handles summarization synthesis, case reasoning, RAG answer generation, and supervisor output validation. The low tier, using Gemini 2.5 Flash-Lite, handles document intake classification, role classification, thematic clustering, and routing decisions. Both tiers operate at zero temperature to maximize output determinism.
