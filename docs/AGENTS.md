# Agents

This document describes the five specialist agents and the supervisor that orchestrates them. All agents implement the `AgentAdapter` interface defined in [`Supervisor/agents/base.py`](../Supervisor/agents/base.py).

## Agent Adapter Interface

Every adapter must implement a single method:

```python
def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult
```

**Parameters:**
- `query` -- The (possibly rewritten) judge query
- `context` -- Additional data from `SupervisorState`:
  - `uploaded_files` -- List of file paths
  - `case_id` -- Active case identifier
  - `conversation_history` -- Previous turns
  - `agent_results` -- Results from agents that ran earlier in the same turn
  - `validation_feedback` -- Feedback from failed validation (on retries)

**Returns:** An [`AgentResult`](../Supervisor/agents/base.py) with fields:
- `response: str` -- Main textual output
- `sources: List[str]` -- Citations or references
- `raw_output: Dict[str, Any]` -- Full agent state for validation
- `error: Optional[str]` -- Error message if invocation failed

## Adapter Registry

The mapping from canonical agent names to adapter classes is defined in [`Supervisor/nodes/dispatch_agents.py`](../Supervisor/nodes/dispatch_agents.py:24):

```python
ADAPTER_REGISTRY: Dict[str, type] = {
    "ocr": OCRAdapter,
    "summarize": SummarizeAdapter,
    "civil_law_rag": CivilLawRAGAdapter,
    "case_doc_rag": CaseDocRAGAdapter,
    "reason": CaseReasonerAdapter,
}
```

---

## 1. OCR Agent

| Property | Value |
|---|---|
| **Adapter** | `Supervisor/agents/ocr_adapter.py` -> `OCRAdapter` |
| **Pipeline** | `OCR/` |
| **Trigger Intent** | `ocr` |
| **LLM Tier** | None (uses Surya OCR engine directly) |

### Purpose

Extracts text from scanned legal documents (PDF, PNG, JPEG, TIFF, BMP) using the Surya OCR engine with Arabic-specific preprocessing.

### Trigger Conditions

The `classify_intent` node routes to OCR when:
- The judge mentions uploading an image or PDF and asks to extract text
- Keywords: "استخرج النص", "حوّل الصورة إلى نص", "اقرأ المستند المرفق"
- No legal analysis is requested, only text extraction

### Pipeline Details

The OCR pipeline applies:
1. Resolution check (minimum 150 DPI)
2. Deskewing
3. Border removal
4. Contrast enhancement
5. Surya OCR text extraction (Arabic language, batch size 4)
6. Confidence scoring (high > 0.85, medium > 0.60)
7. Dictionary-based post-processing (legal Arabic dictionary at `OCR/dictionaries/legal_arabic.txt`, max Levenshtein distance 2)
8. Arabic-Indic digit normalization

### Input/Output

- **Input**: File paths from `context["uploaded_files"]`
- **Output**: Extracted text per file, with confidence scores

---

## 2. Summarize Agent

| Property | Value |
|---|---|
| **Adapter** | `Supervisor/agents/summarize_adapter.py` -> `SummarizeAdapter` |
| **Pipeline** | `Summerize/` (7-node LangGraph pipeline) |
| **Trigger Intent** | `summarize` |
| **LLM Tier** | Uses its own LLM via `config.get_llm()` |

### Purpose

Generates structured Arabic case briefs from legal documents through a multi-node pipeline with semantic classification and cross-party analysis.

### Trigger Conditions

- Judge requests a summary, overview, or key points
- Keywords: "ملخص", "موجز", "نقاط رئيسية"
- Not requesting legal analysis or specific law articles

### Document Resolution Priority

The adapter resolves input documents in this order (see [`summarize_adapter.py`](../Supervisor/agents/summarize_adapter.py:59)):

1. `context["documents"]` -- Pre-built `{raw_text, doc_id}` dicts
2. `context["agent_results"]["ocr"]` -- OCR output from earlier this turn
3. `context["uploaded_files"]` -- File paths; read from disk
4. **MongoDB fallback** -- Fetch by `context["case_id"]` from the `Document Storage` collection

### MongoDB Fallback Details

When fetching from MongoDB, the adapter uses these field names (from [`summarize_adapter.py`](../Supervisor/agents/summarize_adapter.py:36)):

```python
raw_text = doc.get("text", "")
doc_id = str(doc.get("title") or doc.get("source_file") or doc.get("_id", "unknown"))
```

> **Warning**: If the MongoDB document schema changes these field names (`text`, `title`, `source_file`), the summarizer fallback will silently return empty documents.

### Pipeline Nodes

| Node | Schema | Description |
|---|---|---|
| Node 0 | `Node0Output` -> `List[NormalizedChunk]` | Text normalization, chunking, metadata extraction |
| Node 1 | `Node1Output` -> `List[ClassifiedChunk]` | Semantic role classification (facts, requests, defenses, evidence, legal basis, procedures) |
| Node 2 | `Node2Output` -> `List[LegalBullet]` | Atomic legal bullet extraction with traceability |
| Node 3 | `Node3Output` -> `List[RoleAggregation]` | Cross-party aggregation: agreed, disputed, party-specific |
| Node 4A | `Node4AOutput` -> `List[ThemedRole]` | Thematic clustering (3-7 themes per role) |
| Node 4B | -- | Theme-level synthesis |
| Node 5 | -- | Final Arabic case brief generation |

---

## 3. Civil Law RAG Agent

| Property | Value |
|---|---|
| **Adapter** | `Supervisor/agents/civil_law_rag_adapter.py` -> `CivilLawRAGAdapter` |
| **Pipeline** | `RAG/Civil Law RAG/` (LangGraph state machine) |
| **Trigger Intent** | `civil_law_rag` |
| **LLM Tier** | Uses its own LLM via RAG config |

### Purpose

Retrieves and answers questions about Egyptian civil law articles using retrieval-augmented generation.

### Trigger Conditions

- Judge asks about a specific law article or legal provision
- Keywords: "مادة قانونية", "نص مادة", "حكم قانوني"
- Question is about general civil law, NOT about specific case documents

### Retrieval Strategy

| Property | Value |
|---|---|
| Qdrant collection | `judicial_docs` |
| Embedding model | `BAAI/bge-m3` |
| Vector dimension | 1024 |
| Distance metric | COSINE |
| Search type | MMR (Maximal Marginal Relevance) |
| Top-k | 8 |

### sys.path Manipulation

The adapter performs `sys.path` and `sys.modules` manipulation to import the RAG pipeline (see [`civil_law_rag_adapter.py`](../Supervisor/agents/civil_law_rag_adapter.py:20)):

```python
_RAG_MODULES = ("config", "graph", "nodes", "state", "edges", "utils")
```

Before each import, it:
1. Puts the RAG directory at the front of `sys.path`
2. Evicts cached versions of modules with conflicting names (e.g., `config`, `graph`)
3. Imports `graph.app` and `config.rag.default_state_template` fresh

> **Known issue**: This `sys.modules` cache eviction can cause subtle bugs if other code holds references to the evicted modules. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Output

- `response` -- The generated answer about civil law
- `sources` -- Article references extracted from document metadata
- `raw_output` -- Includes `final_answer`, `classification`, `retrieval_confidence`

---

## 4. Case Document RAG Agent

| Property | Value |
|---|---|
| **Adapter** | `Supervisor/agents/case_doc_rag_adapter.py` -> `CaseDocRAGAdapter` |
| **Pipeline** | `RAG/Case Doc RAG/` |
| **Trigger Intent** | `case_doc_rag` |
| **LLM Tier** | Uses its own LLM configuration |

### Purpose

Retrieves information from case-specific documents that have been ingested and embedded into the vector store.

### Trigger Conditions

- Judge asks about a specific document within the case file
- Keywords: "عقد", "مذكرة", "تقرير خبير", "إنذار", "إيصال"
- References to specific case documents, NOT general law

### Retrieval Strategy

| Property | Value |
|---|---|
| Qdrant collection | `case_docs` |
| Embedding model | `BAAI/bge-m3` |
| Vector dimension | 1024 |
| Distance metric | COSINE |
| Search type | MMR |
| Top-k | 8 |
| Payload filter | `case_id` (filters vectors to the active case) |

---

## 5. Case Reasoner Agent

| Property | Value |
|---|---|
| **Adapter** | `Supervisor/agents/case_reasoner_adapter.py` -> `CaseReasonerAdapter` |
| **Pipeline** | `Case Reasoner/` |
| **Trigger Intent** | `reason` |
| **LLM Tier** | High tier (complex legal reasoning) |

### Purpose

Applies judicial reasoning to case facts against relevant law articles. Handles legal qualification (تكييف قانوني), liability analysis, and defense evaluation.

### Trigger Conditions

- Judge requests legal analysis or qualification
- Keywords: "تكييف قانوني", "أركان المسؤولية", "ترجيح دفوع"
- Asking for an opinion based on case facts

---

## Supervisor Orchestration

### Intent Classification

The `classify_intent` node (at [`Supervisor/nodes/classify_intent.py`](../Supervisor/nodes/classify_intent.py)) uses:
- **LLM tier**: `medium`
- **Structured output**: `IntentClassification` Pydantic model
- **Prompt language**: Arabic (from [`Supervisor/prompts.py`](../Supervisor/prompts.py))

The classification prompt defines all 7 intents with detailed Arabic descriptions of when each should be used, including positive and negative examples.

### Valid Intents

From [`config/supervisor.py`](../config/supervisor.py:43):

```python
VALID_INTENTS = AGENT_NAMES + ["multi", "off_topic"]
# = ["ocr", "summarize", "civil_law_rag", "case_doc_rag", "reason", "multi", "off_topic"]
```

### Multi-Agent Dispatch

When intent is `multi`, multiple agents in `target_agents` are invoked **sequentially** (not in parallel). Each agent receives results from previously-run agents via `context["agent_results"]`, enabling chained workflows like OCR -> Summarize.

### Retry Mechanism

On validation failure:
1. `retry_count` is incremented
2. `validation_feedback` is appended to the query on the next dispatch
3. Agents are re-invoked with the enriched query
4. Maximum retries: 3 (from `config/settings.yaml`, line 107)
5. If retries are exhausted, the `fallback_response` node generates a graceful degradation message
