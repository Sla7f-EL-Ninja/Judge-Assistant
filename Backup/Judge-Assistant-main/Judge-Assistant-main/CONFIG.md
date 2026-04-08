# Configuration System

Judge Assistant uses a centralized configuration system based on a single YAML file. All modules read their settings from this file instead of maintaining independent config files.

## File Layout

```
config/
  __init__.py          # Loader, AppConfig class, get_llm() factory
  settings.yaml        # Default configuration (committed to repo)
  settings.local.yaml  # Local overrides (gitignored)
```

## How It Works

1. `config/settings.yaml` is loaded as the base configuration.
2. If `config/settings.local.yaml` exists, its values are deep-merged on top of the defaults.
3. Environment variables with the `JA_` prefix override any YAML value.

### Precedence (highest wins)

```
Environment variables (JA_*)  >  settings.local.yaml  >  settings.yaml
```

## Environment Variable Overrides

Nested YAML keys are joined with `_` and upper-cased, prefixed with `JA_`.

| YAML Path | Environment Variable |
|-----------|---------------------|
| `llm.high.model` | `JA_LLM_HIGH_MODEL` |
| `llm.low.provider` | `JA_LLM_LOW_PROVIDER` |
| `mongodb.uri` | `JA_MONGODB_URI` |
| `ocr.language` | `JA_OCR_LANGUAGE` |
| `api.debug` | `JA_API_DEBUG` |

Values are automatically cast to match the type of the YAML default (bool, int, float, str, list).

## API Keys

API keys are **not** stored in YAML. They stay in `.env` or the process environment and are read directly by the LangChain provider classes:

- `GROQ_API_KEY` -- required when any tier uses the `groq` provider
- `GOOGLE_API_KEY` -- required when any tier uses the `google` provider

## LLM Tier System

Models are organized into three tiers based on task complexity. Each tier maps to a provider and model that can be changed in a single place.

| Tier | Default Provider | Default Model | Use Cases |
|------|-----------------|---------------|-----------|
| **high** | groq | llama-3.3-70b-versatile | Legal reasoning, response merging, summarization, RAG answer generation |
| **medium** | groq | llama-3.3-70b-versatile | Intent classification, document classification, query rewriting |
| **low** | google | gemini-1.5-flash | Output validation, off-topic detection, simple routing |

### Using `get_llm()`

```python
from config import get_llm

# Get a model for the requested tier
llm = get_llm("high")
llm = get_llm("medium")
llm = get_llm("low")

# Override temperature for a specific call
llm = get_llm("high", temperature=0.3)
```

The function reads the tier config, instantiates the correct LangChain provider (`ChatGroq` or `ChatGoogleGenerativeAI`), and returns a ready-to-use chat model.

### Tier Assignments by File

| File | Task | Tier |
|------|------|------|
| `Supervisor/nodes/classify_intent.py` | Intent classification | medium |
| `Supervisor/nodes/validate_output.py` | Output validation | low |
| `Supervisor/nodes/merge_responses.py` | Multi-agent response synthesis | high |
| `Supervisor/agents/summarize_adapter.py` | Summarization pipeline | high |
| `Case Reasoner/case_reasoner.py` | Legal reasoning | high |
| `RAG/Case Doc RAG/rag_docs.py` | Document Q&A | high |
| `RAG/Case Doc RAG/document_classifier.py` | Document type classification | medium |
| `RAG/Civil Law RAG/nodes.py` | Query processing + answer gen | high |
| `Summerize/main.py` | Summarization pipeline | high |

## Accessing Other Config Sections

```python
from config import cfg

# MongoDB
cfg.mongodb["uri"]
cfg.mongodb["database"]

# Chroma vector store
cfg.chroma["collection"]
cfg.chroma["persist_dir"]

# Embedding model
cfg.embedding["model"]

# OCR settings
cfg.ocr["language"]
cfg.ocr["preprocessing"]["enable_deskew"]

# Supervisor settings
cfg.supervisor["max_retries"]
cfg.supervisor["agent_names"]

# API settings
cfg.api["app_name"]
cfg.api["jwt_algorithm"]
```

## Local Development

Create `config/settings.local.yaml` to override defaults without modifying the committed file:

```yaml
# config/settings.local.yaml -- not committed to git
llm:
  high:
    provider: google
    model: gemini-1.5-pro
mongodb:
  uri: mongodb+srv://user:pass@cluster.mongodb.net/
api:
  debug: true
```

## Module-Level Config Files

The old per-module config files (`Supervisor/config.py`, `OCR/config.py`, `RAG/Civil Law RAG/config.py`, `api/config.py`) still exist as thin wrappers that read from the central config. They export the same module-level constants for backward compatibility, so existing import patterns continue to work.
