# Bro Architecture

Bro is a local-first Streamlit application organized around a central user interface, shared runtime configuration, local model execution, local document retrieval, and SQLite-backed persistence.

## 🧭 Purpose

This page explains how Bro's application layers interact. It is intended for developers, maintainers, and analysts who need to understand where state is stored, how document context reaches the model, and how each mode participates in the overall workflow.

## 🧱 Architectural Layers

| Layer | Primary Responsibility |
| --- | --- |
| Streamlit UI | Renders sidebar mode selection, controls, tabs, forms, chat messages, tables, and visualizations. |
| Session State | Preserves runtime values such as selected mode, prompts, model parameters, active documents, retrieval settings, and semantic-search results. |
| Configuration | Defines application paths, constants, default model settings, mode names, labels, regex patterns, and logging paths. |
| Local Model Runtime | Loads the configured GGUF model through `llama-cpp-python` and executes prompt turns. |
| Document Processing | Extracts PDF or text content, chunks text, computes fingerprints, and prepares retrieved excerpts. |
| Embedding Runtime | Loads sentence-transformer embeddings and converts chunks or queries into vector representations. |
| Retrieval Storage | Uses `sqlite-vec` when available and cosine-similarity fallback when the extension is unavailable. |
| SQLite Persistence | Stores chat history, prompts, embeddings, document metadata, chunks, embedding metadata, and image metadata. |
| Error Logging | Wraps exceptions with `Error` and persists diagnostic records through `Logger`. |

## 🏛 System View

```text
┌─────────────────────────────────────────────────────────────┐
│                         Streamlit UI                         │
│  Sidebar mode selector, controls, tabs, chat, tables, charts │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                       Session State                          │
│  messages, prompts, model settings, active docs, retrieval   │
│  settings, semantic rows, data-management selections          │
└───────┬───────────────┬─────────────────┬───────────────────┘
        │               │                 │
        ▼               ▼                 ▼
┌──────────────┐ ┌────────────────┐ ┌────────────────────────┐
│ Text Runtime │ │ Document Q&A   │ │ Semantic Search         │
│ llama.cpp    │ │ extraction +   │ │ chunking + embeddings   │
│ GGUF model   │ │ retrieval      │ │ ranked vector results   │
└──────┬───────┘ └───────┬────────┘ └──────────┬─────────────┘
       │                 │                     │
       └──────────────┬──┴──────────────┬──────┘
                      ▼                 ▼
           ┌──────────────────┐ ┌──────────────────┐
           │ SQLite Database  │ │ sqlite-vec /     │
           │ chat, prompts,   │ │ cosine fallback  │
           │ assets, chunks   │ │ retrieval        │
           └────────┬─────────┘ └──────────────────┘
                    │
                    ▼
           ┌──────────────────┐
           │ Error Logging    │
           │ Error + Logger   │
           └──────────────────┘
```

## 🔄 Runtime Flow

### Text Generation

```text
User input
  ▼
System instructions + task controls
  ▼
Optional chat history and document context
  ▼
Prompt construction
  ▼
llama.cpp local model call
  ▼
Response display and optional chat-history persistence
```

### Document Q&A

```text
Uploaded document bytes
  ▼
Text extraction
  ▼
Chunking
  ▼
Embedding
  ▼
sqlite-vec table or fallback vector rows
  ▼
Top-k retrieval
  ▼
Grounded prompt construction
  ▼
Local model answer
```

### Semantic Search

```text
Uploaded files
  ▼
Text extraction
  ▼
Chunking
  ▼
Embedding
  ▼
SQLite embeddings table
  ▼
Query embedding
  ▼
Cosine similarity ranking
  ▼
Selectable context rows
```

### Prompt Engineering

```text
Prompt table
  ▼
Search, sort, page, edit, clone, generate, apply
  ▼
Shared system instructions and task metadata
  ▼
Text Generation or Document Q&A
```

### Data Management

```text
SQLite database
  ▼
Tables, schemas, rows, profiles, filters, aggregations
  ▼
CRUD, visualization, read-only SQL, AI asset registration
```

## 🧩 Core Modules

| Module | Role |
| --- | --- |
| `app.py` | Main Streamlit UI, runtime orchestration, LLM utilities, retrieval utilities, prompt utilities, and SQLite operations. |
| `config.py` | Configuration constants, environment helpers, paths, labels, modes, and descriptive text. |
| `boogr.py` | Exception wrapper and SQLite-backed logging utility. |

## 🧠 State Model

Bro relies heavily on `st.session_state`. Session-state keys support:

| State Group | Examples |
| --- | --- |
| Mode and UI state | `mode`, selected tabs, prompt-selection fields. |
| Chat state | `messages`, `system_instructions`, chat history. |
| Runtime controls | `context_window`, `cpu_threads`, `max_tokens`, `temperature`, `top_percent`, `top_k`. |
| Text-generation presets | `task_preset`, `response_format`, `reasoning_depth`, coding and translation controls. |
| Document Q&A | `active_docs`, `doc_bytes`, retrieval controls, chunk counts, diagnostics, fallback rows. |
| Semantic Search | chunk size, overlap, top-k, threshold, indexed document count, result rows. |
| Prompt Engineering | category, task, style, generated template, selected prompt fields. |
| Data Management | selected asset table, import flags, asset sync status, asset counts. |

## ✅ Design Constraints

| Constraint | Rationale |
| --- | --- |
| Preserve local-first execution | Bro is intended to operate with local files, local model runtime, and local SQLite persistence. |
| Preserve safe fallback behavior | Optional components such as `sqlite-vec`, PyMuPDF, and embedding models may be unavailable. |
| Preserve session-state names | Streamlit mode transitions and tab workflows depend on stable keys. |
| Keep API pages source-driven | MkDocs should reflect the source docstrings rather than duplicating manual prose. |
| Log without masking failures | Logging failures should not obscure the original application failure. |
