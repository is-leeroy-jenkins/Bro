# Bro Application

## 🧭 Purpose

Bro is a local-first AI application that provides text generation, document Q&A, semantic search,
prompt engineering, and SQLite-backed data management through a Streamlit interface.

## 🧱 Application Structure

| Area               | Runtime Focus                  | Purpose                                                                                                                                        |
| ------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Text Generation    | Local LLM chat workflow        | Generates responses from user prompts using configurable inference controls, system instructions, chat history, and optional document context. |
| Document Q&A       | Grounded document workflow     | Extracts text from uploaded documents, chunks content, retrieves relevant excerpts, and routes grounded questions through the local model.     |
| Semantic Search    | Embedding-backed retrieval     | Builds a local semantic index from uploaded documents and returns ranked chunks based on similarity to a user query.                           |
| Prompt Engineering | Prompt-template workflow       | Creates, edits, applies, converts, and manages reusable prompt templates stored in SQLite.                                                     |
| Data Management    | SQLite administration workflow | Inspects, profiles, filters, modifies, and manages local application tables through guarded database utilities.                                |
| Logging            | Structured exception workflow  | Captures exception context through shared `Error` and `Logger` objects for troubleshooting and auditability.                                   |

## ⚙️ Runtime Responsibilities

| Responsibility        | Application Behavior                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local model execution | Loads the configured GGUF model through llama.cpp and executes generation turns with user-selected context, token, thread, and sampling settings. |
| Prompt construction   | Combines system instructions, task presets, chat history, semantic context, document context, and user input into a model-ready prompt.           |
| Chat persistence      | Stores and reloads chat history from SQLite so prior exchanges can be reused across application sessions.                                         |
| Document ingestion    | Accepts uploaded files, extracts text, computes fingerprints, chunks content, and tracks active document state.                                   |
| Grounded answering    | Retrieves relevant document chunks and instructs the model to answer from available excerpts when grounding controls are enabled.                 |
| Semantic indexing     | Encodes document chunks with a sentence-transformer model and stores vectors for similarity-based search.                                         |
| Prompt management     | Reads, writes, clones, updates, deletes, and applies prompt templates from the local `Prompts` table.                                             |
| Database inspection   | Lists tables, reads schema metadata, previews rows, creates indexes, profiles columns, and supports safe read-only SQL queries.                   |
| Asset registration    | Registers uploaded documents, chunks, embeddings, and images into local governance tables for traceability.                                       |
| Error handling        | Logs structured exception metadata including module, cause, method, message, and trace context.                                                   |

## 🧩 Core Workflows

| Workflow           | Input                                                            | Processing                                                                                             | Output                                                                    |
| ------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Text Generation    | User prompt, system instructions, inference settings             | Builds a local model prompt and executes a llama.cpp generation turn.                                  | Assistant response.                                                       |
| Document Q&A       | Uploaded documents and user question                             | Extracts text, chunks content, retrieves relevant excerpts, and routes a grounded prompt to the model. | Grounded answer, summary, outline, entities, key points, or comparison.   |
| Semantic Search    | Uploaded text-bearing files and search query                     | Chunks files, generates embeddings, scores similarity, and ranks matching chunks.                      | Ranked semantic search results and reusable context.                      |
| Prompt Engineering | Prompt goal, category, task type, format, and stored templates   | Builds or edits reusable prompt templates and applies selected templates to active workflows.          | Updated system instructions or saved prompt records.                      |
| Data Management    | SQLite table selection, filters, schema controls, or guarded SQL | Reads tables, profiles data, applies filters, creates indexes, and manages schema operations.          | Dataframes, metrics, profile tables, charts, or updated database objects. |

## 🧪 Recommended Operating Sequence

1. Start in Text Generation to confirm the local model path, runtime settings, and basic response
   behavior.
2. Use System Instructions to define the assistant role, response style, and task constraints.
3. Upload documents in Document Q&A when grounded answers, summaries, outlines, or extractions are
   required.
4. Build a Semantic Search index when reusable chunk retrieval or context selection is needed.
5. Use Prompt Engineering to save repeatable prompt templates for common workflows.
6. Use Data Management to inspect local tables, prompt records, document metadata, embeddings, and
   chat history.
7. Review logged exceptions when runtime behavior needs troubleshooting.

## ✅ Application Runtime Notes

| Component               | Role                                                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `streamlit`             | Provides the interactive web application interface.                                                       |
| `llama-cpp-python`      | Runs the configured local GGUF model.                                                                     |
| `sentence-transformers` | Generates embeddings for semantic search and document retrieval.                                          |
| `sqlite3`               | Stores chat history, prompt templates, embeddings, document metadata, chunk metadata, and image metadata. |
| `pandas`                | Supports table previews, filtering, profiling, and dataframe operations.                                  |
| `numpy`                 | Supports vector operations and cosine similarity scoring.                                                 |
| `plotly`                | Provides interactive charts for data-management workflows.                                                |
| `PyMuPDF`               | Extracts native text from PDF uploads when available.                                                     |

## 🔎 Operational Controls

| Control Area       | Purpose                                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------------- |
| Task Preset        | Selects the active task pattern, including chat, reasoning, coding, translation, summarization, or extraction. |
| Response Controls  | Adjusts temperature, top-p, top-k, and grounding behavior.                                                     |
| Inference Settings | Adjusts repeat window, repeat penalty, presence penalty, and frequency penalty.                                |
| Context Controls   | Adjusts context window, CPU threads, max tokens, and random seed.                                              |
| Document Controls  | Adjusts retrieval count, chunk size, chunk overlap, grounding behavior, and excerpt-only answering.            |
| Semantic Controls  | Adjusts semantic chunking, top-k results, minimum similarity, diagnostics, and index behavior.                 |
| Prompt Controls    | Manages prompt category, task type, format, language, generator goal, constraints, style, and saved templates. |
| Data Controls      | Manages selected tables, filters, aggregation, visualization, schema operations, and safe SQL access.          |

## 🔗 Related Pages

| Page                                           | Description                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------- |
| [Architecture](../architecture.md)             | Application layers, runtime flow, and module relationships.                     |
| [Text Generation](../text-generation.md)       | Local model prompting, inference settings, and response generation.             |
| [Document Q&A](../document-qna.md)             | Grounded document workflows, extraction, summarization, and retrieval behavior. |
| [Semantic Search](../semantic-search.md)       | Embedding-backed chunk search and semantic context reuse.                       |
| [Prompt Engineering](../prompt-engineering.md) | Prompt templates, metadata, conversion, and workflow routing.                   |
| [Data Management](../data-management.md)       | SQLite inspection, profiling, filtering, visualization, and administration.     |
| [Development](../development.md)               | Local setup, maintenance checks, and build workflow.                            |
