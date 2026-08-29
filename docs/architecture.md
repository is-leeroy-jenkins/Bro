# Architecture

Bro uses a local-first architecture in which Streamlit owns interaction state, SQLite owns local
persistence, sentence-transformers provides local embeddings, and llama.cpp executes Gemma 3 text
and multimodal inference.

## Component topology

```text
Streamlit UI
    |
    +-- Text Generation
    |     +-- task instruction builder
    |     +-- system instructions
    |     +-- context/message builder
    |     +-- inference controls
    |     +-- runtime controls
    |     +-- Llama.create_chat_completion()
    |     +-- Gemma 3 4B IT GGUF
    |
    +-- Image to Text
    |     +-- image uploader
    |     +-- vision instruction builder
    |     +-- MTMDChatHandler
    |     +-- mmproj GGUF
    |     +-- Gemma 3 4B IT GGUF
    |     +-- text result
    |
    +-- Document Q&A
    |     +-- PDF/TXT/DOCX parsing
    |     +-- optional Gemma vision OCR
    |     +-- chunking
    |     +-- sentence-transformers
    |     +-- sqlite-vec or cosine fallback
    |     +-- grounding policy
    |     +-- shared text-generation path
    |
    +-- Semantic Search
    |     +-- document-aware embeddings table
    |     +-- similarity ranking
    |     +-- optional group-by-document
    |     +-- context routing
    |
    +-- Prompt Engineering
    |     +-- Prompts(ID, Caption, Name, Category, Text)
    |     +-- capability-filtered category-aware selectors
    |     +-- prompt application metadata
    |
    +-- Data Management
          +-- SQLite CRUD
          +-- Excel import
          +-- profile/filter/aggregate/visualize
          +-- schema/index administration
          +-- AI asset governance
```

## Text-generation request flow

1. A user selects a bounded **Task Type**.
2. Task-specific controls are read from `st.session_state`.
3. `build_task_instruction_block()` converts task and response selections into model instructions.
4. `build_chat_messages()` adds system instructions, history, document context, and semantic context.
5. Runtime controls resolve the cached llama.cpp model configuration.
6. Inference controls are passed to `create_chat_completion()`.
7. The result is streamed or rendered as text.
8. Chat history is persisted locally.

## Vision request flow

```text
uploaded image(s)
    |
    +-- build vision instruction
    |
    +-- image -> data URI / multimodal message content
    |
    +-- MTMDChatHandler + matching mmproj
    |
    +-- Gemma 3 4B IT
    |
    +-- text output
```

The multimodal path is capability-gated. A missing projector does not silently route an image into
the text-only model.

## Document OCR flow

When `Enable OCR` is active, Bro prefers native PDF text when configured to do so. A page with no
usable native text can be rendered to an image and routed through the same Gemma vision runtime used
by Image to Text.

```text
PDF page
  |
  +-- native text available? -- yes --> use text
  |
  no
  |
  +-- OCR enabled?
        |
        +-- multimodal runtime available?
              |
              +-- render page -> image -> Gemma vision -> extracted text
```

## State ownership

Controls are grouped by execution responsibility:

| Group | Function responsibility |
| --- | --- |
| Task Preset / task-specific controls | Prompt/instruction construction |
| Response Controls | Output format, language, length, headings |
| Context Controls | History, document context, semantic context, grounding, context window |
| Inference Settings | Sampling, penalties, seed, output-token limit |
| Runtime Settings | CPU threads, batch size, micro-batch size |
| Vision Runtime Settings | Context/runtime plus projector device |

Each expander owns one Reset action that resets the controls within that expander.

## Persistence

SQLite persists:

- `chat_history`;
- `Prompts`;
- `embeddings`;
- `documents`;
- `document_chunks`;
- `document_embeddings`;
- `images`;
- user-imported tables.

The `Prompts` table uses the authoritative schema documented in
[Prompt Engineering](prompt-engineering.md).
