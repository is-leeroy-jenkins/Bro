# App API

`app.py` is the executable Streamlit application and contains top-level UI code. This page therefore
documents its public functional areas **without importing `app` during the MkDocs build**.

## Prompt and option contracts

Key public helpers include:

- `get_prompt_categories()`
- `get_prompt_task_types()`
- `get_response_formats()`
- `get_spoken_languages()`
- `fetch_prompt_categories()`
- prompt CRUD/query helpers

## Text-generation pipeline

- `build_task_instruction_block()`
- `build_chat_messages()`
- `get_runtime_llm()`
- `run_llm_turn()`

## Vision pipeline

- multimodal-projector resolution/capability helpers
- vision instruction/message construction
- vision runtime loading
- `run_vision_turn()`

## Document Q&A

- extraction helpers
- chunking
- sqlite-vec loading/schema helpers
- index rebuild
- chunk retrieval
- document prompt construction
- document action routing

## Semantic Search

- semantic index construction
- embedding-row decoding
- semantic query/ranking
- group-by-document logic
- selected-context routing

## Prompt Engineering

- Category-aware prompt listing
- ID-based prompt selection
- insert/update/delete/clone
- prompt-application/cascade helpers

## Data Management

- table listing/reading
- schema inspection
- table/column administration
- indexes
- guarded SQL
- profiling/aggregation/visualization helpers

!!! note "Why this page does not use `::: app`"
    Importing `app.py` can execute Streamlit page construction and runtime initialization. Keeping
    the API page manual prevents MkDocs from turning documentation generation into an application
    launch.
