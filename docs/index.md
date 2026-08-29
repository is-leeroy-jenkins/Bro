
![](img/bro-project.png)

___

Bro is a local-llm application built around **Gemma 3 4B IT in GGUF format**. It
combines local text generation, Gemma multimodal Image-to-Text, document-grounded retrieval,
semantic search, prompt engineering, and SQLite-backed data management in a single application.

![Bro](https://github.com/is-leeroy-jenkins/Bro/blob/main/resources/images/bro_project.png)

## Application capabilities

| Capability | What Bro provides |
| --- | --- |
| Text generation | Chat, analysis, reasoning, coding, writing, editing, summarization, extraction, classification, translation, comparison, and structured output. |
| Image understanding | Visible-text extraction, image description, image Q&A, screenshot/chart/diagram analysis, structured image extraction, and image comparison. |
| Document Q&A | Local PDF/TXT/DOCX retrieval with grounding controls, sqlite-vec/cosine search, document actions, and optional Gemma vision OCR. |
| Semantic search | Local chunk embeddings, similarity search, document-aware results, and reusable context routing. |
| Prompt engineering | Category-aware SQLite prompt management using `ID`, `Caption`, `Name`, `Category`, and `Text`. |
| Data management | Excel import, SQLite CRUD, filtering, aggregation, visualization, schema administration, indexing, governance metadata, and guarded SQL. |

## Model modality boundary

Gemma 3 4B IT supports:

- text input;
- image input;
- text output.

Bro therefore exposes **Image-to-Text and image understanding**, but does not claim native image
generation, transcription, text-to-speech, or audio generation for the installed model.

## Execution design

The application separates controls according to the layer they actually affect:

```text
Task Controls
    -> task/system instructions

Response Controls
    -> output contract

Context Controls
    -> message/context construction

Inference Settings
    -> create_chat_completion(...)

Runtime Settings
    -> Llama(...)
```

This organization keeps the UI aligned with the actual generation functions and prevents task,
sampling, context, and runtime settings from being conflated.

## Documentation map

- [Architecture](architecture.md) — runtime boundaries and request flows.
- [Local Model](local-model.md) — Gemma 3 GGUF and multimodal-projector configuration.
- [Text Generation](text-generation.md) — text task and model-control reference.
- [Image to Text](image-to-text.md) — priority Gemma vision workflow.
- [Document Q&A](document-qna.md) — retrieval, grounding, OCR, and document actions.
- [Semantic Search](semantic-search.md) — embedding-index workflows.
- [Prompt Engineering](prompt-engineering.md) — prompt schema and category-aware templates.
- [Data Management](data-management.md) — local SQLite operations.
- [Development](development.md) — extension, validation, and documentation guidance.
