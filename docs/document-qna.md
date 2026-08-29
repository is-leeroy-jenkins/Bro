# Document Q&A

Document Q&A provides local retrieval-augmented analysis over uploaded PDF, TXT, and DOCX files.

## Workflow

```text
load documents
    -> extract/OCR text
    -> chunk
    -> embed
    -> retrieve
    -> apply grounding policy
    -> build Gemma request
    -> create_chat_completion()
```

## 🧲 Retrieval Controls

- Chunks to Retrieve
- Chunk Size
- Chunk Overlap
- Show Retrieved Chunks

These values govern retrieval, not grounding policy or response presentation.

## 🛡️ Grounding Controls

- Require Grounding
- Answer From Excerpts Only
- Insufficient-Evidence Behavior

Supported insufficient-evidence behaviors include:

- State Insufficient Information
- Return Retrieved Excerpts
- Best Supported Answer

## 🧮 Retrieval Backend

Bro can use:

- Automatic
- sqlite-vec
- Cosine Similarity

Additional controls govern cosine fallback and whether the index is rebuilt for a query.

## 🗂️ Document Actions

The bounded action selector includes:

- Answer Question
- Summarize Active Document
- Extract Key Points
- Generate Outline
- Extract Entities
- Extract Tables
- Compare Active Documents
- Classify Document
- Find Evidence
- Generate Executive Summary
- Extract Dates
- Extract Organizations
- Extract Requirements
- Extract Action Items
- Identify Contradictions
- Identify Missing Information

## 📄 Document Parsing

- Enable OCR
- Prefer Native PDF Text
- Include Page Markers
- OCR Page Limit

OCR Page Limit is bounded rather than manually entered.

### Vision-assisted OCR

When native PDF text is unavailable and OCR is enabled, Bro can render eligible PDF pages as images
and route them through the Gemma multimodal runtime.

The OCR path therefore requires the same compatible `mmproj` configuration as
[Image to Text](image-to-text.md).

## 🔎 Diagnostics

Diagnostics are separated from parsing controls:

- Show Diagnostics
- Show OCR Status
- Show Runtime Metadata

## Response, inference, context, and runtime controls

Document Q&A uses the same conceptual ownership as Text Generation:

- Response Controls describe output;
- Inference Settings feed generation;
- Context Controls govern context assembly;
- Runtime Settings govern llama.cpp initialization.

This keeps the same session-state parameter from having different semantic homes in different modes.

## Retrieved chunks

When enabled, Bro renders retrieved chunks with source information and score/distance metadata so the
user can inspect the evidence base used by the answer.
