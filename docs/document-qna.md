# Document Q&A

Document Q&A provides retrieval-augmented answering over uploaded documents. It extracts text, chunks the content, builds or refreshes a retrieval index, retrieves relevant excerpts, and sends a grounded prompt to the local model.

## 🧭 Purpose

This mode is used when answers must be grounded in uploaded source material. It is appropriate for document review, policy analysis, technical-documentation review, contract review, research notes, and structured extraction from local files.

## 🧱 Workflow Position

```text
Upload documents
  ▼
Select active documents
  ▼
Extract text
  ▼
Chunk text
  ▼
Embed chunks
  ▼
Retrieve top-k excerpts
  ▼
Build grounded prompt
  ▼
Generate local answer
```

## 📥 Document Intake

Bro supports local upload workflows. The application stores file bytes in session state and tracks active document names. The document inventory can display size, extracted text length, chunk count, and load status.

| Intake Element | Role |
| --- | --- |
| Uploaded files | Source files provided through the Streamlit interface. |
| Active documents | User-selected files included in the current Q&A workflow. |
| Document bytes | Runtime byte cache used for extraction and fingerprinting. |
| Fingerprint | Stable hash used to detect changes in selected document content. |
| Inventory rows | Diagnostic summary of loaded documents and chunk counts. |

## 🧾 Text Extraction

Document Q&A uses native PDF text extraction when available and falls back to decoded text when appropriate.

| Setting | Description |
| --- | --- |
| Prefer native PDF text | Uses PyMuPDF extraction for PDF files when available. |
| Include page markers | Adds page markers to extracted text when enabled. |
| OCR enabled | Reserved for workflows where OCR is explicitly configured. |
| Show diagnostics | Displays document-processing details for troubleshooting. |

## 🧩 Chunking

Chunks are built from extracted text and controlled by retrieval settings.

| Setting | Description |
| --- | --- |
| Retrieval chunk size | Maximum character length for each chunk. |
| Retrieval chunk overlap | Overlap between adjacent chunks to preserve context. |
| Retrieval k | Number of chunks retrieved for the question. |

A larger chunk size provides more local context per retrieved result. A smaller chunk size can improve targeted retrieval but may split related facts across chunks.

## 🔎 Retrieval Backends

Bro supports two retrieval paths.

| Backend | Use |
| --- | --- |
| `sqlite-vec` | Preferred vector search path when the SQLite extension is available. |
| Cosine fallback | In-memory fallback over stored vector blobs when `sqlite-vec` is unavailable or disabled. |

The fallback path is important because local development environments may not always have the SQLite vector extension available.

## 🧠 Grounded Prompt Construction

The document prompt combines instructions, active document names, retrieved excerpts, optional semantic context, and the user request.

```text
Document Q&A Instructions
  + Active document list
  + Retrieved excerpts
  + Optional semantic context
  + User request
  + Answer instruction
```

## 🧪 Document Actions

| Action | Purpose |
| --- | --- |
| Answer Question | Answers the user's question using retrieved excerpts. |
| Summarize Active Document | Produces a structured summary of active documents. |
| Extract Key Points | Extracts the most important points from the evidence. |
| Generate Outline | Builds an outline from retrieved document content. |
| Extract Entities | Identifies names, organizations, dates, and references. |
| Extract Tables | Describes or extracts structured tabular content visible in excerpts. |
| Compare Active Documents | Compares active documents for alignment, differences, and gaps. |

## ✅ Recommended Sequence

1. Upload documents.
2. Confirm active documents.
3. Review inventory and text extraction status.
4. Set chunk size and overlap.
5. Set retrieval count.
6. Enable grounding and answer-from-excerpts behavior.
7. Ask the question or select a document action.
8. Review retrieved chunks when diagnostics are needed.

## ⚠️ Practical Notes

| Issue | Resolution |
| --- | --- |
| Empty answer | Confirm text was extracted and active documents are selected. |
| Weak retrieval | Increase retrieval count or adjust chunk size. |
| Missing PDF text | Confirm PyMuPDF is installed and the PDF contains selectable text. |
| Slow indexing | Reduce active document set or use larger chunks. |
| Hallucinated answer | Enable answer-from-excerpts-only and require grounding. |

## 🧯 Failure Handling

Document processing should preserve safe fallback behavior. Extraction, vector loading, indexing, and retrieval failures should be logged through the project logger while allowing the UI to report unavailable text, unavailable vector search, or empty retrieval results.

## 🔗 Related Pages

- [Text Generation](text-generation.md)
- [Semantic Search](semantic-search.md)
- [Data Management](data-management.md)
