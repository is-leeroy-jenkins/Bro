# Semantic Search

Semantic Search builds and queries a local embedding index from uploaded documents. It retrieves text chunks by conceptual similarity rather than exact keyword matching.

## 🧭 Purpose

This mode helps users discover relevant passages across uploaded files, select useful context, and route selected chunks into Text Generation or Document Q&A. It is useful when the user does not know the exact wording used in the source documents.

## 🧱 Workflow Position

```text
Upload files
  ▼
Extract text
  ▼
Chunk text
  ▼
Generate embeddings
  ▼
Store vectors in SQLite
  ▼
Embed query
  ▼
Rank chunks by cosine similarity
  ▼
Select and route context
```

## 📥 Index Builder

The index builder processes uploaded files and stores vectors in the local `embeddings` table.

| Control | Description |
| --- | --- |
| Chunk size | Maximum character length per semantic chunk. |
| Chunk overlap | Overlap between adjacent chunks to preserve surrounding context. |
| Clear existing index | Removes previous embedding rows before indexing. |
| Append existing index | Adds new chunks without clearing existing rows. |
| Show diagnostics | Displays document count, chunk count, and vector dimension. |

## 🔎 Query Controls

| Control | Description |
| --- | --- |
| Query text | User search request converted into an embedding vector. |
| Top-k | Maximum number of ranked chunks returned. |
| Minimum similarity | Filters out weak matches below the selected threshold. |
| Group by document | Organizes results by source document when enabled. |

## 📊 Result Rows

Semantic results are represented as selectable rows.

| Field | Meaning |
| --- | --- |
| Selected | Indicates whether the row should be routed as context. |
| Rank | Position in the similarity-sorted result list. |
| Score | Cosine similarity score. |
| Chunk | Retrieved text. |
| Length | Character length of the chunk. |

## 🔁 Context Routing

Selected semantic chunks can be routed to other workflows.

| Action | Destination |
| --- | --- |
| Send to Text Generation | Adds selected chunks to shared document context and enables semantic context for general generation. |
| Send to Document Q&A | Adds selected chunks to the semantic context buffer used during grounded answering. |
| Save as prompt context | Preserves selected chunks for reuse in prompt-oriented workflows. |

## 🧪 Example Workflow

1. Open **Semantic Search**.
2. Upload one or more documents.
3. Set chunk size and overlap.
4. Build the index.
5. Enter a conceptual query.
6. Review ranked chunks.
7. Select the best chunks.
8. Route selected chunks to Text Generation or Document Q&A.

## 🧠 Similarity Behavior

Semantic Search compares the query embedding against stored chunk embeddings. A high score indicates conceptual similarity between the query and the chunk. It does not guarantee that the chunk fully answers the question; it identifies candidate evidence for review or routing.

## ✅ Recommended Settings

| Scenario | Suggested Setting |
| --- | --- |
| Broad discovery | Higher top-k and moderate chunk size. |
| Precise lookup | Lower top-k and smaller chunks. |
| Long policy documents | Larger chunks with overlap. |
| High-confidence routing | Use a minimum similarity threshold. |
| Re-indexing new files | Clear existing index unless intentionally combining corpora. |

## 🧯 Failure Handling

Semantic indexing depends on text extraction and the embedding model. If an uploaded file cannot be read, that file should be skipped without blocking the entire batch. If the embedding model is unavailable, the mode should report that semantic search cannot run and preserve application stability.

## 🔗 Related Pages

- [Text Generation](text-generation.md)
- [Document Q&A](document-qna.md)
- [Data Management](data-management.md)
