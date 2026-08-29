# Semantic Search

Semantic Search builds a reusable local semantic index from uploaded documents and retrieves the most
similar chunks for a query.

## Index Builder

The index builder supports:

- PDF/TXT/DOCX upload;
- Chunk Size;
- Chunk Overlap;
- Clear Existing Index;
- Append to Existing Index;
- embedding diagnostics.

Embeddings are generated locally through `sentence-transformers`.

## Document identity

The `embeddings` table retains source-document identity in addition to chunk text and vector data.
This enables document-aware query behavior.

## Semantic Query

Controls include:

- Top K;
- Minimum Similarity;
- Group by Document;
- query text.

## Group by Document

`Group by Document` is functional rather than decorative.

When enabled:

1. chunks are scored by cosine similarity;
2. results are ordered by similarity;
3. only the strongest result for each source document is retained;
4. Top K is applied to the document-grouped result set.

## Result routing

Selected semantic chunks can be:

- sent to Text Generation;
- sent to Document Q&A;
- saved into shared prompt/document context.

## Failure behavior

Missing embedding support or malformed/stale vector dimensions degrade through guarded paths rather
than producing an uncaught UI-to-model exception.
