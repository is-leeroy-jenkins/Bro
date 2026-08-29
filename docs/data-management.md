# Data Management

Data Management exposes Bro's local SQLite environment for general data operations and AI asset
governance.

## Tabs

| Tab | Purpose |
| --- | --- |
| 📥 Import | Import Excel workbooks and register AI assets. |
| 🗂 Browse | Browse selected SQLite tables. |
| 💉 CRUD | Insert, update, and delete records. |
| 📊 Explore | Inspect/paginate table records. |
| 🔎 Filter | Filter selected table data. |
| 🧮 Aggregate | Compute supported aggregate metrics. |
| 📈 Visualize | Render Plotly visualizations. |
| ⚙ Admin | Manage schemas, indexes, tables, columns, profiles, and governed assets. |
| 🧠 SQL | Run guarded read-only SQL and export results. |

## Core tables

| Table | Purpose |
| --- | --- |
| `chat_history` | Persistent chat history. |
| `embeddings` | Semantic chunks, source identity, and vectors. |
| `Prompts` | Prompt templates and persisted Category metadata. |
| `documents` | Document governance metadata. |
| `document_chunks` | Governed chunk metadata/content. |
| `document_embeddings` | Embedding governance metadata. |
| `images` | Uploaded-image governance metadata. |

## Prompt schema

The current `Prompts` contract is:

```text
ID
Caption
Name
Category
Text
```

Data-management operations should not recreate the removed `PromptsId` or `Version` fields.

## Guarded SQL

The SQL console is intentionally read-only. Mutation statements are rejected so ad hoc analysis does
not become an uncontrolled schema/data-write path.

## AI asset governance

Bro can register active:

- documents;
- document chunks;
- embedding metadata;
- uploaded image metadata.

This metadata is distinct from the model's in-memory context and allows local inspection of managed
assets.

## Identifier handling

Table/column administration validates identifiers before composing schema operations. Arbitrary data
values continue to use parameterized SQL where applicable.
