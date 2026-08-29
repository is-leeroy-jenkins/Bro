# Prompt Engineering

Prompt Engineering manages reusable system/prompt templates stored in SQLite.

## Authoritative schema

```sql
CREATE TABLE IF NOT EXISTS Prompts
(
    ID INTEGER NOT NULL UNIQUE,
    Caption TEXT(80),
    Name TEXT(80),
    Category TEXT(80),
    Text TEXT(2048),
    PRIMARY KEY(ID AUTOINCREMENT)
);
```

`ID` is the database identity. `Category` is persisted metadata and is no longer inferred from prompt
text during normal operation.

## Categories

Bro's supported category vocabulary includes:

- General Assistant
- Analysis & Reasoning
- Software Development
- Writing & Editing
- Summarization
- Information Extraction
- Classification
- Translation
- Structured Output
- Document Analysis
- Vision & Image Analysis
- Federal / Administrative Analysis

## Capability-filtered categories

The SQLite database may retain legacy prompt categories from earlier provider or model iterations.
Those records are preserved, but they are not automatically valid for the currently loaded model.

Model-facing selectors apply this contract:

```text
persisted Prompts.Category values
        ∩
supported Gemma 3 categories
        ∩
effective runtime capabilities
        =
categories exposed in the UI
```

For the current Gemma 3 4B IT runtime:

| Category / capability | UI treatment |
| --- | --- |
| Speech API | Excluded |
| Transcription API | Excluded |
| Text-to-Speech | Excluded |
| Audio Generation | Excluded |
| Image Generation | Excluded |
| Vision & Image Analysis | Included only when the multimodal projector runtime is available |
| Translation | Included |
| Structured Output | Included |
| Software Development | Included |
| Document Analysis | Included |

This prevents a stale SQLite category from advertising functionality the current model/runtime cannot
execute.

Persisted database categories are filtered against the supported Gemma 3 category vocabulary
and the effective runtime capabilities before being exposed in model-facing selectors.

## Category-aware System Instructions

System-instruction template selection follows:

```text
Category
   -> prompts in Category
   -> selected Prompts.ID
   -> Text
   -> system_instructions
```

Caption is a display label, not the database key.

## Prompt management

Prompt Engineering preserves:

- search;
- category filter;
- sort;
- pagination;
- Go to ID;
- table selection;
- apply to Text Generation;
- apply to Document Q&A;
- clone as new template;
- starter-prompt generation;
- prompt generator;
- create;
- update;
- delete;
- clear/reset.

## Edit surface

The database edit surface corresponds to persisted fields:

- ID
- Category
- Caption
- Name
- Text

Task Type, Response Format, and Response Language are application/cascade settings rather than fake
database columns.

## Language controls

Human-language values are bounded selectbox options. A generic response-language selection does not
silently overwrite the separate Translation Target Language setting.

## Legacy migration

Bro can migrate the previous prompt schema into the current five-column contract while preserving
existing prompt records.
