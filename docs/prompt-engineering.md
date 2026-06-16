# Prompt Engineering

Prompt Engineering manages reusable prompt templates stored in the local SQLite database. It supports searching, editing, cloning, generating, applying, and cascading prompt metadata into other application modes.

## 🧭 Purpose

This mode turns prompt design into a governed local asset. Analysts can store reusable system instructions, categorize prompt templates, apply them to Text Generation or Document Q&A, and generate starter templates from structured metadata.

## 🧱 Workflow Position

```text
Prompt metadata
  ▼
Prompt table
  ▼
Search, edit, clone, generate, apply
  ▼
Shared system instructions
  ▼
Text Generation or Document Q&A
```

## 🗃 Prompt Table

Prompt templates are stored in the local `Prompts` table.

| Field | Use |
| --- | --- |
| `PromptsId` | Primary key for prompt records. |
| `Caption` | User-facing prompt label. |
| `Name` | Internal or descriptive prompt name. |
| `Text` | Prompt body or system instruction text. |
| `Version` | Version label for template tracking. |
| `ID` | Optional external or user-defined identifier. |

## 🔎 Search and Browse

Prompt Engineering supports prompt discovery by searching prompt metadata and content. Pagination and direct ID navigation help maintain large local prompt libraries.

| Capability | Description |
| --- | --- |
| Search | Searches captions, names, and prompt text. |
| Sort | Orders prompt records for review. |
| Pagination | Displays manageable slices of the prompt table. |
| Go to ID | Opens a known prompt by primary key. |
| Category inference | Infers prompt category from caption, name, and text content. |

## 🧠 Prompt Categories

Bro can classify prompt templates into practical task categories.

| Category | Typical Use |
| --- | --- |
| General Chat | General assistant behavior. |
| Reasoning | Structured analysis and careful conclusion generation. |
| Coding | Code generation, refactoring, debugging, or review. |
| Translation | Language conversion. |
| Summarization | Concise condensation of longer content. |
| Extraction | Pulling facts from text. |
| Document Extraction | Evidence-based extraction from uploaded material. |
| OCR | Text extraction or cleanup workflows. |
| JSON Output | Machine-readable responses. |

## 🧰 Prompt Actions

| Action | Result |
| --- | --- |
| Apply to Text Generation | Copies prompt text into shared system instructions for Text Generation. |
| Apply to Document Q&A | Copies prompt text into shared system instructions and enables grounded-answer settings. |
| Clone | Creates a new editable prompt draft based on an existing record. |
| Generate starter prompt | Uses metadata such as task type, format, language, goal, constraints, and style to draft a template. |
| Create | Inserts a new prompt record. |
| Update | Saves changes to an existing prompt record. |
| Delete | Removes a prompt record. |
| Clear | Resets the edit surface. |

## 🧪 Example Workflow

1. Open **Prompt Engineering**.
2. Search for an existing template.
3. Select a prompt record.
4. Review category, task type, response format, and language.
5. Apply the prompt to Text Generation or Document Q&A.
6. Adjust mode-specific settings.
7. Run the target workflow.

## 🧷 Prompt Design Guidance

| Pattern | Recommendation |
| --- | --- |
| System instructions | State role, task, constraints, and output format clearly. |
| Extraction prompts | Define fields and missing-value behavior. |
| Coding prompts | Specify language, target framework, formatting, and validation expectations. |
| Document prompts | Require grounding in retrieved excerpts. |
| JSON prompts | Require valid JSON only and specify the schema. |
| Reusable templates | Keep project-specific assumptions explicit. |

## 🧯 Failure Handling

Prompt database operations should preserve the local SQLite workflow. Lookup failures should return safe defaults where the UI expects optional records, while write failures should be logged with stable method signatures and without persisting prompt text inside the method field.

## 🔗 Related Pages

- [Text Generation](text-generation.md)
- [Document Q&A](document-qna.md)
- [Data Management](data-management.md)
