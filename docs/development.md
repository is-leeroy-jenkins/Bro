# Bro Development

This page describes the development workflow for maintaining Bro, validating source changes, and building the MkDocs documentation site.

## 🧭 Purpose

Development work on Bro should preserve the Streamlit workflow, local-first model architecture, SQLite persistence, and source-driven documentation pattern. The safest maintenance approach is to make narrow changes, validate them immediately, and treat MkDocs warnings as defects.

## 🧰 Environment Setup

Create and activate a virtual environment from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application through Python so the active virtual environment is used.

```powershell
python -m streamlit run app.py
```

## 🧪 Validation Commands

Run these checks after source changes.

```powershell
python -m py_compile .\app.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
python -m compileall .
```

Run the documentation build.

```powershell
mkdocs build
```

Serve the documentation locally.

```powershell
mkdocs serve
```

## 📚 Documentation Rules

Bro documentation is generated from Google-style source docstrings and Markdown pages.

### Source Docstrings

Use this profile:

```text
Purpose:
Args:
Attributes:
Returns:
Raises:
Notes:
Examples:
```

Do not use NumPy-style underlines in docstrings.

```text
Parameters:
-----------
Returns:
--------
```

Do not document `self`.

Do not add `Returns:` sections to `__init__`.

Do not add meaningless returns such as:

```text
Returns:
    None: This method does not return a value.
```

### Good Function Pattern

```python
def get_timestamp_text( ) -> str:
    """Return a timestamp for local metadata records.

    Purpose:
        Builds the timestamp text used by document, chunk, embedding, and image
        metadata rows. This keeps local asset-governance records consistent across
        the Data Management and Document Q&A workflows.

    Returns:
        str: Timestamp text formatted for local SQLite metadata fields.
    """
```

### Good Procedure Pattern

```python
def clear_semantic_index( ) -> None:
    """Clear the semantic-search index.

    Purpose:
        Deletes stored embedding rows and resets Semantic Search diagnostic state.
        This prepares the application to build a new semantic index without retaining
        stale result rows, selected chunks, or document counters.
    """
```

## 🧯 Logging Pattern

Bro uses the project `Error` and `Logger` pattern from `boogr.py`.

Use this pattern for handlers that should re-raise:

```python
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'Semantic Search'
    exception.method = 'query_semantic_index( query_text: str ) -> List[Dict[str, Any]]'
    Logger( ).write( exception )
    raise exception
```

Use this pattern for safe fallback handlers:

```python
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'LLM Runtime'
    exception.method = 'load_embedder( ) -> Any | None'
    Logger( ).write( exception )
    return None
```

## 🧷 Source Preservation Checklist

Before accepting any source update, confirm:

| Check | Required Result |
| --- | --- |
| Function count | No public functions removed unintentionally. |
| Signatures | Existing parameter names, defaults, and return annotations preserved. |
| Imports | No unnecessary dependency added. |
| Session state | All session-state keys are written before read. |
| UI structure | Columns, tabs, expanders, button order, and workflow layout preserved. |
| Runtime behavior | LLM loading, retrieval, fallback, and database behavior preserved. |
| Logging | Existing handlers log through `Logger` without leaking user input or secrets. |
| Compilation | `py_compile` and `compileall` pass. |
| Documentation | `mkdocs build` completes without griffe warnings. |

## 🏗 MkDocs Structure

Recommended documentation structure:

```text
docs/
├── index.md
├── architecture.md
├── development.md
├── text-generation.md
├── document-qna.md
├── semantic-search.md
├── prompt-engineering.md
├── data-management.md
├── assets/
│   ├── css/
│   │   └── bro.css
│   └── js/
│       └── bro.js
└── api/
    ├── index.md
    ├── app.md
    ├── config.md
    └── boogr.md
```

## 🧭 Recommended Development Sequence

1. Make one source change at a time.
2. Run `py_compile` on the changed file.
3. Run `python -m compileall .`.
4. Run `mkdocs build`.
5. Fix griffe warnings before continuing.
6. Run the affected Streamlit mode manually.
7. Commit only after both source and docs validations pass.

## ✅ Release Readiness

Bro is ready for a documentation release when:

| Area | Result |
| --- | --- |
| Source | Python compiles cleanly. |
| App | Streamlit starts from the project root. |
| Docs | MkDocs builds cleanly. |
| API reference | `app`, `config`, and `boogr` render without griffe warnings. |
| Navigation | Every Markdown page is included in `mkdocs.yml`. |
| Styling | Dark-mode theme, header color, tables, code blocks, and API objects render correctly. |
| GitHub Pages | Built site deploys from the selected branch or GitHub Actions workflow. |
