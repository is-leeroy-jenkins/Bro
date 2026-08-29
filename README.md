###### Bro

![](https://github.com/is-leeroy-jenkins/Bro/blob/main/resources/images/bro_project.png)

<p align="center">
  <a href="#-key-features">Features</a> ·
  <a href="#-application-modes">Modes</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-repository-structure">Structure</a> ·
  <a href="#-installation--setup">Install</a> ·
  <a href="#-configuration">Configuration</a> ·
  <a href="#-text-generation">AI</a> ·
  <a href="#-document-qa">RAG</a> ·
  <a href="#-semantic-search">Search</a> ·
  <a href="#-prompt-engineering">Prompts</a> ·
  <a href="#-data-management">Data</a> ·
  <a href="#-requirements">Requirements</a> ·
</p>

___

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0078FC?style=for-the-badge&logo=github)](https://is-leeroy-jenkins.github.io/Bro/)


Bro is a local-first Streamlit application for text generation, document-grounded retrieval,
semantic search, prompt engineering, and SQLite-backed data management. It is designed to run a local 
GGUF language model through `llama-cpp-python` while giving analysts direct control over inference
parameters, prompt templates, document context, retrieval behavior, semantic chunking, and local
application data.



## 🎬 Demo


![](https://github.com/is-leeroy-jenkins/Bro/blob/main/resources/images/bro-demo.gif)

___

![](https://github.com/is-leeroy-jenkins/Bro/blob/main/resources/images/Bro-streamlit.gif)


## ☁️ Cloud


<table>
<tr>
<th align="center"><img width="190" height="1" alt=""><br>🧊 Azure</th>
<th align="center"><img width="190" height="1" alt=""><br>🧠 GPT</th>
<th align="center"><img width="190" height="1" alt=""><br>🔥 Streamlit</th>
<th align="center"><img width="190" height="1" alt=""><br>🧱 Databricks</th>
</tr>
<tr>
<td align="center">
<a href="https://bro.gentlebush-abcd8721.eastus.azurecontainerapps.io">
<img src="https://img.shields.io/badge/Docker-App-2496ED?logo=docker&logoColor=white" alt="Docker App">
</a>
</td>
<td align="center">
<a href="https://chatgpt.com/g/g-6759fe553bd481919d3cebfb4c875830-bro">
<img src="https://img.shields.io/badge/OpenAI-GPT-412991?logo=openai&logoColor=white" alt="OpenAI GPT">
</a>
</td>
<td align="center">
<a href="https://bro-py.streamlit.app/">
<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit App">
</a>
</td>
<td align="center">
<a href="https://dbc-a0c21f80-7bb3.cloud.databricks.com/browse/folders/3169291152440505?o=7474645703081351">
<img src="https://img.shields.io/badge/Databricks-Bro-FF3621?logo=databricks&logoColor=white" alt="Databricks Bro">
</a>
</td>
</tr>
</table>

## 🧠 Local Model

Bro is currently designed around **Gemma 3 4B IT in GGUF format** and runs the model locally
through `llama-cpp-python`.

The text-generation runtime uses the model's chat-completion path rather than manually composing
model-specific control tokens. When a compatible Gemma 3 multimodal projector (`mmproj*.gguf`) is
available, Bro also enables first-class **Image-to-Text / vision understanding**.

The project model repository remains available here:

[![](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/bro)

> **Modality boundary:** Gemma 3 4B IT accepts text and image input and produces text output. Bro
> therefore exposes image understanding and OCR-style Image-to-Text functionality, but does not
> advertise native transcription, text-to-speech, or image generation for this model.

## ✨ Key Features

| Feature | Description |
| --- | --- |
| Local Gemma 3 GGUF inference | Runs Gemma 3 4B IT locally through `llama-cpp-python` with configurable context, CPU threads, batching, micro-batching, sampling, penalties, seed, and token limits. |
| Gemma-compatible chat completion | Uses `create_chat_completion()` for text generation so the GGUF chat template controls model serialization. |
| Image-to-Text / vision | Uses Gemma 3 multimodal inference with a matching `mmproj` GGUF for visible-text extraction, image description, screenshot/chart/diagram analysis, structured extraction, image Q&A, and image comparison. |
| Vision-assisted PDF OCR | Document Q&A can fall back to Gemma 3 Image-to-Text for image-only PDF pages when OCR is enabled and a compatible multimodal projector is available. |
| Expanded text tasks | Chat, Analysis, Reasoning, Coding, Writing, Editing, Summarization, Extraction, Classification, Translation, Comparison, and Structured Output. |
| Bounded task controls | Enumerable parameters such as task type, spoken/written language, translation mode, response format, coding language, classification type, and vision task use bounded Streamlit controls rather than free-form entry. |
| Structured responses | Plain Text, Markdown, Bullet List, Numbered List, Markdown Table, JSON, XML, YAML, CSV, and Code response contracts. JSON can use llama.cpp's structured response format support. |
| Persistent chat history | Saves local role/content chat messages to SQLite and restores history on startup. |
| Category-aware system instructions | Filters prompt templates by persisted `Prompts.Category`, selects templates by primary key, supports manual editing, reset, XML ↔ Markdown conversion, presets, and effective-prompt preview. |
| Document Q&A | Uploads PDF, TXT, or DOCX files, chunks extracted content, retrieves relevant excerpts, applies grounding policies, and sends the resulting context through the shared Gemma chat path. |
| Expanded document actions | Answer questions, summarize, extract key points/entities/tables/dates/organizations/requirements/action items, generate outlines/executive summaries, find evidence, classify documents, compare documents, identify contradictions, and identify missing information. |
| Semantic search | Builds a local semantic index, filters by minimum similarity, supports Top-K retrieval, and can group results by source document. |
| sqlite-vec + cosine retrieval | Supports sqlite-vec when available and a guarded cosine-similarity fallback. |
| Prompt engineering | Searches, filters, sorts, pages, edits, creates, clones, generates, deletes, and applies prompt templates using the authoritative five-column `Prompts` schema. |
| Data management | Imports Excel data into SQLite; browses, edits, profiles, filters, aggregates, visualizes, administers schemas/indexes, and runs guarded read-only SQL. |
| AI asset governance | Registers document, chunk, embedding, and image metadata in governed local SQLite tables. |
| Runtime-safe capability gating | Model, embedding, sqlite-vec, PDF parsing, and multimodal features degrade through controlled application paths when an optional dependency or model artifact is unavailable. |
| Fixed status footer | Displays current mode and important generation/runtime/context state. |

## 🧭 Application Modes

Bro exposes six functional modes. `Image to Text` is appended to the configured mode list by
`app.py` when it is not already present in `cfg.MODES`.

| Mode | Purpose | Major Controls / Outputs |
| --- | --- | --- |
| **Text Generation** | Primary Gemma 3 text-generation and chat interface. | Task presets; reasoning, coding, writing, translation, classification, response, inference, context, and runtime controls; category-aware system instructions; prompt preview; streaming chat. |
| **Image to Text** | Gemma 3 multimodal image understanding and visible-text extraction. | Vision task, image detail, response format/language, layout/text preservation, inference controls, runtime controls, image upload, optional user request, streaming vision response. |
| **Document Q&A** | Retrieval-augmented analysis over uploaded documents with optional Gemma vision OCR. | Retrieval, grounding, backend, document-action, parsing/OCR, diagnostics, response, inference, context, and runtime controls; document inventory; retrieved chunks. |
| **Semantic Search** | Build and query a reusable local semantic chunk index. | Index builder, diagnostics, Top-K, minimum similarity, group-by-document, selectable results, context-routing actions, maintenance controls. |
| **Prompt Engineering** | Manage reusable prompt templates and application metadata. | Search/filter/sort/page, Go-to-ID, prompt actions, prompt generator, application settings, create/edit/delete/clone, cascade to Text Generation or Document Q&A. |
| **Data Management** | Manage local SQLite tables and AI asset metadata. | Excel import, browse, CRUD, explore, filter, aggregate, visualize, schema administration, index creation, asset governance, guarded SQL. |

## 🏛 Architecture

```text
Streamlit UI
    │
    ├── Text Generation
    │       ├── task / response instruction builder
    │       ├── context-message builder
    │       ├── llama.cpp runtime controls
    │       ├── Llama.create_chat_completion()
    │       ├── Gemma 3 4B IT GGUF
    │       ├── chat_history
    │       └── category-aware Prompts templates
    │
    ├── Image to Text
    │       ├── PNG / JPG / JPEG / WEBP uploads
    │       ├── Gemma vision instruction builder
    │       ├── MTMDChatHandler
    │       ├── matching mmproj GGUF
    │       ├── Gemma 3 4B IT GGUF
    │       └── text response
    │
    ├── Document Q&A
    │       ├── PDF / TXT / DOCX uploads
    │       ├── native PDF/text extraction
    │       ├── optional Gemma vision OCR
    │       ├── chunking
    │       ├── sentence-transformers embeddings
    │       ├── sqlite-vec when available
    │       ├── cosine fallback
    │       ├── grounding policy
    │       └── shared Gemma chat-completion path
    │
    ├── Semantic Search
    │       ├── chunked uploaded documents
    │       ├── document-aware embeddings table
    │       ├── minimum-similarity / Top-K ranking
    │       ├── optional group-by-document
    │       └── selected-context routing
    │
    ├── Prompt Engineering
    │       ├── Prompts(ID, Caption, Name, Category, Text)
    │       ├── database category filtering
    │       ├── prompt application settings
    │       ├── starter-prompt generation
    │       └── Text Generation / Document Q&A cascade
    │
    └── Data Management
            ├── Excel import
            ├── CRUD / profiling / visualization
            ├── guarded SQL console
            └── AI asset governance tables
```

## 🗂 Repository Structure

A typical local layout is:

```text
bro/
├─ app.py                         # Main Streamlit application
├─ config.py                      # Model path, defaults, modes, help text, images, styling
├─ requirements.txt               # Python dependencies
├─ resources/
│  └─ images/
│     ├─ bro_project.png
│     ├─ Bro-streamlit.gif
│     └─ bro_logo.png
├─ stores/
│  └─ sqlite/
│     └─ bro.db                   # Chat, prompts, embeddings, documents, chunks, images
├─ models/
│  ├─ <gemma-3-4b-it>.gguf       # Configured text/multimodal language model
│  └─ mmproj*.gguf               # Optional matching Gemma 3 multimodal projector
└─ README.md
```

The exact GGUF filenames are controlled by configuration and are not hard-coded by the README.

## ⚙️ System Requirements

| Requirement | Minimum | Recommended |
| --- | ---: | ---: |
| Operating system | Windows 10/11 64-bit, Linux, or macOS | Windows 11 64-bit or Linux |
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB or more |
| CPU | Modern x64 CPU | AVX2-capable multicore CPU |
| Storage | 5–7 GB free | 10+ GB for model/projector files, SQLite assets, and uploaded documents |
| GPU | Not required | Optional acceleration where supported by the installed llama.cpp build |
| Vision projector | Not required for text-only use | Matching Gemma 3 `mmproj*.gguf` for Image-to-Text and vision OCR |

Large context windows and high output-token limits materially increase memory requirements.

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/is-leeroy-jenkins/Bro.git
cd Bro
```

### 2️⃣ Create and Activate a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 📥 Configure Gemma 3

### Text Runtime

Download the Gemma 3 4B IT GGUF used by your local configuration and set `cfg.MODEL_PATH` to that
file.

Example environment-variable pattern when `config.py` resolves the model path from an environment
variable:

```powershell
$env:BRO_LLM_PATH="C:\Users\you\models\gemma-3-4b-it-Q4_K_M.gguf"
```

Bro checks the configured path before attempting model initialization.

### Image-to-Text / Vision Runtime

Vision requires a **matching multimodal projector GGUF** in addition to the language-model GGUF.

Bro resolves the projector in this order:

1. `cfg.MMPROJ_PATH`
2. `cfg.MM_PROJ_PATH`
3. `BRO_MMPROJ_PATH`
4. `GEMMA_MMPROJ_PATH`
Example:

```powershell
$env:BRO_MMPROJ_PATH="C:\Users\you\models\mmproj-gemma-3-4b-it-f16.gguf"
```

If no compatible projector is explicitly configured, text functionality remains available and
**Image to Text** shows a controlled warning instead of attempting an invalid multimodal call.

## ▶️ Running Bro

Run Streamlit through the active Python environment:

```bash
python -m streamlit run app.py
```

Bro opens in wide layout and renders the mode selector from `cfg.MODES`, with `Image to Text`
appended by `app.py` when necessary.

## 🔧 Configuration

| Configuration Item | Purpose |
| --- | --- |
| `cfg.MODEL_PATH` | Path to the local Gemma 3 4B IT GGUF. |
| `cfg.MMPROJ_PATH` / `cfg.MM_PROJ_PATH` | Optional configured Gemma 3 multimodal-projector path. |
| `BRO_MMPROJ_PATH` / `GEMMA_MMPROJ_PATH` | Environment-variable alternatives for the multimodal projector. |
| `cfg.DEFAULT_CTX` | Default llama.cpp context window. |
| `cfg.CORES` | Maximum CPU thread count exposed by runtime controls. |
| `cfg.DB_PATH` | SQLite database used by chat, prompts, embeddings, documents, chunks, images, and imported data. |
| `cfg.FAVICON` | Streamlit page icon. |
| `cfg.LOGO` | Sidebar logo. |
| `cfg.APP_SUBTITLE` | Startup caption. |
| `cfg.MODES` | Configured sidebar application modes. |
| `cfg.BLUE_DIVIDER` | Shared divider styling. |
| `cfg.XML_BLOCK_PATTERN` | XML-like prompt-section pattern used by conversion utilities. |
| `cfg.TEXT_GENERATION` | Text Generation help text. |
| `cfg.RETRIEVAL_AUGMENTATION` | Document Q&A help text. |
| `cfg.SEMANTIC_SEARCH` | Semantic Search help text. |
| `cfg.PROMPT_ENGINEERING` | Prompt Engineering help text. |
| `cfg.DATA_MANAGEMENT` | Data Management help text. |

## 💬 Text Generation

Text Generation uses a layered execution contract:

```text
Task Controls
      ↓
build_task_instruction_block()

Response Controls
      ↓
response instructions / structured output contract

Context Controls
      ↓
build_chat_messages()

Inference Settings
      ↓
create_chat_completion()

Runtime Settings
      ↓
Llama(...)
```

### Supported Tasks

```text
Chat
Analysis
Reasoning
Coding
Writing
Editing
Summarization
Extraction
Classification
Translation
Comparison
Structured Output
```

### 🧭 Task Preset

| Control | Options / Purpose |
| --- | --- |
| Task Type | Complete supported task list shown above. |
| Task Detail | Concise, Standard, Detailed. |
| Task Focus | Accuracy, Balanced, Creativity. |

### 🧩 Reasoning Controls

- Reasoning Depth
- Answer Only
- Use Self-Check
- Prefer Deterministic Reasoning

### 🧾 Coding Controls

| Control | Options / Purpose |
| --- | --- |
| Code Language | Python, C, C++, C#, Java, JavaScript, TypeScript, SQL, VBA, PowerShell, Bash, HTML, CSS, Markdown, JSON, YAML, Other. |
| Coding Task | Generate, Complete, Refactor, Debug, Review, Explain, Optimize, Convert, Test, Document, Design. |
| Include Comments | Includes useful documentation/comments. |
| Use Editor Format | Requests editor-ready code. |
| Emit Fenced Code | Wraps code in Markdown fences when enabled. |

### ✍️ Writing Controls

- Writing Task: Draft, Rewrite, Edit, Proofread, Expand, Condense, Reformat
- Tone: Neutral, Professional, Formal, Conversational, Technical, Academic
- Audience: General, Technical, Executive, Federal, Academic
- Length

### 🌐 Translation Controls

- Source Language
- Target Language
- Translation Mode: Natural, Literal, Formal, Technical, Localization
- Preserve Formatting

Human languages are supplied through bounded selectboxes. `Auto Detect` is available where
appropriate for the source language.

### 🏷️ Classification Controls

- Classification Type: Binary, Multi-Class, Multi-Label, Sentiment, Intent, Topic, Relevance
- Return Confidence
- Allow Unknown
- Explain Classification

### ↔️ Response Controls

- Response Format
- Response Language
- Response Length
- Include Headings

Supported response formats:

```text
Plain Text
Markdown
Bullet List
Numbered List
Markdown Table
JSON
XML
YAML
CSV
Code
```

### 🎚️ Inference Settings

**Row 1**

- Temperature
- Top-P
- Top-K
- Repeat Penalty
- Repeat Window

**Row 2**

- Presence Penalty
- Frequency Penalty
- Random Seed
- Max Tokens

These values feed the shared llama.cpp generation path. Max Tokens is exposed up to 8,192.

### 🎛️ Context Controls

- Context Window
- Use Conversation History
- Use Document Context
- Use Semantic Context
- Use Grounding

The context control permits values up to 131,072 tokens; practical limits depend on available
memory and GGUF/runtime configuration.

### ⚙️ Runtime Settings

- CPU Threads
- Batch Size
- Micro Batch Size

### 🖥️ System Instructions

The System Instructions surface supports:

- persisted Category selection;
- category-filtered template selection;
- primary-key-backed template loading;
- manual instruction editing;
- clear/reset;
- XML ↔ Markdown conversion;
- preset application in Text Generation;
- effective-prompt preview.

The selected template ultimately supplies `system_instructions`; downstream generation does not need
to know how the template was located in SQLite.

## 🖼️ Image to Text

Image to Text is Bro's first-class Gemma 3 vision mode.

### Supported Inputs

```text
PNG
JPG
JPEG
WEBP
```

Multiple images can be uploaded for comparison.

### 👁️ Vision Controls

| Control | Purpose |
| --- | --- |
| Vision Task | Selects the multimodal operation. |
| Image Detail | Concise, Standard, Detailed. |
| Response Format | Uses Bro's standard bounded response formats. |
| Response Language | Uses the same bounded human-language vocabulary as text generation. |
| Preserve Layout | Requests preservation of visible spatial/text structure when practical. |
| Include Visible Text | Requests explicit transcription of visible text in the response. |

Supported Vision Tasks:

```text
Extract Visible Text
Describe Image
Answer Questions
Analyze Screenshot
Analyze Chart
Analyze Diagram
Extract Structured Data
Compare Images
```

`Extract Visible Text` is the default priority workflow.

### Vision Inference Settings

Vision uses the same generation settings as the text runtime:

- Temperature
- Top-P
- Top-K
- Repeat Penalty
- Repeat Window
- Presence Penalty
- Frequency Penalty
- Random Seed
- Max Tokens

### Vision Runtime Settings

- Context Window
- CPU Threads
- Batch Size
- Micro Batch Size

The multimodal runtime is created with `MTMDChatHandler` and the matching `mmproj` GGUF. Vision
execution is capability-gated so a missing projector produces a controlled application message.

### Unsupported Model Outputs

The current Gemma 3 4B IT configuration does **not** provide native:

- image generation;
- image editing/inpainting;
- audio transcription;
- speech generation;
- text-to-speech.

Those features would require a different or additional model.

## 📚 Document Q&A

Document Q&A performs retrieval-augmented analysis over uploaded PDF, TXT, and DOCX files and shares
the same guarded Gemma generation path used by Text Generation.

### 🧲 Retrieval Controls

- Chunks to Retrieve
- Chunk Size
- Chunk Overlap
- Show Retrieved Chunks

### 🛡️ Grounding Controls

- Require Grounding
- Answer From Excerpts Only
- Insufficient-Evidence Behavior:
  - State Insufficient Information
  - Return Retrieved Excerpts
  - Best Supported Answer

### 🧮 Retrieval Backend

- Automatic
- sqlite-vec
- Cosine Similarity
- Allow Cosine Fallback
- Rebuild Index Each Query

### 🗂️ Document Actions

Bro supports:

```text
Answer Question
Summarize Active Document
Extract Key Points
Generate Outline
Extract Entities
Extract Tables
Compare Active Documents
Classify Document
Find Evidence
Generate Executive Summary
Extract Dates
Extract Organizations
Extract Requirements
Extract Action Items
Identify Contradictions
Identify Missing Information
```

Document actions also expose a bounded detail level.

### 📄 Document Parsing

- Enable OCR
- Prefer Native PDF Text
- Include Page Markers
- OCR Page Limit: 1, 2, 5, 10, or All Pages

#### Vision OCR Flow

PDF extraction is evaluated page-by-page, allowing digital and scanned pages to coexist in the same document.

When OCR is enabled:

```text
PDF page
   ↓
native text available?
   ├── Yes → use native text
   └── No
        ↓
Gemma vision runtime available?
   ├── No  → controlled fallback/status
   └── Yes
        ↓
render page to image
        ↓
run_vision_turn()
        ↓
visible text
        ↓
chunk / embed / retrieve
```

OCR results are cached in session state for the active workflow.

### 🔎 Diagnostics

- Show Diagnostics
- Show OCR Status
- Show Runtime Metadata

### ↔️ Response Controls

- Response Format
- Response Language
- Response Length
- Include Headings

### 🎚️ Inference Settings

Document Q&A uses the same inference contract as Text Generation.

### 🎛️ Context Controls

- Context Window
- Include Semantic Context
- Context Order:
  - Retrieved First
  - Semantic First

### ⚙️ Runtime Settings

- CPU Threads
- Batch Size
- Micro Batch Size

### Document Loader

The loader provides:

- PDF/TXT/DOCX upload;
- active-document selection;
- PDF or extracted-text preview;
- active-document inventory;
- unload/reset;
- optional diagnostics.

## 🔍 Semantic Search

Semantic Search creates and queries a reusable local embedding index.

| Section | Controls / Outputs |
| --- | --- |
| Index Builder | Upload files, chunk size, overlap, clear/append behavior, diagnostics, build index. |
| Diagnostics | Indexed documents, indexed chunks, vector dimension. |
| Semantic Query | Query text, Top-K, minimum similarity, and embedding diagnostics. |
| Results | Selectable ranked chunks with document name, score, chunk text, and length. |
| Actions | Send selected chunks to Text Generation, Document Q&A, or shared prompt context. |
| Maintenance | Delete/rebuild index state and clear result state. |

## 📝 Prompt Engineering

Prompt Engineering uses the authoritative schema:

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

`ID` is the immutable database identity. Category is persisted rather than inferred at runtime.

### Prompt Categories

Bro preserves the existing `Prompts.Category` values rather than replacing them with a new taxonomy.
The project/database category vocabulary includes values such as:

```text
Business / Finance / Marketing
Compliance / Legal / Budget
Data Analytics & Governance
Instruction/ Training / Planning
Prompt Engineering
Research / Academic
Software Engineering
Writing / Administrative
Image Analysis
Image Editing
Image Generation
Speech API
Transcription API
Translation API
```

Model-facing selectors do not rewrite these values. Instead, each workflow defines which existing
categories are appropriate and then shows only those categories that actually contain usable prompt
templates.

### Mode-specific category filtering

The `Prompts` table remains authoritative and unchanged.

For **Text Generation**, Bro can expose populated templates from:

```text
Writing / Administrative
Research / Academic
Data Analytics & Governance
Software Engineering
Business / Finance / Marketing
Compliance / Legal / Budget
Prompt Engineering
Instruction/ Training / Planning
```

For **Document Q&A**, Bro can expose populated templates from:

```text
Research / Academic
Data Analytics & Governance
Business / Finance / Marketing
Compliance / Legal / Budget
Instruction/ Training / Planning
Writing / Administrative
```

For **Image to Text**, Bro can expose populated `Image Analysis` templates only when the multimodal
runtime is available.

`Speech API`, `Transcription API`, `Image Generation`, and `Image Editing` remain preserved in SQLite
for Prompt Engineering/database management but are not offered to incompatible Gemma execution
paths.

### Capabilities

| Capability | Description |
| --- | --- |
| Search and filter | Search prompt content and filter using persisted Category. |
| Sort and pagination | Sort records and page through prompt results. |
| Go to ID | Jump directly to an integer `Prompts.ID`. |
| Template selection | Uses the database primary key instead of assuming Caption is unique. |
| Prompt actions | Apply to Text Generation, apply to Document Q&A, clone, or create a starter template. |
| Prompt generator | Generates editable prompt text from bounded task, response, language, style, goal, and constraint controls. |
| Edit surface | Edits schema-backed fields: ID, Category, Caption, Name, and Text. |
| Application settings | Task Type, Response Format, and Response Language are treated as application/cascade settings rather than database columns. |
| Create/update/delete | CRUD operations use the authoritative five-column schema. |

Selecting a generic response language does not silently overwrite the separate Translation Target
Language setting.

## 🗄️ Data Management

> **Database contract:** Bro does not rebuild or normalize the `Prompts` table to satisfy UI
> taxonomy changes. It validates that `ID`, `Caption`, `Name`, `Category`, and `Text` exist and
> otherwise leaves stored prompt values unchanged. The semantic `embeddings` table also retains
> its existing `id`, `chunk`, and `vector` persistence contract.


Data Management continues to provide general SQLite administration and AI asset governance.

| Tab | Purpose |
| --- | --- |
| **📥 Import** | Import Excel workbooks into SQLite and register active AI assets. |
| **🗂 Browse** | Browse SQLite tables. |
| **💉 CRUD** | Insert, update, and delete table records with type-aware controls. |
| **📊 Explore** | Page through records and inspect table contents. |
| **🔎 Filter** | Filter selected table data. |
| **🧮 Aggregate** | Compute aggregate metrics over supported columns. |
| **📈 Visualize** | Render Plotly-based visualizations. |
| **⚙ Admin** | Inspect schemas, manage tables/columns/indexes, profile data, and manage AI asset tables. |
| **🧠 SQL** | Execute guarded read-only SQL and export query results. |

### Core SQLite Tables

| Table | Purpose |
| --- | --- |
| `chat_history` | Persistent local chat history. |
| `embeddings` | Semantic-search chunks and vectors. |
| `Prompts` | `ID`, `Caption`, `Name`, `Category`, `Text`. |
| `documents` | Registered document metadata. |
| `document_chunks` | Registered document chunks. |
| `document_embeddings` | Document embedding metadata. |
| `images` | Uploaded-image governance metadata. |

## 📊 Status Footer

Bro's fixed footer summarizes active runtime state, including the current mode and relevant
generation/context settings such as temperature, Top-P, Top-K, penalties, repeat window, maximum
tokens, context window, CPU threads, semantic state, and shared document count.

## 📦 Requirements

Use `requirements.txt` as the version-pinning source of truth. The current application functionality
depends on the following packages or standard-library modules.

| Requirement | Package / Import | Purpose | Used By |
| --- | --- | --- | --- |
| Python | `python>=3.10` | Runtime and modern typing syntax. | Entire application |
| Streamlit | `streamlit` | UI, chat, uploaders, expanders, controls, tables, session state. | All modes |
| llama-cpp-python | `llama_cpp` | Local GGUF text and multimodal inference. | Text Generation, Image to Text, Document Q&A |
| MTMD chat handler | `llama_cpp.llama_chat_format.MTMDChatHandler` | Connects the Gemma 3 GGUF to the matching multimodal projector. | Image to Text, vision OCR |
| NumPy | `numpy` | Vector math and embedding-array handling. | Document Q&A, Semantic Search |
| Pandas | `pandas` | Dataframes, SQL results, prompt tables, imports, inventories. | Prompt Engineering, Data Management, retrieval views |
| Plotly Express | `plotly.express` | Interactive database visualizations. | Data Management |
| SQLite | `sqlite3` | Local persistence. | All persistent workflows |
| sqlite-vec | `sqlite_vec` | Optional vector-search backend. | Document Q&A |
| sentence-transformers | `sentence_transformers` | Local `all-MiniLM-L6-v2` embeddings. | Document Q&A, Semantic Search |
| PyMuPDF | `fitz` / `pymupdf` | Native PDF text extraction and PDF-page rendering for vision OCR. | Document Q&A |
| OpenPyXL | `openpyxl` | Excel workbook support through pandas. | Data Management |
| python-docx | `python-docx` | Native DOCX paragraph/table text extraction. | Document Q&A |
| Pillow | `pillow` | Supporting image handling. | Image workflows / metadata |
| boogr | `Error`, `Logger` | Application error logging. | Guarded execution paths |
| pathlib | Standard library | Model/projector/filesystem handling. | Runtime/configuration |
| base64 | Standard library | Image data-URI and UI image support. | Image to Text / utilities |
| hashlib | Standard library | Stable document/image fingerprints. | Retrieval / governance |
| re | Standard library | Prompt conversion, text normalization, identifier/query guards. | Utilities / data management |

A recent `llama-cpp-python` build with Gemma 3 multimodal/MTMD support is required for the
Image-to-Text path.

## 🔒 Privacy & Design Philosophy

| Principle | Implementation |
| --- | --- |
| Local-first inference | Gemma text and vision inference operate against local GGUF assets. |
| Local multimodal assets | Vision uses a local `mmproj` projector; no external vision API is required by the implemented path. |
| Local persistence | Chat history, prompts, embeddings, documents, chunks, images, and imported tables use SQLite. |
| Inspectable retrieval | Retrieved chunks and diagnostics can be displayed in Document Q&A. |
| Explicit grounding | Grounding behavior is controlled independently from retrieval and response formatting. |
| Capability-aware failure | Missing model/projector/embedder/vector dependencies produce controlled application behavior instead of advertising unavailable functionality. |
| Bounded configuration | Enumerated parameters are selected from explicit options instead of manually typed values. |
| SQL safety | The SQL console blocks mutation statements and permits guarded read-only forms. |
| Operational transparency | The footer and diagnostic controls expose relevant runtime state. |

## 🧬 Related Applications

| Application | Role |
| --- | --- |
| Leeroy | Entry-level instruction assistant. |
| Bro | Local Gemma 3 text, vision, retrieval, prompt-engineering, and data-management application. |
| Gipity | Larger multimodal/OpenAI-centered workflow application. |
| Chonky | Text-processing, tokenization, embeddings, and vector-persistence pipeline. |

## 📜 License

This application is provided for personal, research, and open-source use. Refer to the project and
model repositories for application- and model-specific licensing terms.
