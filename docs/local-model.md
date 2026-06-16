# Local LLM

## 🧭 Purpose

Bro uses a local GGUF large language model to provide private, offline-capable text generation,
reasoning, coding assistance, summarization, extraction, translation, document Q&A, and
semantic-context-assisted responses.

## 🧠 Custom LLM

[![](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/bro)

- Fine-tuned
- Post-trained

## 🧱 LLM Runtime Structure

| Area               | Runtime Focus                       | Purpose                                                                                                       |
| ------------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Model File         | GGUF model stored on disk           | Provides the local language model weights used by llama.cpp.                                                  |
| Model Path         | `cfg.MODEL_PATH`                    | Points Bro to the GGUF model file that should be loaded at runtime.                                           |
| Runtime Engine     | `llama-cpp-python`                  | Loads and executes the local GGUF model from Python.                                                          |
| Streamlit Cache    | `@st.cache_resource`                | Keeps the loaded model available across app interactions without reloading on every request.                  |
| Prompt Builder     | Local prompt-construction functions | Combines system instructions, task presets, chat history, document context, semantic context, and user input. |
| Inference Controls | Streamlit session-state settings    | Controls context window, CPU threads, max tokens, temperature, top-p, top-k, and repetition behavior.         |

## ⚙️ LLM Capabilities

| Capability                | Description                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------- |
| Local Text Generation     | Generates assistant responses from user prompts without requiring a hosted model API.                |
| Reasoning Support         | Applies structured task instructions for analytical, deterministic, or self-checked responses.       |
| Coding Assistance         | Generates, reviews, debugs, explains, and refactors Python, C#, SQL, VBA, JavaScript, and Markdown.  |
| Summarization             | Condenses user-provided or document-derived content while preserving key facts.                      |
| Extraction                | Extracts requested facts, entities, fields, and structured values from supplied context.             |
| Translation               | Translates content into the selected target language while preserving meaning and tone.              |
| Document Q&A              | Answers questions from uploaded documents using retrieved excerpts and grounding controls.           |
| Semantic Context Reuse    | Uses selected semantic-search chunks as context for later text-generation or document-Q&A workflows. |
| Prompt Template Execution | Applies saved system prompts and task templates from the local prompt-management workflow.           |
| Offline Operation         | Runs locally after the model file and Python dependencies are installed.                             |

## 🧩 Supported Runtime Inputs

| Input               | Used By           | Purpose                                                                              |
| ------------------- | ----------------- | ------------------------------------------------------------------------------------ |
| User Prompt         | Text Generation   | Main request submitted to the local model.                                           |
| System Instructions | Prompt Builder    | Defines the assistant role, constraints, response style, and task behavior.          |
| Task Preset         | Prompt Builder    | Selects chat, reasoning, coding, translation, summarization, or extraction behavior. |
| Chat History        | Prompt Builder    | Supplies previous conversation turns when conversation history is enabled.           |
| Document Context    | Prompt Builder    | Adds uploaded document excerpts or selected chunks to the model prompt.              |
| Semantic Context    | Prompt Builder    | Adds similarity-ranked chunks from semantic search to the prompt.                    |
| Inference Settings  | llama.cpp Runtime | Controls sampling behavior, context size, output length, and CPU utilization.        |

## 🎛️ Runtime Controls

| Control           | Purpose                                                  | Typical Starting Value                |
| ----------------- | -------------------------------------------------------- | ------------------------------------- |
| Context Window    | Controls the maximum prompt context passed to the model. | `4096` or `8192`                      |
| CPU Threads       | Controls how many CPU threads llama.cpp uses.            | Physical core count or slightly below |
| Max Tokens        | Controls the maximum generated response length.          | `1024`                                |
| Temperature       | Controls output randomness.                              | `0.0` to `0.3`                        |
| Top-P             | Controls nucleus sampling.                               | `0.90` to `0.95`                      |
| Top-K             | Limits token candidate selection.                        | `40` or `50`                          |
| Repeat Window     | Defines the token window used for repetition control.    | `256` to `512`                        |
| Repeat Penalty    | Penalizes repeated text.                                 | `1.05` to `1.15`                      |
| Presence Penalty  | Encourages new topics or terms.                          | `0.0` to `0.3`                        |
| Frequency Penalty | Reduces repeated token frequency.                        | `0.0` to `0.3`                        |

## 💻 Recommended Hardware

| Hardware Tier          | Recommended Model Size   | Notes                                                               |
| ---------------------- | ------------------------ | ------------------------------------------------------------------- |
| Low-memory CPU laptop  | 1B to 3B GGUF            | Best for basic chat, coding help, summaries, and extraction.        |
| 16 GB RAM CPU laptop   | 1B to 7B GGUF            | Practical for local-first use with moderate context windows.        |
| 32 GB RAM workstation  | 7B to 13B GGUF           | Better reasoning, coding, and document workflows.                   |
| NVIDIA GPU workstation | 7B+ GGUF with CUDA build | Faster generation when CUDA-enabled llama.cpp wheels are installed. |
| AMD GPU workstation    | Vulkan or HIP build      | Hardware acceleration depends on driver and backend support.        |

## 📦 Model Format

Bro expects a local GGUF model file.

| Format   | Description                                        |
| -------- | -------------------------------------------------- |
| `.gguf`  | Quantized llama.cpp-compatible model format.       |
| `Q4_K_M` | Common balanced quantization for size and quality. |
| `Q5_K_M` | Higher quality with more memory use.               |
| `Q8_0`   | Higher precision and larger file size.             |

## ✅ Recommended Model Selection

| Use Case                  | Recommended Model Type                                           |
| ------------------------- | ---------------------------------------------------------------- |
| Fast local chat           | 1B to 3B instruct GGUF                                           |
| Better general assistance | 7B instruct GGUF                                                 |
| Coding-heavy workflows    | Code-focused instruct GGUF                                       |
| Document Q&A              | Instruct model with strong summarization and extraction behavior |
| CPU-only Windows laptop   | Small or medium quantized GGUF, preferably `Q4_K_M`              |

## 🧪 Installation Overview

The LLM setup has four parts:

1. Create and activate the Python virtual environment.
2. Install Bro’s Python dependencies.
3. Download a GGUF model file.
4. Point Bro’s configuration to the downloaded model file.

## 🪟 Windows Installation

### 1. Open PowerShell in the Bro Repository

```powershell
cd C:\Users\terry\source\repos\Bro
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Upgrade Packaging Tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 5. Install Core Runtime Dependencies

```powershell
pip install streamlit pandas numpy plotly pymupdf sentence-transformers huggingface_hub
```

### 6. Install llama-cpp-python for CPU

```powershell
pip install llama-cpp-python
```

### 7. Verify llama-cpp-python Imports Correctly

```powershell
python -c "from llama_cpp import Llama; print('llama-cpp-python ok')"
```

## 🚀 Optional GPU Installation

### NVIDIA CUDA Wheel Example

Use a CUDA wheel only when the installed NVIDIA driver and CUDA runtime match the wheel target.

```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### Vulkan Wheel Example

Use Vulkan when CUDA is not available and the graphics driver supports Vulkan acceleration.

```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan
```

### Reinstall After a Failed Build

```powershell
pip uninstall llama-cpp-python -y
pip cache purge
pip install llama-cpp-python
```

## 📥 Downloading a GGUF Model

### Option A: Download from the Hugging Face Website

1. Open the Hugging Face model repository for the GGUF model.
2. Select the `Files and versions` tab.
3. Choose a `.gguf` file.
4. Download the file locally.
5. Place the file under a local model directory.

Recommended local directory:

```text
C:\Users\terry\source\models\bro\
```

Example final path:

```text
C:\Users\terry\source\models\bro\bro-model.Q4_K_M.gguf
```

### Option B: Download with the Hugging Face CLI

Install the Hugging Face Hub client if it is not already installed.

```powershell
pip install huggingface_hub
```

Download a specific GGUF file.

```powershell
hf download <model-repository> <model-file.gguf> --local-dir C:\Users\terry\source\models\bro
```

Example pattern:

```powershell
hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir C:\Users\terry\source\models\bro
```

Use the actual repository name and GGUF filename selected for Bro.

## 🗂️ Recommended Model Directory Layout

```text
C:\Users\terry\source\models\
└── bro\
    └── bro-model.Q4_K_M.gguf
```

## 🔧 Configure Bro to Use the Model

Bro loads the local model from the path configured in `config.py`.

### Option A: Environment Variable

Set a persistent Windows user environment variable.

```powershell
[Environment]::SetEnvironmentVariable(
  "BRO_LLM_PATH",
  "C:\Users\terry\source\models\bro\bro-model.Q4_K_M.gguf",
  "User"
)
```

Close and reopen PowerShell after setting the variable.

Verify the value.

```powershell
echo $env:BRO_LLM_PATH
```

### Option B: Direct config.py Path

Use this pattern in `config.py` when the project is configured to read from the environment first
and fall back to a default value.

```python
MODEL_PATH = os.getenv(
    "BRO_LLM_PATH",
    r"C:\Users\terry\source\models\bro\bro-model.Q4_K_M.gguf"
)
```

## ✅ Verify the Model File Exists

Run this from the Bro repository.

```powershell
python -c "from pathlib import Path; import config as cfg; print(cfg.MODEL_PATH); print(Path(cfg.MODEL_PATH).exists())"
```

Expected result:

```text
C:\Users\terry\source\models\bro\bro-model.Q4_K_M.gguf
True
```

If the result is `False`, the configured path does not match the downloaded model location.

## ▶️ Run Bro

```powershell
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

Typical local URL:

```text
http://localhost:8501
```

## 🧪 LLM Smoke Test

Use Text Generation mode and submit:

```text
Respond with one sentence confirming the local Bro model is running.
```

Expected behavior:

| Result             | Meaning                                                        |
| ------------------ | -------------------------------------------------------------- |
| Response appears   | The model loaded and generated successfully.                   |
| No response        | Check terminal errors and model path.                          |
| Import error       | Reinstall `llama-cpp-python`.                                  |
| Model path warning | Confirm `BRO_LLM_PATH` or `MODEL_PATH`.                        |
| Very slow response | Reduce model size, reduce context window, or lower max tokens. |

## 🛠️ Troubleshooting

| Problem                          | Likely Cause                                                         | Fix                                                                                |
| -------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `ModuleNotFoundError: llama_cpp` | `llama-cpp-python` is not installed in the active environment.       | Activate `.venv` and run `pip install llama-cpp-python`.                           |
| Model does not load              | `MODEL_PATH` points to a missing file.                               | Verify the `.gguf` path and environment variable.                                  |
| App starts but generation fails  | Model file is incompatible, corrupted, or too large for memory.      | Try a smaller quantized GGUF file.                                                 |
| Very slow generation             | CPU-only inference with a large model.                               | Use a smaller model, reduce context size, or install GPU acceleration.             |
| Out-of-memory error              | Model size or context window exceeds available RAM.                  | Use `Q4_K_M`, reduce context window, or select a smaller model.                    |
| PDF Q&A works poorly             | Extracted text is incomplete or the model context is too small.      | Enable native PDF text extraction, reduce chunk size, or increase retrieval count. |
| Semantic search unavailable      | Sentence-transformer dependency failed or embeddings were not built. | Reinstall `sentence-transformers` and rebuild the semantic index.                  |

## ⚖️ Model Selection Guidance

| Priority              | Recommended Choice                                            |
| --------------------- | ------------------------------------------------------------- |
| Maximum speed         | Small instruct model, 1B to 3B, `Q4_K_M`.                     |
| Balanced quality      | 7B instruct model, `Q4_K_M` or `Q5_K_M`.                      |
| Better coding         | Code-specialized instruct GGUF.                               |
| Better summaries      | General instruct model with strong summarization performance. |
| Lower memory use      | Smaller parameter count and lower quantization size.          |
| Higher answer quality | Larger model, higher quantization, and larger context window. |

## 🔐 Local-First Operating Notes

| Area                | Behavior                                                          |
| ------------------- | ----------------------------------------------------------------- |
| Model Execution     | Runs locally through llama.cpp after installation.                |
| Model File          | Remains on the local filesystem.                                  |
| Chat History        | Stored locally in SQLite when chat persistence is enabled.        |
| Prompt Templates    | Stored locally in SQLite.                                         |
| Document Context    | Extracted and processed locally.                                  |
| Semantic Index      | Stored locally through SQLite-backed embedding tables.            |
| Network Requirement | Not required after dependencies and the GGUF model are installed. |

## 🧭 Recommended Setup Sequence

1. Install Python and create the `.venv`.
2. Install Bro dependencies.
3. Install `llama-cpp-python`.
4. Download a small GGUF model first.
5. Set `BRO_LLM_PATH`.
6. Verify the configured model path returns `True`.
7. Start Bro with `streamlit run app.py`.
8. Run a short Text Generation smoke test.
9. Increase model size or context window only after the small model runs reliably.
10. Document the selected model name, quantization, file path, and hardware assumptions in the
    project README.

## 🔗 Related Pages

| Page                                        | Description                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| [Architecture](architecture.md)             | Application layers, runtime flow, and module relationships.             |
| [Text Generation](text-generation.md)       | Prompt construction, inference controls, and local response generation. |
| [Document Q&A](document-qna.md)             | Grounded document workflows and retrieval behavior.                     |
| [Semantic Search](semantic-search.md)       | Embedding-backed search and semantic context reuse.                     |
| [Prompt Engineering](prompt-engineering.md) | Prompt templates, metadata, and workflow routing.                       |
| [Data Management](data-management.md)       | SQLite inspection, profiling, and administration workflows.             |
| [Development](development.md)               | Local setup, validation, documentation builds, and maintenance checks.  |
