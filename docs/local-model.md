# Local Model

Bro's current local inference target is **Gemma 3 4B IT in GGUF format**, executed through
`llama-cpp-python`.

## Text model

The configured model is read from:

```python
cfg.MODEL_PATH
```

Bro checks that this file exists before loading the llama.cpp runtime.

A typical environment-driven configuration may resolve from:

```powershell
$env:BRO_LLM_PATH="C:\models\gemma-3-4b-it-Q4_K_M.gguf"
```

## Chat serialization

Bro uses `Llama.create_chat_completion()` rather than manually assembling foreign role tokens.
The GGUF/runtime chat template therefore owns the model-specific serialization contract.

## Gemma 3 modalities

| Modality | Supported by Gemma 3 4B IT | Bro exposure |
| --- | ---: | --- |
| Text input | Yes | Text Generation, Document Q&A |
| Image input | Yes | Image to Text, PDF vision OCR |
| Text output | Yes | All model-facing modes |
| Image output | No | Not exposed |
| Audio input | No | Not exposed |
| Audio output | No | Not exposed |
| Native transcription | No | Not exposed |

## Multimodal projector

Image understanding requires a **matching Gemma 3 multimodal projector GGUF**.

Bro resolves the projector from:

1. `cfg.MMPROJ_PATH`;
2. `cfg.MM_PROJ_PATH`;
3. `BRO_MMPROJ_PATH`;
4. `GEMMA_MMPROJ_PATH`;
5. an `mmproj*.gguf` file beside `cfg.MODEL_PATH`.

Example:

```powershell
$env:BRO_MMPROJ_PATH="C:\models\mmproj-gemma-3-4b-it-f16.gguf"
```

When the projector cannot be resolved, text generation remains available while Image to Text reports
the missing multimodal dependency.

## Runtime controls

Text Generation exposes:

- CPU Threads;
- Batch Size;
- Micro Batch Size.

Image to Text additionally exposes:

- Context Window;
- Projector Device (`CPU` or `GPU`).

The selected values are used by model/runtime initialization rather than being cosmetic UI state.

## Context and output limits

Bro exposes a context control up to **131,072 tokens** and a maximum-generation control up to
**8,192 tokens**. These are upper UI/model limits; practical capacity depends on quantization,
available RAM/VRAM, the llama.cpp build, and the selected context size.

## Optional embedding runtime

Document Q&A and Semantic Search use a separate local embedding model:

```text
all-MiniLM-L6-v2
```

This is independent of Gemma generation and is loaded through `sentence-transformers`.

## Prompt-category capability boundary

Model-facing prompt categories must reflect the capabilities of the loaded Gemma/runtime combination.
Legacy categories stored in SQLite are not automatically surfaced. Audio/transcription and image-
generation categories are excluded for Gemma 3 4B IT, while vision categories are exposed only when
the multimodal projector runtime is usable.
