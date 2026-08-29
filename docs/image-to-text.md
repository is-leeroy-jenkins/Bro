# Image to Text

Image to Text is a first-class Bro mode and the priority use of Gemma 3's multimodal capability.

It accepts image input and returns text. It does **not** generate images.

## Requirements

Image analysis requires both:

1. the configured Gemma 3 4B IT GGUF; and
2. a matching `mmproj` multimodal projector GGUF.

See [Local Model](local-model.md) for projector discovery and configuration.

## Supported image types

- PNG
- JPG
- JPEG
- WEBP

Multiple images can be uploaded when the selected task requires comparison.

## 👁️ Vision Controls

A single six-control row provides:

| Control | Purpose |
| --- | --- |
| Vision Task | Selects the image-understanding operation. |
| Image Detail | Controls response depth. |
| Response Format | Uses Bro's standard bounded output formats. |
| Response Language | Uses the shared bounded human-language list. |
| Preserve Layout | Requests preservation of visible document/layout structure. |
| Include Visible Text | Requests explicit transcription of visible text. |

### Vision tasks

- Extract Visible Text
- Describe Image
- Answer Questions
- Analyze Screenshot
- Analyze Chart
- Analyze Diagram
- Extract Structured Data
- Compare Images

`Extract Visible Text` is the primary OCR/Image-to-Text workflow.

## 🎚️ Inference Settings

Vision uses the same generation controls as text generation:

### Row 1
- Temperature
- Top-P
- Top-K
- Repeat Penalty
- Repeat Window

### Row 2
- Presence Penalty
- Frequency Penalty
- Random Seed
- Max Tokens

One bottom Reset resets both rows.

## ⚙️ Runtime Settings

- Context Window
- CPU Threads
- Batch Size
- Micro Batch Size
- Projector Device

`Projector Device` is a bounded `CPU` / `GPU` selectbox.

## System Instructions

Image to Text retains the standard `🖥️ System Instructions` surface so a saved vision-oriented
prompt can be applied without creating a separate prompt-storage mechanism.

## Request flow

```text
upload image(s)
    |
    +-- select Vision Task
    |
    +-- optional Image Request
    |
    +-- build vision message
    |
    +-- MTMDChatHandler + mmproj
    |
    +-- Gemma 3
    |
    +-- streamed text response
```

## Capability gating

If the multimodal projector is missing or incompatible, Bro warns the user instead of attempting to
send images to a text-only runtime.

## Relationship to Document Q&A

Document Q&A can reuse this vision pipeline when OCR is enabled and a PDF page lacks usable native
text. This makes Image-to-Text a shared runtime capability rather than an isolated demonstration.
