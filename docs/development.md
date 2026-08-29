# Development

This page summarizes the contracts that should be preserved when extending Bro.

## Source-of-truth rule

A new UI capability is complete only when the full execution chain exists:

```text
UI control
  -> session state
  -> instruction/context/runtime builder
  -> model or retrieval call
  -> response handling
  -> UI output
```

Do not expose controls that terminate in session state without affecting execution.

## UI organization

For model-control expanders:

- each control row contains **3–6 controls**;
- an expander may contain multiple rows;
- one full-width `🔄 Reset` sits at the bottom;
- that Reset owns all controls in that expander and does not reset unrelated control groups.

Finite option sets should use selectboxes, multiselects, radio controls, sliders, or toggles rather
than arbitrary text input.

## Model-control ownership

| UI group | Implementation owner |
| --- | --- |
| Task controls | task/prompt instruction builder |
| Response controls | response/output contract |
| Context controls | context/message builder |
| Inference settings | `create_chat_completion()` arguments |
| Runtime settings | `Llama(...)` / multimodal runtime initialization |
| Vision controls | vision instruction/message construction |

## Gemma multimodal development

Do not enable image controls merely because the underlying Gemma model supports images. The effective
vision capability requires:

```text
model capability
    AND llama.cpp multimodal support
    AND matching mmproj available
```

All three must be true.

## Runtime-error acceptance criterion

Every user-reachable model execution path should be tested for:

- valid default widget/session state;
- model unavailable;
- projector unavailable where required;
- embedding model unavailable where required;
- empty input;
- malformed optional data;
- streaming and non-streaming response handling;
- reset/rerun behavior;
- stale dependent selectbox state;
- retrieval backend fallback.

## Documentation comments

Generated Python functions should retain project-style documentation comments with Purpose, Args, and
Returns sections so source documentation can be reused by MkDocs/mkdocstrings where import safety
permits.

## MkDocs import safety

`app.py` is a Streamlit application with top-level UI execution. Importing it during documentation
generation can execute page code and require runtime-only dependencies.

For that reason, this documentation package uses a **manual App API map** instead of placing
`::: app` in the API page.

Import-safe modules such as `config` and `boogr` may continue to use mkdocstrings.

## Build

```bash
python -m mkdocs build --strict
```

Serve locally with:

```bash
python -m mkdocs serve
```

Run the strict build before publishing to GitHub Pages.
