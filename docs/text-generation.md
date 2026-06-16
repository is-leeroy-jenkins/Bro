# Text Generation

Text Generation is Bro's primary local model interaction mode. It builds a prompt from user input, optional system instructions, task-specific settings, chat history, semantic context, and document context, then routes the prompt to the configured GGUF model through `llama-cpp-python`.

## 🧭 Purpose

This mode supports general chat, reasoning, coding, translation, summarization, and extraction workflows. It is designed for local workstation execution where the user controls inference settings and prompt composition directly.

## 🧱 Workflow Position

```text
User request
  ▼
System instructions
  ▼
Task preset and response controls
  ▼
Optional chat history
  ▼
Optional document or semantic context
  ▼
Prompt builder
  ▼
Local GGUF model
  ▼
Assistant response
```

## ⚙️ Main Controls

| Control Group | Description |
| --- | --- |
| Task Preset | Selects Chat, Reasoning, Coding, Translation, Summarization, or Extraction behavior. |
| Response Format | Shapes model output as Markdown, plain text, JSON, or another supported format. |
| Conversation Context | Determines whether prior messages are included in the prompt. |
| Document Context | Includes shared document or semantic context when enabled. |
| Reasoning Controls | Adjusts answer directness, self-check behavior, and deterministic reasoning preference. |
| Coding Controls | Sets language, task type, comment inclusion, editor-ready formatting, and fenced-code behavior. |
| Runtime Controls | Sets context window, CPU thread count, maximum tokens, temperature, top-p, top-k, and repeat penalty. |
| System Instructions | Supplies high-priority behavior instructions for the local model. |

## 🧠 Prompt Construction

Bro builds the effective prompt from several possible inputs.

```text
[System Instructions]
  + [Task Instruction Block]
  + [Semantic Context when enabled]
  + [Document Context when enabled]
  + [Chat History when enabled]
  + [Current User Message]
```

This approach keeps prompt composition explicit and inspectable. The effective prompt preview is useful when troubleshooting unexpected model behavior.

## 🧪 Example Workflows

### General Chat

Use this workflow for normal assistant behavior.

1. Select **Text Generation**.
2. Set **Task Preset** to `Chat`.
3. Enter optional system instructions.
4. Submit a user message.

### Reasoning

Use this workflow for structured analysis.

1. Set **Task Preset** to `Reasoning`.
2. Choose a reasoning depth.
3. Enable deterministic reasoning for repeatable analytical answers.
4. Enable self-check when the answer must be verified before output.

### Coding

Use this workflow for editor-ready source output.

1. Set **Task Preset** to `Coding`.
2. Select the target language.
3. Choose the coding task.
4. Enable editor-ready output.
5. Enable or disable fenced-code output depending on copy/paste needs.

### Translation

Use this workflow for language conversion.

1. Set **Task Preset** to `Translation`.
2. Select the target language.
3. Provide the source text.
4. Keep temperature low for consistent translations.

### Summarization

Use this workflow for concise condensation.

1. Set **Task Preset** to `Summarization`.
2. Select the desired response format.
3. Paste text or enable document context.
4. Ask for the specific summary style needed.

### Extraction

Use this workflow when structured facts are needed.

1. Set **Task Preset** to `Extraction`.
2. Select JSON when machine-readable output is required.
3. Provide the source content or route relevant semantic chunks.
4. Ask for exact fields and missing-value behavior.

## 🔧 Runtime Settings

| Setting | Practical Guidance |
| --- | --- |
| Context Window | Increase for longer prompts and document context, within local model limits. |
| CPU Threads | Use a value appropriate for the workstation CPU. |
| Max Tokens | Set high enough for the expected answer, but low enough to control latency. |
| Temperature | Use lower values for factual, deterministic, or structured outputs. |
| Top-P | Controls nucleus sampling and output diversity. |
| Top-K | Limits candidate tokens and can reduce drift. |
| Repeat Penalty | Reduces repeated phrases and looping. |
| Random Seed | Use a fixed value when reproducibility matters. |

## 🧯 Failure Handling

Text Generation depends on local model availability. If the configured GGUF model is missing or cannot be loaded, Bro should fail safely and preserve the UI. Logged handlers should record the failure through the project logger without exposing prompt text, file contents, tokens, or secrets in the logged method signature.

## ✅ Recommended Defaults

| Scenario | Suggested Settings |
| --- | --- |
| Federal analysis or policy drafting | Low temperature, Markdown response, chat history enabled. |
| Code generation | Low to moderate temperature, editor-ready output, comments enabled when documentation matters. |
| Structured extraction | Low temperature, JSON response, answer-only enabled. |
| Exploratory brainstorming | Moderate temperature, chat history enabled. |
| Grounded document work | Use Document Q&A instead of plain Text Generation unless selected chunks were routed in. |

## 🔗 Related Pages

- [Document Q&A](document-qna.md)
- [Semantic Search](semantic-search.md)
- [Prompt Engineering](prompt-engineering.md)
- [Data Management](data-management.md)
