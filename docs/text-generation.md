# Text Generation

Text Generation is Bro's primary local Gemma chat and text-task interface.

## Supported tasks

The Task Type selectbox exposes the same canonical task list consumed by the instruction builder:

- Chat
- Analysis
- Reasoning
- Coding
- Writing
- Editing
- Translation
- Summarization
- Extraction
- Classification
- Comparison
- Structured Output

This avoids maintaining a UI subset that disagrees with the execution path.

## 🧭 Task Preset

One row contains:

| Control | Purpose |
| --- | --- |
| Task Type | Selects the instruction branch. |
| Task Detail | Concise, Standard, or Detailed guidance. |
| Task Focus | Accuracy, Balanced, or Creativity focus. |

The bottom `🔄 Reset` resets all Task Preset values.

## 🧩 Reasoning Controls

- Reasoning Depth
- Answer Only
- Use Self-Check
- Prefer Deterministic Reasoning

These are prompt-strategy controls; they are not presented as a separate native Gemma thinking API.

## 🧾 Coding Controls

One row contains:

- Code Language
- Coding Task
- Include Comments
- Use Editor Format
- Emit Fenced Code

### Coding tasks

- Generate
- Complete
- Refactor
- Debug
- Review
- Explain
- Optimize
- Convert
- Test
- Document
- Design

### Coding languages

Bro uses a bounded language selector including Python, C, C++, C#, Java, JavaScript, TypeScript,
SQL, VBA, PowerShell, Bash, HTML, CSS, Markdown, JSON, YAML, and Other.

## ✍️ Writing Controls

- Writing Task
- Tone
- Audience
- Length

These settings apply only to writing/editing instruction construction.

## 🌐 Translation Controls

Translation has its own expander rather than sharing Coding Controls.

- Source Language
- Target Language
- Translation Mode
- Preserve Formatting

Translation modes are:

- Natural
- Literal
- Formal
- Technical
- Localization

Language arguments are bounded selectbox values. The source selector includes `Auto Detect`.

## 🏷️ Classification Controls

- Classification Type
- Return Confidence
- Allow Unknown
- Explain Classification

Classification types include Binary, Multi-Class, Multi-Label, Sentiment, Intent, Topic, and
Relevance.

## ↔️ Response Controls

Response Controls describe the output contract rather than the sampling algorithm:

- Response Format
- Response Language
- Response Length
- Include Headings

Supported response formats are:

- Plain Text
- Markdown
- Bullet List
- Numbered List
- Markdown Table
- JSON
- XML
- YAML
- CSV
- Code

JSON requests can use llama.cpp's JSON response-format constraint.

## 🎚️ Inference Settings

Inference Settings map to generation/runtime arguments.

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

The single bottom Reset resets both rows.

## 🎛️ Context Controls

- Context Window
- Use Conversation History
- Use Document Context
- Use Semantic Context
- Use Grounding

These settings affect context/message construction, not response formatting.

## ⚙️ Runtime Settings

- CPU Threads
- Batch Size
- Micro Batch Size

## 🖥️ System Instructions

System Instructions provide:

- Category selectbox;
- category-filtered template selectbox;
- ID-backed template loading;
- editable system-instruction text;
- clear/reset;
- XML ↔ Markdown conversion;
- Apply Preset;
- Effective Prompt Preview.

Template captions are display values; database identity is the integer `Prompts.ID`.

## Generation pipeline

```text
task controls
   -> build_task_instruction_block()
   -> build_chat_messages()
   -> get_runtime_llm()
   -> create_chat_completion()
   -> stream/render response
```
