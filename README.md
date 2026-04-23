###### Bro
![](https://github.com/is-leeroy-jenkins/Bro/blob/main/resources/images/bro_project.png)



A small application designed to run a LLM  (bro-)locally in GGUF format for fast, reliable instruction following, contextual comprehension, and structured reasoning. It runs entirely on your machine using a fine-tuned version Google's Gemma-3-4B-IT model, with no cloud APIs, no telemetry, and full control over your data.

Bro is lightweight making it ideal for **everyday analysis, drafting, summarization, and reasoning tasks** on CPU-only systems.



## ✨ Key Features

* 🧠 **Gemma-3-4B-IT–GGUF LLM (GGUF, llama.cpp)**
* 🔒 **100% local inference** — no external APIs
* 💬 **Persistent chat history** (SQLite)
* 📄 **Document-based RAG**
* 🔍 **Semantic search with embeddings**
* ⚙️ **Interactive model parameter controls**
* 📊 **Live token & context usage tracking**
* 📝 **Export chat history to Markdown or PDF**
* 🖥️ **CPU-friendly and fast**

## 🧠 LLM Repository

[![](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/bro)

### Sup with Lil Bro?

A fine-tuned variant of Google's **Gemma-3-4B-IT**, optimized for:

* Instruction following
* Contextual understanding
* Concise, structured responses
* Low-latency, local inference

**Model characteristics:**

* ~4B parameters
* GGUF format (`Q4_K_M`)
* Optimized for llama.cpp
* Text-only (no image or audio input)


---

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://bro-py.streamlit.app/)

![](https://github.com/is-leeroy-jenkins/Bro/blob/main/resources/images/Bro-streamlit.gif)


## 🗂 Repository Structure

```
bro/
├─ app.py                     # Main Streamlit application
├─ config.py                  # Environment variable access
├─ requirements.txt           # Python dependencies
├─ resources/
│  └─ images/
│     └─ bro_logo.png
├─ stores/
│  └─ sqlite/
│     └─ bro.db               # Chat + embedding storage
└─ README.md
```



## ⚙️ System Requirements

### Minimum

* **Windows 10/11 (64-bit)**
* **Python 3.10 or 3.11**
* **8 GB RAM** (16 GB recommended)
* Modern CPU with AVX2 support
* ~5–7 GB free disk space

Bro is designed to run comfortably on **most modern laptops and desktops**.



## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-org-or-username>/bro.git
cd bro
```

---

### 2️⃣ Create and Activate a Virtual Environment

#### Windows (Git Bash / PowerShell)

```bash
python -m venv .venv
source .venv/Scripts/activate
```

You should see:

```
(.venv)
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```



## 📥 Download the Bro Model

1. Go to the Hugging Face model page:
   👉  [![](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/bro)

2. Download the GGUF file:

   ```
   bro-3-1b-it-Q4_K_M.gguf
   ```

3. Place the file anywhere on your system, for example:

   ```
   C:\Users\<you>\source\llm\lmstudio\lmstudio-community\leeroy-jankins\bro\bro-3-1b-it-Q4_K_M.gguf
   ```



## 🔑 Environment Variable Configuration

Bro locates the model **exclusively via an environment variable**.

### Required Variable

| Variable       | Description                     |
| -------------- | ------------------------------- |
| `BRO_LLM_PATH` | Full path to the Bro GGUF model |

### Example (Windows)

```
BRO_LLM_PATH=C:\Users\terry\source\llm\lmstudio\lmstudio-community\leeroy-jankins\bro\bro-3-1b-it-Q4_K_M.gguf
```

⚠️ **Restart your terminal / IDE (PyCharm) after setting this variable.**



## ▶️ Running Bro

To ensure the correct virtual environment is used, always launch Streamlit via Python:

```bash
python -m streamlit run app.py
```

If configured correctly, the Bro UI will open in your browser.



## 🧭 Application Tabs

| Tab                    | Description                                  |
| ---------------------- | -------------------------------------------- |
| System Instructions    | Define the assistant’s system-level behavior |
| Text Generation        | Primary chat interface                       |
| Retrieval Augmentation | Upload documents for basic RAG               |
| Semantic Search        | Build & query vector embeddings              |
| Export                 | Download chat history (Markdown / PDF)       |



## 📊 Token & Context Monitoring

A persistent footer displays:

* Prompt token count
* Response token count
* Percentage of context window used
* Active model parameters

This makes it easy to tune performance and avoid context overflows.



## 🔒 Privacy & Design Philosophy

* No cloud calls
* No telemetry
* No API keys
* All data stored locally (SQLite + filesystem)
* Deterministic, inspectable inference

Bro is built for **control, transparency, and reliability**.



## 🧬 Related Applications

* **Leeroy** — entry-level instruction assistant
* **Bro** — balanced, fast instruction & reasoning
* **Gipity** — heavyweight deep-reasoning (20B)

Each app targets a different performance / reasoning profile.



## 📜 License

This application is provided for **personal and research use**.
Refer to the Hugging Face model repository for model-specific licensing terms.



## 🙌 Acknowledgements

* llama.cpp community
* Hugging Face
* LM Studio ecosystem
* Open-source Python & ML tooling

