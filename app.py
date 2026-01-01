# ******************************************************************************************
# Assembly:                Bro
# Filename:                app.py
# Author:                  Terry D. Eppler (integration)
# Created:                 01-01-2026
# Last Modified On:        01-01-2026
# ******************************************************************************************

import base64
import io
import multiprocessing
import os
import sqlite3
import textwrap
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from llama_cpp import Llama
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer

# ==============================================================================
# Constants
# ==============================================================================
APP_NAME = "Bro"

HF_MODEL_URL = "https://huggingface.co/leeroy-jankins/bro"
DB_PATH = "stores/sqlite/bro.db"

DEFAULT_CTX = 4096
CPU_CORES = multiprocessing.cpu_count()

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant optimized for instruction following, contextual comprehension, "
    "and structured reasoning. Provide clear, concise, and technically accurate responses."
)

# NOTE:
# - Bro uses BRO_LLM_PATH (not LEEROY_LLM_PATH).
# - If you want to allow fallback to a repo-relative default model path, set DEFAULT_MODEL_PATH.
DEFAULT_MODEL_PATH = "models/Bro-3B-Instruct.Q4_K_M.gguf"
MODEL_PATH = os.getenv("BRO_LLM_PATH") or DEFAULT_MODEL_PATH


# ==============================================================================
# Streamlit Config
# ==============================================================================
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    page_icon=r"resources/images/favicon.ico"
)

# Ensure the fixed footer can never cover content
st.markdown(
    """
    <style>
      .block-container { padding-bottom: 80px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# Guards
# ==============================================================================
MODEL_PATH_OBJ = Path(MODEL_PATH)

if not MODEL_PATH_OBJ.exists():
    st.error(
        "❌ **Model not found**\n\n"
        f"Expected at:\n`{MODEL_PATH}`\n\n"
        "Set **BRO_LLM_PATH** to the full path of your GGUF model.\n\n"
        f"{HF_MODEL_URL}"
    )
    st.stop()


# ==============================================================================
# Utilities
# ==============================================================================
def image_to_base64(path: str) -> str:
    """
    Purpose:
        Loads an image from disk and returns a base64-encoded string for HTML embedding.

    Parameters:
        path: str
            The file path to the image.

    Returns:
        str
            Base64 representation of the file bytes.
    """
    return base64.b64encode(Path(path).read_bytes()).decode()


def chunk_text(text: str, size: int = 1200, overlap: int = 250) -> List[str]:
    """
    Purpose:
        Splits text into overlapping chunks for basic and semantic retrieval augmentation.

    Parameters:
        text: str
            The raw text to chunk.
        size: int
            Chunk size in characters.
        overlap: int
            Overlap size in characters.

    Returns:
        List[str]
            A list of chunk strings.
    """
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += max(1, (size - overlap))
    return chunks


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Purpose:
        Computes cosine similarity between two vectors.

    Parameters:
        a: np.ndarray
            First vector.
        b: np.ndarray
            Second vector.

    Returns:
        float
            Cosine similarity in [-1, 1].
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ==============================================================================
# Database
# ==============================================================================
def ensure_db() -> None:
    """
    Purpose:
        Ensures the SQLite database and required tables exist.

    Parameters:
        None

    Returns:
        None
    """
    Path("stores/sqlite").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk TEXT,
                vector BLOB
            )
            """
        )


def save_message(role: str, content: str) -> None:
    """
    Purpose:
        Persists a chat message to SQLite.

    Parameters:
        role: str
            'user' or 'assistant' (and potentially others if you extend).
        content: str
            The message content.

    Returns:
        None
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_history (role, content) VALUES (?, ?)",
            (role, content)
        )


def load_history() -> List[Tuple[str, str]]:
    """
    Purpose:
        Loads chat history from SQLite in insertion order.

    Parameters:
        None

    Returns:
        List[Tuple[str, str]]
            List of (role, content).
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_history ORDER BY id"
        ).fetchall()
    return [(r, c) for r, c in rows]


# ==============================================================================
# Loaders
# ==============================================================================
@st.cache_resource
def load_llm(ctx: int, threads: int) -> Llama:
    """
    Purpose:
        Loads and caches the Llama model for the current runtime.

    Parameters:
        ctx: int
            Context window size (n_ctx).
        threads: int
            Number of CPU threads.

    Returns:
        Llama
            The loaded llama-cpp-python model.
    """
    return Llama(
        model_path=str(MODEL_PATH_OBJ),
        n_ctx=ctx,
        n_threads=threads,
        n_batch=256,
        verbose=False
    )


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """
    Purpose:
        Loads and caches the sentence transformer embedder.

    Parameters:
        None

    Returns:
        SentenceTransformer
            Embedding model instance.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================================
# Sidebar (parameters only; Leeroy parity)
# ==============================================================================
with st.sidebar:
    try:
        logo_b64 = image_to_base64("resources/images/bro_logo.png")
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; margin-bottom:10px;">
                <img src="data:image/png;base64,{logo_b64}" style="width:60px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        # Do not break the app if branding assets are missing
        st.write(APP_NAME)

    st.header("⚙️ Mind Controls")
    ctx = st.slider("Context Window", 2048, 8192, DEFAULT_CTX, 512)
    threads = st.slider("CPU Threads", 1, CPU_CORES, max(2, CPU_CORES // 2))
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 40, 10)
    repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.1, 0.05)

    # Optional: prompt inspection toggle (does not change functionality; just visibility)
    show_prompt_debug = st.checkbox("Show Prompt Debug", value=False)


# ==============================================================================
# Init
# ==============================================================================
ensure_db()
llm = load_llm(ctx, threads)
embedder = load_embedder()

if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

if "last_valid_system_prompt" not in st.session_state:
    st.session_state.last_valid_system_prompt = DEFAULT_SYSTEM_PROMPT

if "basic_docs" not in st.session_state:
    st.session_state.basic_docs = []

if "use_semantic" not in st.session_state:
    st.session_state.use_semantic = False

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "prompt": 0,
        "response": 0,
        "context_pct": 0.0
    }


# ==============================================================================
# Prompt Builder (Leeroy parity: deterministic ordering + semantic/basic injection)
# ==============================================================================
def build_prompt(user_input: str) -> Tuple[str, dict]:
    """
    Purpose:
        Builds a llama-chat formatted prompt with optional semantic and document context,
        preserving deterministic assembly order.

    Parameters:
        user_input: str
            The latest user message.

    Returns:
        Tuple[str, dict]
            (prompt_text, debug_sections)
    """
    debug = {
        "system_prompt": st.session_state.system_prompt,
        "semantic_chunks": [],
        "document_chunks": [],
        "history_count": len(st.session_state.messages),
        "user_input": user_input,
    }

    prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"

    # Semantic Context (optional)
    if st.session_state.use_semantic:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT chunk, vector FROM embeddings").fetchall()

        if rows:
            q_vec = embedder.encode([user_input])[0].astype(np.float32)

            scored: List[Tuple[str, float]] = []
            for chunk, vec_blob in rows:
                # IMPORTANT: vectors stored as float32 bytes
                v = np.frombuffer(vec_blob, dtype=np.float32)
                scored.append((chunk, cosine_sim(q_vec, v)))

            top_chunks = [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]]

            debug["semantic_chunks"] = top_chunks

            prompt += "<|system|>\nSemantic Context:\n"
            for c in top_chunks:
                prompt += f"- {c}\n"
            prompt += "</s>\n"

    # Basic Document Context (optional)
    if st.session_state.basic_docs:
        doc_chunks = st.session_state.basic_docs[:6]
        debug["document_chunks"] = doc_chunks

        prompt += "<|system|>\nDocument Context:\n"
        for d in doc_chunks:
            prompt += f"- {d}\n"
        prompt += "</s>\n"

    # Conversation history
    for role, content in st.session_state.messages:
        prompt += f"<|{role}|>\n{content}\n</s>\n"

    # Current user turn + assistant open
    prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
    return prompt, debug


# ==============================================================================
# Tabs (Leeroy parity)
# ==============================================================================
tab_system, tab_chat, tab_basic, tab_semantic, tab_export = st.tabs(
    [
        "System Instructions",
        "Text Generation",
        "Retrieval Augmentation",
        "Semantic Search",
        "Export",
    ]
)

# ==============================================================================
# Tab 1 — System Instructions (Leeroy parity)
# ==============================================================================
with tab_system:
    st.subheader("System Prompt")

    new_prompt = st.text_area(
        "Edit the system instructions used for all responses:",
        value=st.session_state.system_prompt,
        height=240
    )

    if new_prompt.strip():
        st.session_state.system_prompt = new_prompt
        st.session_state.last_valid_system_prompt = new_prompt
    else:
        st.warning("System prompt cannot be empty. Reverting to last valid prompt.")
        st.session_state.system_prompt = st.session_state.last_valid_system_prompt

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Reset to Default"):
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
            st.session_state.last_valid_system_prompt = DEFAULT_SYSTEM_PROMPT
            st.success("System prompt reset to default.")

    with col2:
        st.download_button(
            "Download Prompt",
            st.session_state.system_prompt,
            file_name="bro_system_prompt.txt",
            mime="text/plain"
        )

    with col3:
        uploaded = st.file_uploader(
            "Load Prompt",
            type=["txt"],
            label_visibility="collapsed"
        )
        if uploaded:
            text = uploaded.read().decode(errors="ignore").strip()
            if text:
                st.session_state.system_prompt = text
                st.session_state.last_valid_system_prompt = text
                st.success("System prompt loaded.")
            else:
                st.error("Uploaded prompt file was empty.")

# ==============================================================================
# Tab 2 — Text Generation (Leeroy parity: fixed streaming + token usage)
# ==============================================================================
with tab_chat:
    # Render history
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask Bro...")

    if user_input:
        # Persist user message immediately
        save_message("user", user_input)
        st.session_state.messages.append(("user", user_input))

        prompt, dbg = build_prompt(user_input)

        if show_prompt_debug:
            with st.expander("Prompt Debug (effective sections)"):
                st.markdown("### System Prompt")
                st.code(dbg["system_prompt"])
                st.markdown("### Semantic Chunks")
                st.write(dbg["semantic_chunks"])
                st.markdown("### Document Chunks")
                st.write(dbg["document_chunks"])
                st.markdown(f"### History Count: {dbg['history_count']}")

            with st.expander("Raw Prompt (full)"):
                st.code(prompt)

        # Stream assistant response correctly (single placeholder)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""

            for chunk in llm(
                prompt,
                stream=True,
                max_tokens=1024,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=["</s>"]
            ):
                response += chunk["choices"][0]["text"]
                placeholder.markdown(response)

        # Persist assistant after completion
        save_message("assistant", response)
        st.session_state.messages.append(("assistant", response))

        # Token telemetry (Leeroy parity)
        prompt_tokens = len(llm.tokenize(prompt.encode()))
        response_tokens = len(llm.tokenize(response.encode()))
        context_pct = (prompt_tokens + response_tokens) / float(ctx) * 100.0

        st.session_state.token_usage = {
            "prompt": prompt_tokens,
            "response": response_tokens,
            "context_pct": context_pct
        }

# ==============================================================================
# Tab 3 — Retrieval Augmentation (Leeroy parity: deterministic rebuild + feedback)
# ==============================================================================
with tab_basic:
    st.subheader("Basic Retrieval Augmentation")
    st.caption(
        "Upload documents to inject **non-semantic document context** directly into the prompt "
        "during generation."
    )

    uploads = st.file_uploader(
        "Upload TXT / MD / PDF files",
        accept_multiple_files=True
    )

    if uploads:
        st.session_state.basic_docs.clear()

        total_files = len(uploads)
        total_chunks = 0

        for f in uploads:
            text = f.read().decode(errors="ignore")
            chunks = chunk_text(text)
            st.session_state.basic_docs.extend(chunks)
            total_chunks += len(chunks)

        st.success(f"Loaded {total_files} file(s) → {total_chunks} total document chunks.")

# ==============================================================================
# Tab 4 — Semantic Search (Leeroy parity: rebuild, toggle, feedback; float32-safe)
# ==============================================================================
with tab_semantic:
    st.subheader("Semantic Search Index")
    st.caption(
        "Build a semantic embedding index from uploaded documents. When enabled, the most "
        "relevant chunks are injected into the prompt."
    )

    st.session_state.use_semantic = st.checkbox(
        "Use Semantic Context in Text Generation",
        value=st.session_state.use_semantic
    )

    uploads = st.file_uploader(
        "Upload documents to build / rebuild the semantic index",
        accept_multiple_files=True
    )

    if uploads:
        chunks: List[str] = []
        file_count = len(uploads)

        for f in uploads:
            text = f.read().decode(errors="ignore")
            chunks.extend(chunk_text(text))

        if not chunks:
            st.warning("No text content found in uploaded files.")
        else:
            vectors = embedder.encode(chunks)
            vectors = np.asarray(vectors, dtype=np.float32)

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM embeddings")
                for c, v in zip(chunks, vectors):
                    conn.execute(
                        "INSERT INTO embeddings (chunk, vector) VALUES (?, ?)",
                        (c, v.tobytes())
                    )

            st.success(
                f"Semantic index rebuilt: {file_count} file(s), {len(chunks)} embedded chunk(s)."
            )

    # Optional: preview top semantic matches for a query (non-destructive)
    if st.session_state.use_semantic:
        st.markdown("### Semantic Preview")
        preview_q = st.text_input("Preview query (does not send to model):", value="")
        if preview_q.strip():
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("SELECT chunk, vector FROM embeddings").fetchall()

            if not rows:
                st.info("No embeddings in index. Upload documents above to build the index.")
            else:
                q_vec = embedder.encode([preview_q])[0].astype(np.float32)
                scored = []
                for chunk, vec_blob in rows:
                    v = np.frombuffer(vec_blob, dtype=np.float32)
                    scored.append((chunk, cosine_sim(q_vec, v)))

                scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
                for i, (chunk, score) in enumerate(scored_sorted, start=1):
                    st.markdown(f"**{i}. score={score:.4f}**")
                    st.write(chunk)

# ==============================================================================
# Tab 5 — Export (Leeroy parity: deterministic session order, clean markdown, wrapped PDF)
# ==============================================================================
with tab_export:
    st.subheader("Export Conversation")

    history = st.session_state.messages

    if not history:
        st.info("No conversation history to export.")
    else:
        md_lines: List[str] = []
        for role, content in history:
            md_lines.append(f"## {role.upper()}\n\n{content}\n")
        md_text = "\n".join(md_lines)

        st.download_button(
            "Download Markdown",
            md_text,
            file_name="bro_chat.md",
            mime="text/markdown"
        )

        pdf_buf = io.BytesIO()
        pdf = canvas.Canvas(pdf_buf, pagesize=LETTER)
        width, height = LETTER
        y = height - 40

        for role, content in history:
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(40, y, f"{role.upper()}:")
            y -= 14

            pdf.setFont("Helvetica", 10)
            for line in textwrap.wrap(content, 90):
                if y < 40:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 10)
                    y = height - 40
                pdf.drawString(40, y, line)
                y -= 12

            y -= 10

        pdf.save()

        st.download_button(
            "Download PDF",
            pdf_buf.getvalue(),
            file_name="bro_chat.pdf",
            mime="application/pdf"
        )

# ==============================================================================
# Footer Telemetry (safe DOM injection; cannot blank tabs)
# ==============================================================================
st.markdown(
    f"""
    <style>
        .bro-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 6px 12px;
            font-size: 0.8rem;
            background-color: rgba(20,20,20,1);
            color: #ddd;
            display: flex;
            justify-content: space-between;
            z-index: 9999;
            border-top: 1px solid rgba(255,255,255,0.08);
        }}
    </style>

    <div class="bro-footer">
        <div>
            🧮 Tokens —
            Prompt: {st.session_state.token_usage["prompt"]} |
            Response: {st.session_state.token_usage["response"]} |
            Context Used: {st.session_state.token_usage["context_pct"]:.1f}%
        </div>

        <div>
            ⚙️ ctx={ctx} · temp={temperature} · top_p={top_p} ·
            top_k={top_k} · repeat={repeat_penalty}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
