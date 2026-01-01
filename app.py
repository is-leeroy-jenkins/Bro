# ******************************************************************************************
# Assembly:                Bro
# Filename:                app.py
# Author:                  Terry D. Eppler (integration)
# Purpose:                 Bro application — full Leeroy parity (CRUD + Prompt Engineering)
# ******************************************************************************************

from __future__ import annotations

import base64
import io
import os
import sqlite3
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from llama_cpp import Llama
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer

# ==========================================================================================
# Configuration & Guards
# ==========================================================================================
APP_NAME = "Bro"
DB_PATH = "stores/sqlite/bro.db"
HF_MODEL_URL = "https://huggingface.co/leeroy-jankins/bro"
DEFAULT_CTX = 4096

MODEL_PATH = os.getenv("BRO_LLM_PATH")
if not MODEL_PATH or not Path(MODEL_PATH).exists():
    st.error(
        "Model path not found. Set BRO_LLM_PATH to a valid GGUF model.\n\n"
        f"Reference: {HF_MODEL_URL}"
    )
    st.stop()

st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="resources/images/favicon.ico")
st.markdown("<style>.block-container{padding-bottom:90px}</style>", unsafe_allow_html=True)

# ==========================================================================================
# Database
# ==========================================================================================
def ensure_db() -> None:
    Path("stores/sqlite").mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as c:
        # Sessions table
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions(
                session_id TEXT PRIMARY KEY,
                created_at TEXT
            )
        """)

        # Chat history (base table)
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                created_at TEXT
            )
        """)

        # 🔴 MIGRATION: add session_id if missing
        cols = [r[1] for r in c.execute("PRAGMA table_info(chat_history)").fetchall()]
        if "session_id" not in cols:
            c.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT")
            default_sid = "legacy"
            c.execute(
                "UPDATE chat_history SET session_id = ? WHERE session_id IS NULL",
                (default_sid,)
            )
            c.execute(
                "INSERT OR IGNORE INTO sessions VALUES (?, ?)",
                (default_sid, datetime.utcnow().isoformat())
            )

        # Prompt store
        c.execute("""
            CREATE TABLE IF NOT EXISTS prompt_store(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                prompt_text TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Prompt templates
        c.execute("""
            CREATE TABLE IF NOT EXISTS prompt_templates(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                template_text TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Embeddings
        c.execute("""
            CREATE TABLE IF NOT EXISTS embeddings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_name TEXT,
                chunk TEXT,
                vector BLOB
            )
        """)


ensure_db()

# ==========================================================================================
# Utilities
# ==========================================================================================
def image_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

def chunk_text(text: str, size: int = 1200, overlap: int = 250) -> List[str]:
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += max(1, size - overlap)
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d else 0.0

# ==========================================================================================
# Loaders
# ==========================================================================================
@st.cache_resource
def load_llm(ctx: int, threads: int) -> Llama:
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=ctx,
        n_threads=threads,
        n_batch=256,
        verbose=False
    )

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================================================
# Session State Initialization (strict)
# ==========================================================================================
if "active_session_id" not in st.session_state:
    sid = str(uuid.uuid4())
    st.session_state.active_session_id = sid
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT OR IGNORE INTO sessions VALUES (?,?)", (sid, datetime.utcnow().isoformat()))

if "messages" not in st.session_state:
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            "SELECT role, content FROM chat_history WHERE session_id=? ORDER BY id",
            (st.session_state.active_session_id,)
        ).fetchall()
    st.session_state.messages = rows[:]

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are a helpful assistant optimized for instruction following, "
        "contextual comprehension, and structured reasoning."
    )

if "active_prompt_template" not in st.session_state:
    st.session_state.active_prompt_template = ""

if "basic_docs" not in st.session_state:
    st.session_state.basic_docs = []

if "semantic_enabled" not in st.session_state:
    st.session_state.semantic_enabled = False

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {"prompt": 0, "response": 0, "context_pct": 0.0}

# ==========================================================================================
# Sidebar (parameters only)
# ==========================================================================================
with st.sidebar:
    try:
        logo = image_to_base64("resources/images/bro_logo.png")
        st.markdown(
            f"<div style='text-align:center'><img src='data:image/png;base64,{logo}' width='60'></div>",
            unsafe_allow_html=True
        )
    except Exception:
        st.write(APP_NAME)

    st.header("⚙️ Mind Controls")
    ctx = st.slider("Context Window", 2048, 8192, DEFAULT_CTX, 512)
    threads = st.slider("CPU Threads", 1, os.cpu_count() or 4, 4)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 40, 10)
    repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.1, 0.05)

llm = load_llm(ctx, threads)
embedder = load_embedder()

# ==========================================================================================
# Prompt Builder (Leeroy parity)
# ==========================================================================================
def build_prompt(user_input: str) -> str:
    prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"

    if st.session_state.active_prompt_template:
        prompt += (
            "<|system|>\nPrompt Template:\n"
            f"{st.session_state.active_prompt_template}\n</s>\n"
        )

    if st.session_state.semantic_enabled:
        with sqlite3.connect(DB_PATH) as c:
            rows = c.execute("SELECT chunk, vector FROM embeddings").fetchall()
        if rows:
            q = embedder.encode([user_input])[0].astype(np.float32)
            scored = [
                (chunk, cosine_sim(q, np.frombuffer(vec, dtype=np.float32)))
                for chunk, vec in rows
            ]
            top = [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]]
            prompt += "<|system|>\nSemantic Context:\n"
            for c in top:
                prompt += f"- {c}\n"
            prompt += "</s>\n"

    if st.session_state.basic_docs:
        prompt += "<|system|>\nDocument Context:\n"
        for d in st.session_state.basic_docs[:6]:
            prompt += f"- {d}\n"
        prompt += "</s>\n"

    for r, c in st.session_state.messages:
        prompt += f"<|{r}|>\n{c}\n</s>\n"

    prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
    return prompt

# ==========================================================================================
# Tabs
# ==========================================================================================
tab_sys, tab_pe, tab_chat, tab_rag, tab_sem, tab_exp = st.tabs(
    [
        "System Instructions",
        "Prompt Engineering",
        "Text Generation",
        "Retrieval Augmentation",
        "Semantic Search",
        "Export",
    ]
)

# ------------------------------------------------------------------------------------------
# Tab 1 — System Instructions (CRUD)
# ------------------------------------------------------------------------------------------
with tab_sys:
    st.subheader("System Prompt")
    new_prompt = st.text_area("Edit system prompt:", st.session_state.system_prompt, height=220)
    if new_prompt.strip():
        st.session_state.system_prompt = new_prompt

    name = st.text_input("Save as name:")
    if st.button("Save Prompt") and name.strip():
        with sqlite3.connect(DB_PATH) as c:
            c.execute(
                "INSERT OR REPLACE INTO prompt_store(name,prompt_text,created_at,updated_at) "
                "VALUES (?,?,?,?)",
                (name, st.session_state.system_prompt, datetime.utcnow().isoformat(),
                 datetime.utcnow().isoformat())
            )

    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT name,prompt_text FROM prompt_store").fetchall()

    if rows:
        sel = st.selectbox("Load prompt:", [""] + [r[0] for r in rows])
        if sel:
            st.session_state.system_prompt = dict(rows)[sel]
        if st.button("Delete Selected") and sel:
            with sqlite3.connect(DB_PATH) as c:
                c.execute("DELETE FROM prompt_store WHERE name=?", (sel,))

# ------------------------------------------------------------------------------------------
# Tab 2 — Prompt Engineering (CRUD)
# ------------------------------------------------------------------------------------------
with tab_pe:
    st.subheader("Prompt Templates")
    tmpl = st.text_area("Template:", st.session_state.active_prompt_template, height=200)
    st.session_state.active_prompt_template = tmpl

    tname = st.text_input("Template name:")
    if st.button("Save Template") and tname.strip():
        with sqlite3.connect(DB_PATH) as c:
            c.execute(
                "INSERT OR REPLACE INTO prompt_templates(name,template_text,created_at,updated_at) "
                "VALUES (?,?,?,?)",
                (tname, tmpl, datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
            )

    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT name,template_text FROM prompt_templates").fetchall()

    if rows:
        sel = st.selectbox("Activate template:", [""] + [r[0] for r in rows])
        if sel:
            st.session_state.active_prompt_template = dict(rows)[sel]
        if st.button("Delete Template") and sel:
            with sqlite3.connect(DB_PATH) as c:
                c.execute("DELETE FROM prompt_templates WHERE name=?", (sel,))

# ------------------------------------------------------------------------------------------
# Tab 3 — Text Generation
# ------------------------------------------------------------------------------------------
with tab_chat:
    for r, c in st.session_state.messages:
        with st.chat_message(r):
            st.markdown(c)

    user_input = st.chat_input("Ask Bro…")
    if user_input:
        with sqlite3.connect(DB_PATH) as c:
            c.execute(
                "INSERT INTO chat_history(session_id,role,content,created_at) VALUES (?,?,?,?)",
                (st.session_state.active_session_id, "user", user_input, datetime.utcnow().isoformat())
            )
        st.session_state.messages.append(("user", user_input))

        prompt = build_prompt(user_input)

        with st.chat_message("assistant"):
            ph = st.empty()
            resp = ""
            for chunk in llm(
                prompt,
                stream=True,
                max_tokens=1024,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=["</s>"],
            ):
                resp += chunk["choices"][0]["text"]
                ph.markdown(resp)

        with sqlite3.connect(DB_PATH) as c:
            c.execute(
                "INSERT INTO chat_history(session_id,role,content,created_at) VALUES (?,?,?,?)",
                (st.session_state.active_session_id, "assistant", resp, datetime.utcnow().isoformat())
            )
        st.session_state.messages.append(("assistant", resp))

        pt = len(llm.tokenize(prompt.encode()))
        rt = len(llm.tokenize(resp.encode()))
        st.session_state.token_usage = {
            "prompt": pt,
            "response": rt,
            "context_pct": (pt + rt) / ctx * 100.0,
        }

# ------------------------------------------------------------------------------------------
# Tab 4 — Retrieval Augmentation (CRUD)
# ------------------------------------------------------------------------------------------
with tab_rag:
    st.subheader("Document Context")
    uploads = st.file_uploader("Upload documents", accept_multiple_files=True)
    if uploads:
        st.session_state.basic_docs.clear()
        for f in uploads:
            st.session_state.basic_docs.extend(chunk_text(f.read().decode(errors="ignore")))
        st.success(f"{len(st.session_state.basic_docs)} chunks loaded")
    if st.button("Clear Documents"):
        st.session_state.basic_docs.clear()

# ------------------------------------------------------------------------------------------
# Tab 5 — Semantic Search (CRUD)
# ------------------------------------------------------------------------------------------
with tab_sem:
    st.subheader("Semantic Index")
    st.session_state.semantic_enabled = st.checkbox(
        "Enable semantic context", st.session_state.semantic_enabled
    )

    uploads = st.file_uploader("Upload for semantic index", accept_multiple_files=True)
    if uploads:
        chunks = []
        for f in uploads:
            chunks.extend(chunk_text(f.read().decode(errors="ignore")))
        vecs = embedder.encode(chunks).astype(np.float32)
        with sqlite3.connect(DB_PATH) as c:
            c.execute("DELETE FROM embeddings")
            for ch, v in zip(chunks, vecs):
                c.execute(
                    "INSERT INTO embeddings(collection_name,chunk,vector) VALUES (?,?,?)",
                    ("default", ch, v.tobytes())
                )
        st.success(f"Indexed {len(chunks)} chunks")
    if st.button("Clear Semantic Index"):
        with sqlite3.connect(DB_PATH) as c:
            c.execute("DELETE FROM embeddings")

# ------------------------------------------------------------------------------------------
# Tab 6 — Export
# ------------------------------------------------------------------------------------------
with tab_exp:
    if not st.session_state.messages:
        st.info("Nothing to export")
    else:
        md = "\n\n".join(f"## {r.upper()}\n{c}" for r, c in st.session_state.messages)
        st.download_button("Download Markdown", md, "bro_chat.md")

        buf = io.BytesIO()
        pdf = canvas.Canvas(buf, pagesize=LETTER)
        y = LETTER[1] - 40
        for r, c in st.session_state.messages:
            pdf.drawString(40, y, f"{r.upper()}:")
            y -= 14
            for line in textwrap.wrap(c, 90):
                if y < 40:
                    pdf.showPage()
                    y = LETTER[1] - 40
                pdf.drawString(40, y, line)
                y -= 12
            y -= 10
        pdf.save()
        st.download_button("Download PDF", buf.getvalue(), "bro_chat.pdf")

# ==========================================================================================
# Footer Telemetry (safe)
# ==========================================================================================
st.markdown(
    f"""
    <style>
      .bro-footer {{
        position: fixed; bottom: 0; left: 0; width: 100%;
        padding: 6px 12px; background:#141414; color:#ddd;
        display:flex; justify-content:space-between;
        font-size:0.8rem; border-top:1px solid #333;
        z-index:9999;
      }}
    </style>
    <div class="bro-footer">
      <div>
        Tokens — Prompt: {st.session_state.token_usage['prompt']} |
        Response: {st.session_state.token_usage['response']} |
        Context: {st.session_state.token_usage['context_pct']:.1f}%
      </div>
      <div>
        ctx={ctx} · temp={temperature} · top_p={top_p} · top_k={top_k} · repeat={repeat_penalty}
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
