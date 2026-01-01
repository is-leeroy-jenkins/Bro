"""
******************************************************************************************
Assembly:                Bro
Filename:                app.py
Author:                  Terry D. Eppler
Last Modified On:        2025-01-01
******************************************************************************************
"""
from __future__ import annotations

import base64
import io
import multiprocessing
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from llama_cpp import Llama
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer

# ==============================================================================
# Markdown / XML Converters  (NEW)
# ==============================================================================

_HEADING_RE = re.compile(r"^(#{2,6})\s+(.*)$")


def xml_converter(text: str) -> str:
    def normalize(value: str | None) -> str:
        return value.strip() if value else ""

    def render(elem: ET.Element, depth: int = 2) -> List[str]:
        lines: List[str] = []
        heading = "#" * min(depth, 6)
        title = elem.tag.replace("_", " ").strip()

        lines.append(f"{heading} {title}")
        lines.append("")

        txt = normalize(elem.text)
        if txt:
            lines.extend(txt.splitlines())
            lines.append("")

        for child in elem:
            lines.extend(render(child, depth + 1))
            tail = normalize(child.tail)
            if tail:
                lines.extend(tail.splitlines())
                lines.append("")
        return lines

    root = ET.fromstring(text)
    lines = render(root)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def markdown_converter(markdown: str) -> str:
    lines = markdown.splitlines()
    stack: List[Tuple[int, ET.Element]] = []
    root: ET.Element | None = None
    buffer: List[str] = []

    def flush(target: ET.Element) -> None:
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                target.text = (target.text + "\n" if target.text else "") + text
        buffer.clear()

    for line in lines:
        line = line.rstrip()
        match = _HEADING_RE.match(line)
        if match:
            hashes, title = match.groups()
            level = len(hashes)
            tag = title.lower().replace(" ", "_")
            elem = ET.Element(tag)

            if root is None:
                root = elem
                stack.append((level, elem))
                continue

            while stack and stack[-1][0] >= level:
                flush(stack[-1][1])
                stack.pop()

            if not stack:
                raise ValueError(f"Invalid heading structure near: {line}")

            stack[-1][1].append(elem)
            stack.append((level, elem))
        else:
            if stack:
                buffer.append(line)
            elif line.strip():
                raise ValueError("Text encountered before any heading.")

    if stack:
        flush(stack[-1][1])
    if root is None:
        raise ValueError("No headings found.")

    return ET.tostring(root, encoding="unicode")


# ==============================================================================
# Model / App Configuration
# ==============================================================================

DEFAULT_MODEL_PATH = "models/gemma-2-9b-it.Q4_K_M.gguf"
MODEL_PATH = os.getenv("BRO_LLM_PATH", DEFAULT_MODEL_PATH)
DB_PATH = "stores/sqlite/bro.db"
DEFAULT_CTX = 4096
CPU_CORES = multiprocessing.cpu_count()

st.set_page_config(page_title="Bro", layout="wide")

# ==============================================================================
# Utilities
# ==============================================================================

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


# ==============================================================================
# Database (UNCHANGED)
# ==============================================================================

def ensure_db() -> None:
    Path("stores/sqlite").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk TEXT,
            vector BLOB
        );
        CREATE TABLE IF NOT EXISTS Prompts (
            PromptsId INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            Text TEXT,
            Version TEXT,
            ID TEXT
        );
        """)

# (chat + prompt CRUD helpers unchanged — omitted here for brevity but assumed identical)

# ==============================================================================
# Loaders
# ==============================================================================

@st.cache_resource
def load_llm(ctx: int, threads: int) -> Llama:
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=ctx,
        n_threads=threads,
        n_batch=512,
        verbose=False
    )


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================================
# Session State Init (NEW additions marked)
# ==============================================================================

ensure_db()

st.session_state.setdefault("system_prompt", "")
st.session_state.setdefault("system_prompt_markdown", "")   # NEW
st.session_state.setdefault("system_prompt_format", "XML")  # NEW

# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:
    try:
        logo = image_to_base64("resources/images/bro_logo.png")
        st.markdown(
            f"<img src='data:image/png;base64,{logo}' style='max-height:80px;margin:auto;'>",
            unsafe_allow_html=True
        )
    except Exception:
        st.write("Bro")

    st.header("⚙️ Mind Controls")
    ctx = st.slider("Context Window", 2048, 8192, DEFAULT_CTX, 512)
    threads = st.slider("CPU Threads", 1, CPU_CORES, CPU_CORES)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 20, 5)
    repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.1, 0.05)

llm = load_llm(ctx, threads)
embedder = load_embedder()

# ==============================================================================
# Tabs
# ==============================================================================

tab_sys, tab_chat, tab_basic, tab_sem, tab_prompt, tab_export = st.tabs(
    ["System Instructions", "Text Generation",
     "Retrieval Augmentation", "Semantic Search",
     "Prompt Engineering", "Export"]
)

# ==============================================================================
# System Instructions Tab (UPDATED)
# ==============================================================================

with tab_sys:
    st.subheader("System Instructions")

    fmt1, fmt2, fmt3 = st.columns([2, 1, 1])
    with fmt1:
        st.radio("Edit Format", ["XML", "Markdown"],
                 key="system_prompt_format", horizontal=True)
    with fmt2:
        if st.button("XML → Markdown"):
            st.session_state.system_prompt_markdown = xml_converter(
                st.session_state.system_prompt
            )
            st.session_state.system_prompt_format = "Markdown"
    with fmt3:
        if st.button("Markdown → XML"):
            st.session_state.system_prompt = markdown_converter(
                st.session_state.system_prompt_markdown
            )
            st.session_state.system_prompt_format = "XML"

    if st.session_state.system_prompt_format == "XML":
        st.text_area(
            "System Prompt (XML)",
            value=st.session_state.system_prompt,
            key="system_prompt",
            height=280
        )
    else:
        st.text_area(
            "System Prompt (Markdown)",
            value=st.session_state.system_prompt_markdown,
            key="system_prompt_markdown",
            height=280
        )

# ==============================================================================
# Export Tab (UPDATED)
# ==============================================================================

with tab_export:
    st.subheader("Export")

    export_fmt = st.radio(
        "System Prompt Export Format",
        ["XML", "Markdown"],
        horizontal=True
    )

    if export_fmt == "XML":
        prompt_out = st.session_state.system_prompt
        fname = "bro_system_prompt.xml"
    else:
        prompt_out = xml_converter(st.session_state.system_prompt)
        fname = "bro_system_prompt.md"

    st.download_button(
        "Download System Prompt",
        prompt_out,
        file_name=fname
    )

    # Existing chat export (unchanged)

