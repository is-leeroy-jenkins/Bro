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
# XML / Markdown Converters (ADDED)
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

        body = normalize(elem.text)
        if body:
            lines.extend(body.splitlines())
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
            txt = "\n".join(buffer).strip()
            if txt:
                target.text = (target.text + "\n" if target.text else "") + txt
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
                raise ValueError("Text before first heading.")

    if stack:
        flush(stack[-1][1])
    if root is None:
        raise ValueError("No headings found.")

    return ET.tostring(root, encoding="unicode")


# ==============================================================================
# Model Path Resolution
# ==============================================================================

DEFAULT_MODEL_PATH = "models/gemma-2-9b-it.Q4_K_M.gguf"
MODEL_PATH = os.getenv("BRO_LLM_PATH", DEFAULT_MODEL_PATH)
MODEL_PATH_OBJ = Path(MODEL_PATH)

if not MODEL_PATH_OBJ.exists():
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

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


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


# ==============================================================================
# Database + helpers (UNCHANGED)
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


def save_message(role: str, content: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_history (role, content) VALUES (?, ?)",
            (role, content)
        )


def load_history() -> List[Tuple[str, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT role, content FROM chat_history ORDER BY id"
        ).fetchall()


def clear_history() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chat_history")


# Prompt CRUD helpers unchanged (as provided)

# ==============================================================================
# Loaders
# ==============================================================================

@st.cache_resource
def load_llm(ctx: int, threads: int) -> Llama:
    return Llama(
        model_path=str(MODEL_PATH_OBJ),
        n_ctx=ctx,
        n_threads=threads,
        n_batch=512,
        verbose=False
    )


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================================
# Init
# ==============================================================================

ensure_db()
llm = load_llm(DEFAULT_CTX, CPU_CORES)
embedder = load_embedder()

st.session_state.setdefault("messages", load_history())
st.session_state.setdefault("system_prompt", "")
st.session_state.setdefault("system_prompt_markdown", "")
st.session_state.setdefault("system_prompt_format", "XML")
st.session_state.setdefault("basic_docs", [])
st.session_state.setdefault("use_semantic", False)
st.session_state.setdefault("selected_prompt_id", None)
st.session_state.setdefault("pending_system_prompt_name", None)

# ==============================================================================
# Tabs
# ==============================================================================

tab_system, tab_chat, tab_basic, tab_semantic, tab_prompt, tab_export = st.tabs(
    ["System Instructions", "Text Generation", "Retrieval Augmentation",
     "Semantic Search", "Prompt Engineering", "Export"]
)

# ==============================================================================
# System Instructions Tab (RESTORED + EXTENDED)
# ==============================================================================

with tab_system:
    st.subheader("System Instructions")

    df_prompts = fetch_prompts_df()
    names = [""] + df_prompts["Name"].tolist()

    selected = st.selectbox("Load System Prompt", names)
    st.session_state.pending_system_prompt_name = selected or None

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Load", disabled=not selected):
            rec = fetch_prompt_by_name(selected)
            if rec:
                st.session_state.system_prompt = rec["Text"]
                st.session_state.system_prompt_markdown = ""
                st.session_state.system_prompt_format = "XML"
                st.session_state.selected_prompt_id = rec["PromptsId"]

    with c2:
        if st.button("Clear"):
            st.session_state.system_prompt = ""
            st.session_state.system_prompt_markdown = ""

    with c3:
        if st.button("Edit", disabled=not selected):
            rec = fetch_prompt_by_name(selected)
            if rec:
                st.session_state.selected_prompt_id = rec["PromptsId"]

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
            height=260
        )
    else:
        st.text_area(
            "System Prompt (Markdown)",
            value=st.session_state.system_prompt_markdown,
            key="system_prompt_markdown",
            height=260
        )

# ==============================================================================
# Export Tab (EXTENDED)
# ==============================================================================

with tab_export:
    st.subheader("Export")

    fmt = st.radio("System Prompt Format", ["XML", "Markdown"], horizontal=True)
    if fmt == "XML":
        out = st.session_state.system_prompt
        fname = "bro_system_prompt.xml"
    else:
        out = xml_converter(st.session_state.system_prompt)
        fname = "bro_system_prompt.md"

    st.download_button("Download System Prompt", out, file_name=fname)

    hist = load_history()
    md = "\n\n".join([f"**{r.upper()}**\n{c}" for r, c in hist])
    st.download_button("Download Chat (Markdown)", md, "bro_chat.md")

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=LETTER)
    y = 750
    for r, c in hist:
        pdf.drawString(40, y, f"{r.upper()}: {c[:90]}")
        y -= 14
        if y < 50:
            pdf.showPage()
            y = 750
    pdf.save()
    st.download_button("Download Chat (PDF)", buf.getvalue(), "bro_chat.pdf")
