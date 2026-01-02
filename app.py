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
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from llama_cpp import Llama
from reportlab.lib.pagesizes import LETTER
import re
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET

# ==============================================================================
# Model Path Resolution (CHANGED: Gemma)
# ==============================================================================
DEFAULT_MODEL_PATH = "models/gemma-2-9b-it.Q4_K_M.gguf"
MODEL_PATH = os.getenv("BRO_LLM_PATH", DEFAULT_MODEL_PATH)
MODEL_PATH_OBJ = Path(MODEL_PATH)

if not MODEL_PATH_OBJ.exists():
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

# ==============================================================================
# Constants (CHANGED: DB)
# ==============================================================================
DB_PATH = "stores/sqlite/bro.db"
DEFAULT_CTX = 4096
CPU_CORES = multiprocessing.cpu_count()

# ==============================================================================
# Streamlit Config (CHANGED: title)
# ==============================================================================
st.set_page_config(
    page_title="Bro",
    layout="wide",
    page_icon="resources/images/favicon.ico"
)

# ==============================================================================
# Utilities
# ==============================================================================
_HEADING_RE = re.compile(r"^(#{2,6})\s+(.*)$")

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def xml_converter(xml_text: str) -> str:
    """
    
	    Purpose:
	        Convert XML-delimited system instructions into a Markdown document
	        suitable for editing, review, or export.
	
	    Parameters:
	        xml_text (str):
	            XML-formatted system instruction text.
	
	    Returns:
	        str:
	            Markdown-formatted representation of the XML structure.
	            
    """

    def normalize(value: str | None) -> str:
        return value.strip() if value else ""

    def render(elem: ET.Element, depth: int = 2) -> List[str]:
        lines: List[str] = []

        heading = "#" * min(depth, 6)
        title = elem.tag.replace("_", " ").strip()

        # Section heading
        lines.append(f"{heading} {title}")
        lines.append("")

        # Element body text
        body = normalize(elem.text)
        if body:
            for line in body.splitlines():
                lines.append(line.rstrip())
            lines.append("")

        # Child elements
        for child in elem:
            lines.extend(render(child, depth + 1))

            tail = normalize(child.tail)
            if tail:
                for line in tail.splitlines():
                    lines.append(line.rstrip())
                lines.append("")

        return lines

    root = ET.fromstring(xml_text)

    output: List[str] = []
    output.extend(render(root))

    # Trim trailing whitespace
    while output and not output[-1].strip():
        output.pop()

    return "\n".join(output)

def markdown_converter(markdown: str) -> str:
    """
	    
	    Purpose:
	        Convert Markdown-formatted system instructions back into
	        XML-delimited instructions.
	
	    Parameters:
	        markdown (str):
	            Markdown text containing hierarchical headings and body text.
	
	    Returns:
	        str:
	            XML-formatted system instruction text.
	            
    """

    lines = markdown.splitlines()

    # Stack of (heading_level, xml_element)
    stack: List[Tuple[int, ET.Element]] = []
    buffer: List[str] = []

    root: ET.Element | None = None

    def flush(target: ET.Element) -> None:
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                if target.text:
                    target.text += "\n" + text
                else:
                    target.text = text
        buffer.clear()

    for line in lines:
        line = line.rstrip()

        match = _HEADING_RE.match(line)
        if match:
            hashes, title = match.groups()
            level = len(hashes)
            tag = title.lower().replace(" ", "_")

            element = ET.Element(tag)

            # Root element
            if root is None:
                root = element
                stack.append((level, element))
                continue

            # Close elements until a valid parent is found
            while stack and stack[-1][0] >= level:
                flush(stack[-1][1])
                stack.pop()

            if not stack:
                raise ValueError(f"Invalid heading structure near: {line}")

            parent = stack[-1][1]
            parent.append(element)
            stack.append((level, element))
        else:
            if stack:
                buffer.append(line)
            elif line.strip():
                raise ValueError(
                    "Text encountered before any heading; invalid Markdown structure."
                )

    # Flush remaining text
    if stack:
        flush(stack[-1][1])

    if root is None:
        raise ValueError("No headings found; cannot construct XML.")

    return ET.tostring(root, encoding="unicode")

# ==============================================================================
# Database (UNCHANGED SCHEMA)
# ==============================================================================
def ensure_db() -> None:
    Path("stores/sqlite").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk TEXT,
                vector BLOB
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Prompts (
                PromptsId INTEGER NOT NULL UNIQUE,
                Name TEXT(80),
                Text TEXT,
                Version TEXT(80),
                ID TEXT(80),
                PRIMARY KEY(PromptsId AUTOINCREMENT)
            )
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

# ==============================================================================
# Prompt DB helpers (UNCHANGED)
# ==============================================================================
def fetch_prompts_df() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT PromptsId, Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
            conn
        )
    df.insert(0, "Selected", False)
    return df

def fetch_prompt_by_id(pid: int) -> Dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
            (pid,)
        )
        row = cur.fetchone()
        return dict(zip([c[0] for c in cur.description], row)) if row else None

def fetch_prompt_by_name(name: str) -> Dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE Name=?",
            (name,)
        )
        row = cur.fetchone()
        return dict(zip([c[0] for c in cur.description], row)) if row else None

def insert_prompt(data: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO Prompts (Name, Text, Version, ID) VALUES (?, ?, ?, ?)",
            (data["Name"], data["Text"], data["Version"], data["ID"])
        )

def update_prompt(pid: int, data: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE Prompts SET Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
            (data["Name"], data["Text"], data["Version"], data["ID"], pid)
        )

def delete_prompt(pid: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM Prompts WHERE PromptsId=?", (pid,))

# ==============================================================================
# Loaders (UNCHANGED)
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
# Sidebar (CHANGED: logo optional)
# ==============================================================================
with st.sidebar:
    try:
        logo_b64 = image_to_base64("resources/images/bro_logo.png")
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 0.85rem;
            ">
                <img src="data:image/png;base64,{logo_b64}"
                     style="max-height: 50px;" />
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.write("Bro")

    st.header("⚙️ Mind Controls")
    max_tokens = st.slider( "Max Tokens", min_value=128, max_value=14096, value=10024, step=128,
	    help="Maximum number of tokens generated per response")

    ctx = st.slider("Context Window", 2048, 8192, DEFAULT_CTX, 512,
	    help="Maximum number of tokens the model can consider at once, including system instructions, history, and context")
    threads = st.slider("CPU Threads", 1, CPU_CORES, CPU_CORES,
	    help="Number of CPU threads used for inference; higher values improve speed but increase CPU usage" )
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1,
	    help="Controls randomness in generation; lower values are more deterministic, higher values increase creativity")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05,
	    help="Nucleus sampling threshold; limits token selection to the smallest set whose cumulative probability exceeds this value")
    top_k = st.slider("Top-k", 1, 20, 5, help="Limits token selection to the top K most probable tokens at each step")
    repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.1, 0.05,
	    help="Penalizes repeated tokens to reduce looping and redundant responses")
    typical_p = st.slider("Typical P", 0.1, 1.0, 1.0, 0.05,
	    help="Typical sampling threshold; reduces unlikely token choices while preserving coherent variation")
    presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, 0.0, 0.1, help="Encourages introducing new topics by penalizing tokens already present in the context")
    frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1, help="Reduces repetition by penalizing tokens based on how frequently they have appeared")
# ==============================================================================
# Init
# ==============================================================================
ensure_db()
llm = load_llm(ctx, threads)
embedder = load_embedder()

st.session_state.setdefault("messages", load_history())
st.session_state.setdefault("system_prompt", "")
st.session_state.setdefault("basic_docs", [])
st.session_state.setdefault("use_semantic", False)
st.session_state.setdefault("selected_prompt_id", None)
st.session_state.setdefault("pending_system_prompt_name", None)

# ==============================================================================
# Tabs (UNCHANGED order & behavior)
# ==============================================================================
tab_system, tab_chat, tab_basic, tab_semantic, tab_prompt, tab_export = st.tabs(
    ["System Instructions", "Text Generation", "Retrieval Augmentation",
     "Semantic Search", "Prompt Engineering", "Export"]
)

# ==============================================================================
# Prompt Builder (UNCHANGED)
# ==============================================================================
def build_prompt(user_input: str) -> str:
    prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"

    if st.session_state.use_semantic:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT chunk, vector FROM embeddings").fetchall()
        if rows:
            q = embedder.encode([user_input])[0]
            scored = [(c, cosine_sim(q, np.frombuffer(v))) for c, v in rows]
            for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]:
                prompt += f"<|system|>\n{c}\n</s>\n"

    for d in st.session_state.basic_docs[:6]:
        prompt += f"<|system|>\n{d}\n</s>\n"

    for r, c in st.session_state.messages:
        prompt += f"<|{r}|>\n{c}\n</s>\n"

    prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
    return prompt

# ==============================================================================
# System Instructions Tab (WITH XML ⇄ MARKDOWN CONVERSION)
# ==============================================================================
with tab_system:
    st.subheader("System Instructions")

    # ---- Load prompt names ----
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT Name FROM Prompts ORDER BY PromptsId DESC"
            ).fetchall()
        prompt_names = [""] + [r[0] for r in rows]
    except Exception as ex:
        st.error(f"Failed to load system prompts: {ex}")
        prompt_names = [""]

    selected_name = st.selectbox(
        "Load System Prompt",
        prompt_names,
        key="system_prompt_selector"
    )

    st.session_state.pending_system_prompt_name = (
        selected_name if selected_name else None
    )

    # ---- Controls ----
    col_load, col_clear, col_edit = st.columns(3)

    with col_load:
        load_clicked = st.button(
            "Load",
            disabled=st.session_state.pending_system_prompt_name is None
        )

    with col_clear:
        clear_clicked = st.button("Clear")

    with col_edit:
        edit_clicked = st.button(
            "Edit",
            disabled=st.session_state.pending_system_prompt_name is None
        )

    # ---- Actions ----
    if load_clicked:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "SELECT PromptsId, Text FROM Prompts WHERE Name=?",
                (st.session_state.pending_system_prompt_name,)
            )
            row = cur.fetchone()
        if row:
            st.session_state.selected_prompt_id = row[0]
            st.session_state.system_prompt = row[1] or ""

    if clear_clicked:
        st.session_state.system_prompt = ""
        st.session_state.selected_prompt_id = None

    if edit_clicked:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "SELECT PromptsId FROM Prompts WHERE Name=?",
                (st.session_state.pending_system_prompt_name,)
            )
            row = cur.fetchone()
        if row:
            st.session_state.selected_prompt_id = row[0]

    # --------------------------------------------------------------------------
    # XML ⇄ Markdown Conversion Controls (DROP-IN)
    # --------------------------------------------------------------------------
    st.markdown("### Format Tools")

    fmt_col, btn_col = st.columns([3, 1])

    with fmt_col:
        view_format = st.radio(
            "View / Edit Format",
            ["XML (Canonical)", "Markdown (Readable)"],
            horizontal=True,
            key="system_prompt_format"
        )

    with btn_col:
        convert_clicked = st.button("Convert")

    if convert_clicked:
        try:
            current_text = st.session_state.system_prompt or ""

            if view_format.startswith("Markdown"):
                # XML → Markdown
                st.session_state.system_prompt = xml_converter(current_text)
            else:
                # Markdown → XML
                st.session_state.system_prompt = markdown_converter(current_text)

        except Exception as ex:
            st.error(f"Conversion failed: {ex}")

    # ---- System Prompt Text Area (single ownership) ----
    st.text_area(
        "System Prompt",
        key="system_prompt",
        height=300
    )

# ==============================================================================
# Text Generation Tab (UNCHANGED)
# ==============================================================================
with tab_chat:
    if st.button("🧹 Clear Chat"):
        clear_history()
        st.session_state.messages = []
        st.rerun()

    for r, c in st.session_state.messages:
        with st.chat_message(r):
            st.markdown(c)

    user_input = st.chat_input("Ask Bro…")
    if user_input:
        save_message("user", user_input)
        st.session_state.messages.append(("user", user_input))

        prompt = build_prompt(user_input)
        with st.chat_message("assistant"):
            out, buf = st.empty(), ""
            for chunk in llm(prompt, stream=True, max_tokens=max_tokens,
                              temperature=temperature, top_p=top_p,
                              repeat_penalty=repeat_penalty, stop=["</s>"]):
                buf += chunk["choices"][0]["text"]
                out.markdown(buf + "▌")
            out.markdown(buf)

        save_message("assistant", buf)
        st.session_state.messages.append(("assistant", buf))

# ==============================================================================
# Retrieval Augmentation Tab (UNCHANGED)
# ==============================================================================
with tab_basic:
    files = st.file_uploader("Upload documents", accept_multiple_files=True)
    if files:
        st.session_state.basic_docs.clear()
        for f in files:
            st.session_state.basic_docs.extend(chunk_text(f.read().decode(errors="ignore")))
        st.success(f"{len(st.session_state.basic_docs)} chunks loaded")

# ==============================================================================
# Semantic Search Tab (UNCHANGED)
# ==============================================================================
with tab_semantic:
    st.session_state.use_semantic = st.checkbox(
        "Use Semantic Context", st.session_state.use_semantic
    )
    files = st.file_uploader("Upload for embedding", accept_multiple_files=True)
    if files:
        chunks = []
        for f in files:
            chunks.extend(chunk_text(f.read().decode(errors="ignore")))
        vecs = embedder.encode(chunks)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM embeddings")
            for c, v in zip(chunks, vecs):
                conn.execute(
                    "INSERT INTO embeddings (chunk, vector) VALUES (?, ?)",
                    (c, v.tobytes())
                )
        st.success("Semantic index built")

# ==============================================================================
# Prompt Engineering Tab (SELF-CONTAINED)
# ==============================================================================
with tab_prompt:
    st.subheader("Prompt Engineering")

    # ---- Load prompts directly from DB (no helper dependency) ----
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT PromptsId, Name, Text, Version, ID FROM Prompts "
                "ORDER BY PromptsId DESC",
                conn
            )
    except Exception as ex:
        st.error(f"Failed to load prompts: {ex}")
        df = pd.DataFrame(
            columns=["PromptsId", "Name", "Text", "Version", "ID"]
        )

    # Ensure selection column exists
    if "Selected" not in df.columns:
        df.insert(0, "Selected", False)

    # Reflect currently selected prompt (if any)
    if st.session_state.selected_prompt_id is not None:
        df["Selected"] = (
            df["PromptsId"] == st.session_state.selected_prompt_id
        )

    # ---- Editable grid (selection only) ----
    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        disabled=["PromptsId", "Name", "Text", "Version", "ID"]
    )

    selected_ids = edited.loc[edited["Selected"], "PromptsId"].tolist()
    st.session_state.selected_prompt_id = (
        selected_ids[0] if len(selected_ids) == 1 else None
    )

    # ---- Load selected prompt into editor ----
    if st.session_state.selected_prompt_id is not None:
        row = df.loc[
            df["PromptsId"] == st.session_state.selected_prompt_id
        ].iloc[0]
        prompt = {
            "Name": row["Name"] or "",
            "Text": row["Text"] or "",
            "Version": row["Version"] or "",
            "ID": row["ID"] or "",
        }
    else:
        prompt = {"Name": "", "Text": "", "Version": "", "ID": ""}

    # ---- Controls ----
    c1, c2 = st.columns(2)

    with c1:
        if st.button("+ New"):
            st.session_state.selected_prompt_id = None
            prompt = {"Name": "", "Text": "", "Version": "", "ID": ""}

    with c2:
        if st.button(
            "🗑 Delete",
            disabled=st.session_state.selected_prompt_id is None
        ):
            delete_prompt(st.session_state.selected_prompt_id)
            st.session_state.selected_prompt_id = None
            st.rerun()

    # ---- Prompt fields ----
    name = st.text_input("Name", prompt["Name"])
    version = st.text_input("Version", prompt["Version"])
    pid = st.text_input("ID", prompt["ID"])
    text = st.text_area(
        "Prompt Text",
        prompt["Text"],
        height=280
    )

    # ---- Save ----
    if st.button("💾 Save"):
        data = {
            "Name": name,
            "Text": text,
            "Version": version,
            "ID": pid,
        }

        if st.session_state.selected_prompt_id is not None:
            update_prompt(st.session_state.selected_prompt_id, data)
        else:
            insert_prompt(data)

        st.rerun()


# ==============================================================================
# Export Tab (CHANGED filenames only)
# ==============================================================================
with tab_export:
    hist = load_history()
    md = "\n\n".join([f"**{r.upper()}**\n{c}" for r, c in hist])
    st.download_button("Download Markdown", md, "bro_chat.md")

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
    st.download_button("Download PDF", buf.getvalue(), "bro_chat.pdf")

def render_footer() -> None:
    footer_text = (
        f"⚙️ ctx={ctx} · "
        f"threads={threads} · "
        f"temp={temperature:.2f} · "
        f"top_p={top_p:.2f} · "
        f"top_k={top_k} · "
        f"repeat={repeat_penalty:.2f}"
        f" · max_tokens={max_tokens}"
    )

    st.markdown(
        f"""
        <style>
        .bro-footer {{
            text-align: right;
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: rgba(20, 20, 20, 0.95);
            color: #d0d0d0;
            font-size: 0.85rem;
            padding: 0.35rem 0.75rem;
            border-top: 1px solid #333;
            z-index: 100;
        }}
        </style>

        <div class="bro-footer">
            {footer_text}
        </div>
        """,
        unsafe_allow_html=True
    )

render_footer( )