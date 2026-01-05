# MECH503_Chatbot.py  – Streamlit app for Linear Elasticity course
"""
Chatbot for MECH 503 (Linear Elasticity)
----------------------------------------
• Loads **all** LaTeX notes (`*.tex`) **and** PDFs (`*.pdf`) found in a local `notes/` folder.
• Builds an embedding index with cosine similarity + Maximal Marginal Relevance (MMR) re‑ranking.
• Answers questions with **GPT‑3.5‑turbo** **using *only* the course notes as context** (no external web search).
• Each answer ends with a **Sources** section listing the chapter/page names that fed the response.

Usage
-----
1. Put your course notes in a folder called `notes/` (same level as this script).
2. Set your OpenAI key in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk‑..."
```
3. `pip install streamlit openai pypdf numpy`
4. `streamlit run MECH503_Chatbot.py`
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import openai
import streamlit as st
from pypdf import PdfReader
import json

import os
from PIL import Image

from glob import glob

# ---------------------------------------------------------
# Password gate
# ---------------------------------------------------------
import hmac

def require_password():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return

    st.title("MECH503 Course Chatbot")
    pw = st.text_input("Enter course password", type="password")

    if st.button("Enter"):
        expected = st.secrets.get("COURSE_PASSWORD", "")
        if expected and hmac.compare_digest(pw, expected):
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()

require_password()
# ---------------------------------------------------------
# Quota manager
# ---------------------------------------------------------
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

APP_TZ = ZoneInfo("America/Vancouver")

APP_DIR = Path(__file__).resolve().parent
QUOTA_DIR = APP_DIR / ".quota"
QUOTA_FILE = QUOTA_DIR / "usage.json"


def _today_key() -> str:
    return datetime.now(APP_TZ).strftime("%Y-%m-%d")


def _load_usage() -> dict:
    QUOTA_DIR.mkdir(parents=True, exist_ok=True)
    if not QUOTA_FILE.exists():
        return {}
    try:
        return json.loads(QUOTA_FILE.read_text(encoding="utf-8"))
    except Exception:
        # If file ever gets corrupted, reset it
        return {}


def _save_usage(data: dict) -> None:
    QUOTA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = QUOTA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(QUOTA_FILE)  # atomic-ish replace


def _get_limits() -> tuple[int, int, int]:
    max_q_session = int(st.secrets.get("MAX_Q_PER_SESSION", 15))
    max_q_day = int(st.secrets.get("MAX_Q_PER_DAY", 250))
    max_tokens_day = int(st.secrets.get("MAX_TOKENS_PER_DAY", 300000))
    return max_q_session, max_q_day, max_tokens_day


def quota_check_before_call() -> None:
    """Call this right before you call OpenAI."""
    max_q_session, max_q_day, max_tokens_day = _get_limits()

    # Session caps (per device/browser session)
    st.session_state.setdefault("q_count_session", 0)
    if st.session_state["q_count_session"] >= max_q_session:
        st.warning("Session limit reached for today. Please try again later.")
        st.stop()

    # Global daily caps (shared across everyone using the app)
    usage = _load_usage()
    day = _today_key()
    day_rec = usage.get(day, {"questions": 0, "tokens": 0})

    if day_rec["questions"] >= max_q_day:
        st.warning("Daily class quota reached. Please try again tomorrow.")
        st.stop()

    if day_rec["tokens"] >= max_tokens_day:
        st.warning("Daily token quota reached. Please try again tomorrow.")
        st.stop()


def quota_consume_after_call(tokens_used: int) -> None:
    """Call this right after you get an OpenAI response."""
    # Increment session question count
    st.session_state["q_count_session"] = st.session_state.get("q_count_session", 0) + 1

    # Increment global daily usage
    usage = _load_usage()
    day = _today_key()
    day_rec = usage.get(day, {"questions": 0, "tokens": 0})
    day_rec["questions"] += 1
    day_rec["tokens"] += int(tokens_used)
    usage[day] = day_rec
    _save_usage(usage)


def extract_total_tokens(response) -> int:
    """Defensive extraction across SDK variations."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    # openai SDK often provides usage.total_tokens as an int-like field
    total = getattr(usage, "total_tokens", None)
    if total is not None:
        return int(total)
    # fallback if usage is dict-like
    try:
        return int(usage.get("total_tokens", 0))
    except Exception:
        return 0

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 350
CHUNK_OVERLAP = 50
TOP_K = 4

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------
# Helper functions – text processing & embeddings
# ---------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
def mmr(query_vec: np.ndarray, doc_embeds: np.ndarray, k: int = 4, lambda_param: float = 0.6) -> List[int]:
    """Maximal Marginal Relevance – pick *k* diverse yet relevant docs."""
    doc_norms = np.linalg.norm(doc_embeds, axis=1) + 1e-10
    q_norm = np.linalg.norm(query_vec) + 1e-10
    sims = (doc_embeds @ query_vec) / (doc_norms * q_norm)

    selected: List[int] = []
    candidates = list(range(len(doc_embeds)))

    while candidates and len(selected) < k:
        if not selected:
            best = int(np.argmax(sims))
            selected.append(best)
            candidates.remove(best)
            continue
        mmr_scores = []
        for idx in candidates:
            diversity = max(
                (doc_embeds[idx] @ doc_embeds[j]) / (doc_norms[idx] * doc_norms[j])
                for j in selected
            )
            mmr_scores.append(lambda_param * sims[idx] - (1 - lambda_param) * diversity)
        best = candidates[int(np.argmax(mmr_scores))]
        selected.append(best)
        candidates.remove(best)
    return selected

def load_json_chunks(json_path: Path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = [entry["content"] for entry in data]
    meta = [
        f"{entry.get('section', '')} {entry.get('subsection', '')}".strip()
        for entry in data
    ]
    embeds = []
    for i in range(0, len(chunks), 96):
        batch = chunks[i:i + 96]
        res = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeds.extend([np.array(d.embedding, dtype=np.float32) for d in res.data])
    return chunks, np.vstack(embeds), meta

def chunk_text(text: str, max_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

# ---------------------------------------------------------
# Load notes (TEX + PDF)
# ---------------------------------------------------------

def load_tex_notes(path: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for f in path.glob("*.tex"):
        txt = f.read_text(encoding="utf-8", errors="ignore")
        txt = re.sub(r"%.*", "", txt)  # strip comments
        txt = re.sub(r"\\label\{.*?\}", "", txt)  # remove LaTeX labels
        txt = re.sub(r"\\cite\{.*?\}", "", txt)
        txt = re.sub(r"\\ref\{.*?\}", "", txt)
        txt = re.sub(r"\s+", " ", txt)
        entries.append((f.stem, txt.strip()))
    return entries


def _clean_pdf_text(t: str) -> str:
    t = re.sub(r"\\\\\d+", "", t)  # drop control codes like \1
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def load_pdfs(path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for pdf in path.glob("*.pdf"):
        try:
            reader = PdfReader(str(pdf))
        except Exception:
            continue
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            clean = _clean_pdf_text(raw)
            if clean:
                out.append((f"{pdf.stem}-p{i}", clean))
    return out


def build_corpus():
    notes_dir = Path("notes")
    chunks, meta = [], []
    for name, txt in load_tex_notes(notes_dir) + load_pdfs(notes_dir):
        for ch in chunk_text(txt):
            if ch.strip():
                chunks.append(ch)
                meta.append(name)

    embeds = []
    for i in range(0, len(chunks), 96):
        batch = chunks[i : i + 96]
        res = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeds.extend([np.array(d.embedding, dtype=np.float32) for d in res.data])
    return chunks, np.vstack(embeds) if embeds else np.empty((0, 1)), meta


@st.cache_resource(show_spinner="🔍 Loading preprocessed notes…")
def get_index():
    json_path = Path("notes/merged_notes.chunks.json")
    return load_json_chunks(json_path)


# ---------------------------------------------------------
# Math‑block formatter for Streamlit‑Markdown + MathJax
# ---------------------------------------------------------

def _fmt_math(txt: str) -> str:
    if not txt:
        return ""
    
    # Convert LaTeX math
    txt = re.sub(r"\\begin{equation\*?}(.*?)\\end{equation\*?}", r"\n\n$$\1$$\n\n", txt, flags=re.DOTALL)
    txt = re.sub(r"\\begin{align\*?}(.*?)\\end{align\*?}", r"\n\n$$\1$$\n\n", txt, flags=re.DOTALL)
    txt = re.sub(r"\\\[(.*?)\\\]", r"\n\n$$\1$$\n\n", txt, flags=re.DOTALL)
    txt = re.sub(r"\\\((.*?)\\\)", r"$\1$", txt, flags=re.DOTALL)

    # Convert \includegraphics{...} to Markdown <img> tags
    txt = re.sub(
        r"\\includegraphics(?:\[.*?\])?\{(.*?)\}",
        lambda m: f'<img src="{m.group(1).strip()}" style="max-width:100%; height:auto;">',
        txt
    )

    return txt

def render_chunk_with_images(content: str):
    # Extract image paths from markdown
    image_paths = re.findall(r"!\[\]\((media/[^)]+)\)", content)

    # Remove markdown image syntax from content
    cleaned_content = re.sub(r"!\[\]\(media/[^)]+\)", "", content)

    # Show text using markdown and math formatting
    st.chat_message("assistant").markdown(_fmt_math(cleaned_content), unsafe_allow_html=True)

    # Show images using st.image()
    for img_path in image_paths:
        st.chat_message("assistant").image(f"static/{img_path}")

def render_with_images(markdown_text: str):
    pattern = r'\\includegraphics(?:\[.*?\])?\{(.*?)\}'
    parts = re.split(pattern, markdown_text)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            st.markdown(_fmt_math(part), unsafe_allow_html=True)
        else:
            image_path = os.path.join("notes", part.lstrip("./"))
            if os.path.exists(image_path):
                st.image(Image.open(image_path), use_column_width=True)
            else:
                st.warning(f"Image not found: {image_path}")

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(page_title="MECH 503 Chatbot", page_icon="💬")
st.title("💬 MECH 503 Linear Elasticity Chatbot")
st.image("static/media/image004.gif", caption="Header Image Test")

chunks, embeddings, meta = get_index()

# Sidebar – chapter shortcuts
chapters = sorted(set(m for m in meta if not re.match(r".+-p\d+", m)))
st.sidebar.header("📑 Course chapters")
for ch in chapters:
    if st.sidebar.button(ch, key=f"chap-{ch}"):
        snippet = next((c for c, m in zip(chunks, meta) if m == ch), "")
        st.session_state.setdefault("chat", []).append({"role": "assistant", "content": f"**{ch}**\n\n{snippet[:800]}…"})
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Answers are generated *solely* from the course notes and PDFs—no external web sources.")

# Chat history container
if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant", "content": "Hi! Ask me anything about **linear elasticity**."}]

for m in st.session_state.chat:
    if m["role"] == "assistant":
        render_chunk_with_images(m["content"])
    else:
        st.chat_message("user").markdown(m["content"])

# ---------------------------------------------------------
# User prompt handler
# ---------------------------------------------------------
if prompt := st.chat_input("Type your question…"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Embed the query
    q_vec = np.array(
        client.embeddings.create(model=EMBEDDING_MODEL, input=[prompt]).data[0].embedding,
        dtype=np.float32,
    )

    # Similarities and context selection
    top_ids = mmr(q_vec, embeddings, k=TOP_K)
    context_chunks = [chunks[i] for i in top_ids]
    context_meta = [meta[i] for i in top_ids]
    context = "\n\n".join(context_chunks)

    # ---------------------------------------------------------
    # Compose LLM call
    # ---------------------------------------------------------
    sys_prompt = (
        "You are an engineering TA for linear elasticity. "
        "Answer clearly using markdown; all math must be LaTeX ($ for inline, $$ on blank lines for display). "
        "Base your answer *only* on the CONTEXT provided. "
        "End with a 'Sources' section listing where the info came from."
    )

    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {prompt}\nAnswer in markdown."

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_accum = ""
        with st.spinner("Thinking…"):
            # ✅ Quota check goes here (right before OpenAI call)
            quota_check_before_call()

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                stream=True,
            )
            for chunk in resp:
                token = chunk.choices[0].delta.content or ""
                answer_accum += token
                placeholder.markdown(_fmt_math(answer_accum), unsafe_allow_html=True)
            
            # ✅ Consume quota AFTER the call completes
            # For streaming, exact tokens aren't available here in your current setup.
            quota_consume_after_call(tokens_used=0)

        # Also save to chat history
        st.session_state.chat.append({"role": "assistant", "content": answer_accum})

        # Append sources list ourselves (ensures accuracy)
        unique_sources = list(dict.fromkeys(context_meta))
        src_md = "\n".join(f"- {s}" for s in unique_sources)
        answer_full = f"{answer_accum}\n\n---\n**Sources**\n{src_md}"

    st.session_state.chat.append({"role": "assistant", "content": answer_full})

    # Optional: show metadata (e.g., chunk type, tags)
    with st.expander("🔍 Retrieved Context Chunks (Debug View)", expanded=False):
        for i in top_ids:
            chunk = chunks[i]
            meta_label = meta[i]
            preview = chunk[:200].replace("\n", " ").strip() + "..."
            st.markdown(f"- **{meta_label}** — `{preview}`")
