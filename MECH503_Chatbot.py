# MECH503_Chatbot_webfallback_v2.py – Streamlit app for Linear Elasticity course (with trusted web fallback)
"""
MECH 503 Chatbot (Linear Elasticity) – Trusted Web Fallback Edition (v2)
-----------------------------------------------------------------------
Preserves features from your MECH503 chatbot:
- Password gate (COURSE_PASSWORD in Streamlit secrets)
- Quotas + usage meter (MAX_Q_PER_SESSION / MAX_Q_PER_DAY / MAX_TOKENS_PER_DAY)
- Loads pre-chunked JSON notes (notes/merged_notes.chunks.json)
- RAG retrieval with embeddings + MMR
- Image rendering for local static/media assets
- Debug view showing retrieved chunks + similarities

Adds / improves:
- Trusted web fallback (Tavily Search API) when note retrieval confidence is low
- Domain allowlist for web search and fetching
- Clear “Sources (Course notes)” and “Sources (Trusted web)” sections
- Sidebar “Retrieval diagnostics” (top_sim, used_web, web sources)

Secrets (local or Streamlit Cloud):
OPENAI_API_KEY = "sk-..."
COURSE_PASSWORD = "your-course-password"

# Optional (web fallback)
TAVILY_API_KEY = "tvly-..."
WEB_FALLBACK_ENABLED_DEFAULT = true
WEB_FALLBACK_SIM_THRESHOLD = 0.22
WEB_MAX_RESULTS = 3
WEB_MAX_CHARS_PER_PAGE = 15000
WEB_CONTEXT_MAX_CHARS = 6000

"""

from __future__ import annotations

import hmac
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import openai
import streamlit as st
from zoneinfo import ZoneInfo

# Optional dependency for web fallback (recommended)
try:
    import requests  # type: ignore
except Exception:
    requests = None

# ---------------------------------------------------------
# ✅ MUST be the first Streamlit command
# ---------------------------------------------------------
st.set_page_config(
    page_title="MECH503 Chatbot",
    page_icon="🧠",
    layout="wide",
)

# =========================================================
# Paths / constants
# =========================================================
TZ = ZoneInfo("America/Vancouver")
APP_ROOT = Path(__file__).resolve().parent

NOTES_JSON = APP_ROOT / "notes" / "merged_notes.chunks.json"

# Cache for embeddings (notes)
CACHE_DIR = APP_ROOT / ".cache_mech503"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE_PATH = CACHE_DIR / (NOTES_JSON.stem + ".embeddings.npy")
IDX_CACHE_PATH = CACHE_DIR / (NOTES_JSON.stem + ".index.json")

# Quota persistence
QUOTA_DIR = APP_ROOT / ".quota_mech503"
QUOTA_FILE = QUOTA_DIR / "usage.json"

MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 6  # retrieval chunks

# Trusted domains allowlist (edit to taste)
TRUSTED_DOMAINS = [
    "nist.gov",
    "nasa.gov",
    "asme.org",
    "aiaa.org",
    "royalsocietypublishing.org",
    "sciencedirect.com",
    "springer.com",
    "wikipedia.org",
    "solidmechanics.org"
]

# =========================================================
# Password gate
# =========================================================
def require_password() -> None:
    """Simple course password gate."""
    expected = st.secrets.get("COURSE_PASSWORD", "")
    if not expected:
        st.warning("COURSE_PASSWORD is not set in secrets. Password gate is disabled.")
        return

    st.sidebar.subheader("Course access")
    pw = st.sidebar.text_input("Password", type="password")

    if not pw:
        st.info("Enter the course password in the sidebar to continue.")
        st.stop()

    if not hmac.compare_digest(pw.strip(), str(expected).strip()):
        st.error("Incorrect password.")
        st.stop()

# =========================================================
# Quota manager
# =========================================================
def _today_key() -> str:
    return datetime.now(TZ).date().isoformat()

def _load_usage() -> dict:
    QUOTA_DIR.mkdir(parents=True, exist_ok=True)
    if not QUOTA_FILE.exists():
        return {}
    try:
        return json.loads(QUOTA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_usage(data: dict) -> None:
    QUOTA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = QUOTA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(QUOTA_FILE)

def _get_limits() -> Tuple[int, int, int]:
    max_q_session = int(st.secrets.get("MAX_Q_PER_SESSION", 15))
    max_q_day = int(st.secrets.get("MAX_Q_PER_DAY", 250))
    max_tok_day = int(st.secrets.get("MAX_TOKENS_PER_DAY", 30000))
    return max_q_session, max_q_day, max_tok_day

def quota_check_before_call(tokens_to_add: int = 0) -> None:
    """Block if user has hit caps (per session/per day)."""
    max_q_session, max_q_day, max_tok_day = _get_limits()

    # session counts
    st.session_state.setdefault("q_count_session", 0)

    usage = _load_usage()
    today = _today_key()
    usage.setdefault(today, {"q_count": 0, "tokens": 0})

    q_sess = int(st.session_state["q_count_session"])
    q_day = int(usage[today]["q_count"])
    t_day = int(usage[today]["tokens"])

    if q_sess >= max_q_session:
        st.warning(f"Session limit reached ({max_q_session} questions). Try again later.")
        st.stop()
    if q_day >= max_q_day:
        st.warning(f"Daily limit reached ({max_q_day} questions). Try again tomorrow.")
        st.stop()
    if t_day + tokens_to_add > max_tok_day:
        st.warning(f"Daily token budget reached ({max_tok_day} tokens). Try again tomorrow.")
        st.stop()

def quota_record_after_call(tokens_used: int) -> None:
    max_q_session, _, _ = _get_limits()
    st.session_state["q_count_session"] = int(st.session_state.get("q_count_session", 0)) + 1

    usage = _load_usage()
    today = _today_key()
    usage.setdefault(today, {"q_count": 0, "tokens": 0})
    usage[today]["q_count"] = int(usage[today]["q_count"]) + 1
    usage[today]["tokens"] = int(usage[today]["tokens"]) + int(tokens_used)
    _save_usage(usage)

def render_usage_meter() -> None:
    """Sidebar usage remaining."""
    max_q_session, max_q_day, max_tok_day = _get_limits()
    usage = _load_usage()
    today = _today_key()
    usage.setdefault(today, {"q_count": 0, "tokens": 0})

    q_sess = int(st.session_state.get("q_count_session", 0))
    q_day = int(usage[today]["q_count"])
    t_day = int(usage[today]["tokens"])

    st.sidebar.subheader("Usage remaining")
    st.sidebar.caption("Limits reduce abuse and protect budget.")
    st.sidebar.progress(max(0.0, min(1.0, (max_q_session - q_sess) / max(1, max_q_session))))
    st.sidebar.write(f"Session questions: **{max(0, max_q_session - q_sess)} / {max_q_session}**")

    st.sidebar.progress(max(0.0, min(1.0, (max_q_day - q_day) / max(1, max_q_day))))
    st.sidebar.write(f"Daily questions: **{max(0, max_q_day - q_day)} / {max_q_day}**")

    st.sidebar.progress(max(0.0, min(1.0, (max_tok_day - t_day) / max(1, max_tok_day))))
    st.sidebar.write(f"Daily tokens: **{max(0, max_tok_day - t_day)} / {max_tok_day}**")

# =========================================================
# Notes loading (JSON)
# =========================================================
@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_notes(json_path: str) -> Tuple[List[str], List[str]]:
    """Load notes chunks + metadata from JSON.

    Supports multiple schemas:
      A) dict with keys: {"chunks":[...], "meta":[...]}   (legacy)
      B) dict with keys: {"records":[{"text":..., "page":..., "chunk_id":...}, ...]} (textbook-safe)
      C) list of strings: ["chunk1", "chunk2", ...]
      D) list of dicts: [{"text":..., "meta":...}, ...]  (flexible)
    """
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    # --- Case C/D: top-level list ---
    if isinstance(data, list):
        # list of strings
        if all(isinstance(x, str) for x in data):
            chunks = [x.strip() for x in data if x and x.strip()]
            meta = [""] * len(chunks)
            return chunks, meta

        # list of dicts
        if all(isinstance(x, dict) for x in data):
            chunks: List[str] = []
            meta: List[str] = []
            for i, r in enumerate(data):
                # try common keys
                txt = (r.get("text") or r.get("chunk") or r.get("content") or "").strip()
                if not txt:
                    continue
                m = (
                    r.get("meta")
                    or r.get("source")
                    or r.get("label")
                    or f"{p.stem} | item {i}"
                )
                chunks.append(txt)
                meta.append(str(m))
            return chunks, meta

        raise ValueError("Unsupported JSON list schema for notes (expected list[str] or list[dict]).")

    # --- Case A/B: dict ---
    if not isinstance(data, dict):
        raise ValueError("Unsupported JSON schema for notes (expected dict or list).")

    # 1) textbook-safe records
    if "records" in data and isinstance(data["records"], list):
        chunks: List[str] = []
        meta: List[str] = []
        title = data.get("title", p.stem)
        source = data.get("source", p.name)
        for r in data["records"]:
            if not isinstance(r, dict):
                continue
            txt = (r.get("text") or "").strip()
            if not txt:
                continue
            page = r.get("page", "?")
            cid = r.get("chunk_id", "")
            meta.append(f"{title} | {source} | p.{page} | {cid}")
            chunks.append(txt)
        return chunks, meta

    # 2) legacy chunks/meta
    chunks = data.get("chunks", []) or []
    meta = data.get("meta", [""] * len(chunks)) or [""] * len(chunks)
    if len(meta) != len(chunks):
        meta = (meta + [""] * len(chunks))[: len(chunks)]
    return chunks, meta

def compute_or_load_embeddings(client: openai.OpenAI, chunks: List[str]) -> np.ndarray:
    """Compute embeddings once and cache to disk."""
    if EMB_CACHE_PATH.exists() and IDX_CACHE_PATH.exists():
        try:
            idx = json.loads(IDX_CACHE_PATH.read_text(encoding="utf-8"))
            if idx.get("n_chunks") == len(chunks) and idx.get("model") == EMBEDDING_MODEL:
                emb = np.load(EMB_CACHE_PATH)
                if emb.shape[0] == len(chunks):
                    return emb
        except Exception:
            pass

    BATCH = 128
    embs = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embs.extend([d.embedding for d in resp.data])

    emb = np.array(embs, dtype=np.float32)
    np.save(EMB_CACHE_PATH, emb)
    IDX_CACHE_PATH.write_text(json.dumps({"n_chunks": len(chunks), "model": EMBEDDING_MODEL}, indent=2), encoding="utf-8")
    return emb

# =========================================================
# Retrieval utilities
# =========================================================
def _cosine_sims(q_vec: np.ndarray, X: np.ndarray) -> np.ndarray:
    qn = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return (Xn @ qn).astype(np.float32)

def mmr(q_vec: np.ndarray, X: np.ndarray, k: int = 6, lambda_mult: float = 0.7) -> List[int]:
    """Maximal Marginal Relevance selection using cosine similarity."""
    sims = _cosine_sims(q_vec, X)
    if sims.size == 0:
        return []
    k = min(k, sims.size)
    selected: List[int] = []
    candidates = list(range(sims.size))

    # pick best first
    first = int(np.argmax(sims))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        mmr_scores = []
        for c in candidates:
            sim_to_query = float(sims[c])
            sim_to_selected = float(np.max(_cosine_sims(X[c], X[selected]))) if selected else 0.0
            score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
            mmr_scores.append((score, c))
        _, best = max(mmr_scores, key=lambda t: t[0])
        selected.append(best)
        candidates.remove(best)
    return selected

# =========================================================
# Markdown helpers (math + local images)
# =========================================================
def _fmt_math(md: str) -> str:
    """Keep as-is; placeholder hook for your existing math formatting."""
    return md

IMG_PATTERN = re.compile(r"!\[\]\(([^)]+)\)")

def render_markdown_with_local_images(md: str) -> None:
    """Render markdown, pulling out image links and showing them with st.image."""
    image_paths = IMG_PATTERN.findall(md)
    cleaned = IMG_PATTERN.sub("", md).strip()
    st.markdown(_fmt_math(cleaned), unsafe_allow_html=True)

    for raw in image_paths:
        # handle paths like "static/media/..", "media/..", etc.
        path = raw.strip()
        if path.startswith("static/"):
            img_file = APP_ROOT / path
        elif path.startswith("media/"):
            img_file = APP_ROOT / "static" / path
        else:
            # treat as relative to project root
            img_file = APP_ROOT / path

        if img_file.exists():
            st.image(str(img_file))
        else:
            st.caption(f"⚠️ Missing image: {path}")

# =========================================================
# Trusted web fallback (Tavily)
# =========================================================
def _get_web_settings() -> Tuple[bool, float, int, int, int]:
    enabled_default = bool(st.secrets.get("WEB_FALLBACK_ENABLED_DEFAULT", True))
    sim_thr = float(st.secrets.get("WEB_FALLBACK_SIM_THRESHOLD", 0.22))
    max_results = int(st.secrets.get("WEB_MAX_RESULTS", 3))
    max_chars_page = int(st.secrets.get("WEB_MAX_CHARS_PER_PAGE", 15000))
    ctx_max = int(st.secrets.get("WEB_CONTEXT_MAX_CHARS", 6000))
    return enabled_default, sim_thr, max_results, max_chars_page, ctx_max

def tavily_search(query: str, api_key: str, max_results: int) -> List[Dict[str, str]]:
    if requests is None:
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_domains": TRUSTED_DOMAINS,
                "search_depth": "basic",
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("results", []) or []
    except Exception:
        return []

def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = re.sub(r"\s{2,}", " ", html)
    return html.strip()

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_page_text_cached(url: str, max_chars: int) -> str:
    if requests is None:
        return ""
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = _html_to_text(r.text)
        return text[:max_chars]
    except Exception:
        return ""

def _chunk_chars(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    out = []
    start = 0
    n = len(text)
    overlap = max(0, min(overlap, chunk_size // 2))
    while start < n:
        end = min(n, start + chunk_size)
        window = text[start:end]
        cut = window.rfind(". ")
        if cut != -1 and cut > int(0.6 * len(window)):
            end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = end - overlap if end - overlap > start else end
    return out

def build_web_context(
    client: openai.OpenAI,
    q_vec: np.ndarray,
    query: str,
    tavily_key: str,
    max_results: int,
    max_chars_page: int,
    ctx_max_chars: int,
) -> Tuple[str, List[str]]:
    """Search + fetch + chunk + embed + retrieve web snippets."""
    results = tavily_search(query, tavily_key, max_results=max_results)
    if not results:
        return "", []

    web_chunks: List[str] = []
    web_meta: List[str] = []

    for r in results:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip() or url
        if not url:
            continue

        page_text = fetch_page_text_cached(url, max_chars=max_chars_page)
        if not page_text:
            continue

        chunks = _chunk_chars(page_text, chunk_size=1200, overlap=200)[:8]
        for ch in chunks:
            web_chunks.append(ch)
            web_meta.append(f"{title} | {url}")

    if not web_chunks:
        return "", []

    # Embed web chunks
    web_embeds: List[np.ndarray] = []
    for i in range(0, len(web_chunks), 96):
        batch = web_chunks[i : i + 96]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        web_embeds.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])

    W = np.vstack(web_embeds).astype(np.float32)
    sims = _cosine_sims(q_vec, W)
    top = np.argsort(sims)[-min(6, sims.size):][::-1].tolist()

    # Build compact context
    ctx_parts = []
    used_meta = []
    total = 0
    for idx in top:
        block = f"[{web_meta[idx]}]\n{web_chunks[idx]}\n"
        if total + len(block) > ctx_max_chars:
            break
        ctx_parts.append(block)
        used_meta.append(web_meta[idx])
        total += len(block)

    return "\n\n".join(ctx_parts).strip(), used_meta

# =========================================================
# App UI
# =========================================================
st.title("MECH503: Linear Elasticity – Course Chatbot")

# Password + usage meter
require_password()
render_usage_meter()

# Keys
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it to .streamlit/secrets.toml or Streamlit Cloud secrets.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Web settings
web_enabled_default, WEB_FALLBACK_SIM_THRESHOLD, WEB_MAX_RESULTS, WEB_MAX_CHARS_PER_PAGE, WEB_CONTEXT_MAX_CHARS = _get_web_settings()
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.environ.get("TAVILY_API_KEY", ""))

with st.sidebar:
    st.subheader("Web fallback")
    web_enabled = st.toggle(
        "Allow trusted web fallback",
        value=web_enabled_default,
        help="Only activates when note retrieval confidence is low.",
    )
    st.caption(f"Trusted domains: {', '.join(TRUSTED_DOMAINS[:5])}{'…' if len(TRUSTED_DOMAINS)>5 else ''}")

# Load notes
if not NOTES_JSON.exists():
    st.error(f"Notes JSON not found: {NOTES_JSON}.")
    st.stop()

chunks, meta = load_notes(str(NOTES_JSON))
if not chunks:
    st.error("No chunks found in notes JSON.")
    st.stop()

with st.spinner("Loading embeddings (first run may take a few minutes)…"):
    embeddings = compute_or_load_embeddings(client, chunks)

# Session chat
if "chat" not in st.session_state:
    st.session_state.chat = []

# Render history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        render_markdown_with_local_images(msg["content"]) if msg["role"] == "assistant" else st.markdown(msg["content"])

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

    # Similarities + confidence
    sims_notes = _cosine_sims(q_vec, embeddings)
    top_sim = float(np.max(sims_notes)) if sims_notes.size else 0.0

    # Notes retrieval via MMR
    top_ids = mmr(q_vec, embeddings, k=TOP_K)
    context_chunks = [chunks[i] for i in top_ids]
    context_meta = [meta[i] for i in top_ids]
    notes_context = "\n\n".join(context_chunks).strip()

    # Web fallback
    web_context = ""
    web_meta: List[str] = []
    used_web = False

    if web_enabled and top_sim < WEB_FALLBACK_SIM_THRESHOLD:
        if not TAVILY_API_KEY:
            st.sidebar.warning("Web fallback enabled, but TAVILY_API_KEY is missing.")
        elif requests is None:
            st.sidebar.warning("Web fallback enabled, but 'requests' is not installed.")
        else:
            with st.sidebar:
                st.info("Using trusted web fallback (low note match).")
            web_context, web_meta = build_web_context(
                client=client,
                q_vec=q_vec,
                query=prompt,
                tavily_key=TAVILY_API_KEY,
                max_results=WEB_MAX_RESULTS,
                max_chars_page=WEB_MAX_CHARS_PER_PAGE,
                ctx_max_chars=WEB_CONTEXT_MAX_CHARS,
            )
            used_web = bool(web_context.strip())

    # Sidebar diagnostics (always visible after a question)
    with st.sidebar:
        st.subheader("Retrieval diagnostics")
        st.write("Top note similarity:", f"{top_sim:.3f}")
        st.write("Used web fallback:", used_web)
        if used_web and web_meta:
            st.write("Web sources used:")
            for s in list(dict.fromkeys(web_meta)):
                st.write("-", s)

    # Compose final context
    full_context = "COURSE NOTES:\n" + notes_context
    if used_web and web_context:
        full_context += "\n\nTRUSTED WEB:\n" + web_context

    # System prompt (prevents guessing when context is weak)
    sys_prompt = (
        "You are an engineering TA for linear elasticity (MECH503). "
        "Answer clearly using markdown; all math must be LaTeX ($ for inline, $$ on blank lines for display). "
        "Base your answer ONLY on the CONTEXT provided. "
        "If the context is insufficient, say what is missing and suggest what to consult."
    )

    user_msg = f"CONTEXT:\n{full_context}\n\nQUESTION: {prompt}\nAnswer in markdown."

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_accum = ""
        with st.spinner("Thinking…"):
            # Quota check before completion call
            quota_check_before_call()

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                stream=True,
            )

            # We don't get accurate tokens during streaming; approximate later
            for chunk in resp:
                token = chunk.choices[0].delta.content or ""
                answer_accum += token
                placeholder.markdown(_fmt_math(answer_accum), unsafe_allow_html=True)

        # Record usage: approximate tokens from characters (rough but stable)
        # If you have access to usage tokens from non-stream responses, swap this out.
        approx_tokens = max(1, len(answer_accum) // 4)
        quota_record_after_call(tokens_used=approx_tokens)

        # Build sources (notes + web)
        unique_notes = list(dict.fromkeys(context_meta))
        notes_md = "\n".join(f"- {s}" for s in unique_notes) if unique_notes else "- (none)"

        if used_web:
            unique_web = list(dict.fromkeys(web_meta))
            web_md = "\n".join(f"- {s}" for s in unique_web) if unique_web else "- (none)"
            sources_md = f"**Course notes**\n{notes_md}\n\n**Trusted web**\n{web_md}"
        else:
            sources_md = f"**Course notes**\n{notes_md}"

        answer_full = f"{answer_accum}\n\n---\n**Sources**\n{sources_md}"

        st.session_state.chat.append({"role": "assistant", "content": answer_full})
        placeholder.markdown(_fmt_math(answer_full), unsafe_allow_html=True)

    # Debug view
    with st.expander("🔍 Retrieved Context Chunks (Debug View)", expanded=False):
        st.markdown(f"**Top note similarity:** `{top_sim:.3f}` (threshold `{WEB_FALLBACK_SIM_THRESHOLD:.3f}`)")
        st.markdown("**Notes chunks used:**")
        for i in top_ids:
            preview = chunks[i][:220].replace("\n", " ").strip() + "..."
            st.markdown(f"- **{meta[i]}** — `{preview}`")

        if used_web:
            st.markdown("**Trusted web sources used:**")
            for w in list(dict.fromkeys(web_meta)):
                st.markdown(f"- {w}")
