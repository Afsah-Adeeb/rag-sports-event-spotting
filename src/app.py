"""
Streamlit web interface for the RAG pipeline.

This is a presentation layer over retrieve.py/generate.py -- all retrieval and
generation logic stays framework-agnostic so cli.py, evaluate.py, and this app
share one implementation.

TWO WAYS PAPERS GET INTO THE APP
----------------------------------
1. The committed corpus: PDFs in data/papers/, chunked by ingest.py and
   embedded into data/vector_store/ by embed_store.py, both committed to git.
   Permanent, and what a fresh visitor sees.
2. Session uploads (the "Add papers" box below): the PDF is chunked and
   embedded in memory and appended to a per-session copy of the FAISS index.
   This is deliberately NOT written to disk, because:
     - Streamlit Community Cloud's filesystem is ephemeral -- anything written
       at runtime is lost on the next restart/redeploy, so "saving" it would be
       a lie that silently loses the user's papers.
     - The hosted app is public. Persisting uploads server-side would let any
       visitor permanently mutate the corpus everyone else queries.
   So uploads answer questions immediately, for that browser session only, and
   the README documents the git route for permanently growing the corpus.
"""

import os
from collections import defaultdict

import faiss
import streamlit as st

import config

# When deployed to Streamlit Community Cloud there is no .env file -- the key
# comes from Streamlit's secrets manager instead. Copy it into the environment
# *before* generate.py's client is first constructed, so generate.py can stay
# framework-agnostic and keep reading a plain env var (it works unchanged for
# cli.py and evaluate.py, which know nothing about Streamlit).
# Accessing st.secrets raises if no secrets.toml exists, which is the normal
# case when running locally off a .env file -- so this is best-effort only.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

from generate import generate_answer_from_chunks  # noqa: E402  (must follow the secrets bridge)
from ingest import process_pdf_bytes  # noqa: E402
from retrieve import base_index_and_metadata, embed_texts, search  # noqa: E402

st.set_page_config(
    page_title="Sports Event-Spotting Research Assistant",
    page_icon="⚽",
    layout="wide",
)

# --- Styling ---------------------------------------------------------------
# Colors are expressed as translucent overlays on top of Streamlit's own
# background rather than hardcoded hex values, so the cards stay legible in
# both the light and dark themes without maintaining two palettes.
st.markdown(
    """
    <style>
      .hero { padding: .25rem 0 1.25rem 0; }
      .hero h1 { font-size: 2.1rem; margin: 0 0 .35rem 0; letter-spacing: -.02em; }
      .hero p { opacity: .7; margin: 0; font-size: 1.02rem; }

      .stat-row { display: flex; gap: .75rem; flex-wrap: wrap; margin: .25rem 0 1.25rem 0; }
      .stat {
        flex: 1 1 120px; padding: .8rem 1rem; border-radius: 12px;
        background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.20);
      }
      .stat .num { font-size: 1.6rem; font-weight: 650; line-height: 1.1; }
      .stat .lbl { font-size: .78rem; opacity: .65; text-transform: uppercase; letter-spacing: .06em; }

      .paper {
        padding: .85rem 1rem; border-radius: 12px; margin-bottom: .6rem;
        background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.18);
      }
      .paper .title { font-weight: 600; line-height: 1.35; margin-bottom: .4rem; }
      .paper .meta { font-size: .82rem; opacity: .62; }
      .pill {
        display: inline-block; padding: .12rem .5rem; border-radius: 999px;
        font-size: .7rem; font-weight: 600; letter-spacing: .03em;
        background: rgba(255,140,0,.18); color: #ff8c00; margin-left: .4rem;
        vertical-align: middle;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Data loading ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model and vector index...")
def load_base():
    """Load the committed FAISS index once per server process (shared by all sessions)."""
    return base_index_and_metadata()


def pretty_title(filename):
    return filename[:-4] if filename.lower().endswith(".pdf") else filename


def corpus_stats(metadata):
    """Aggregate per-paper chunk counts and page extents for the library view."""
    by_paper = defaultdict(lambda: {"chunks": 0, "max_page": 0})
    for chunk in metadata:
        entry = by_paper[chunk["source_paper"]]
        entry["chunks"] += 1
        entry["max_page"] = max(entry["max_page"], chunk["page_end"])
    return by_paper


def active_index_and_metadata():
    """Base corpus, plus anything uploaded this session, as one searchable pair.

    Uses faiss.clone_index so session vectors are appended to a *copy* -- the
    cached base index is shared across every visitor's session and must never
    be mutated by one user's upload.
    """
    base_index, base_metadata = load_base()
    session_chunks = st.session_state.get("uploaded_chunks", [])
    if not session_chunks:
        return base_index, base_metadata

    if st.session_state.get("session_index_size") != len(session_chunks):
        index = faiss.clone_index(base_index)
        index.add(embed_texts([c["text"] for c in session_chunks]))
        st.session_state.session_index = index
        st.session_state.session_index_size = len(session_chunks)

    return st.session_state.session_index, base_metadata + session_chunks


# --- Startup checks: fail with a readable message, not a stack trace -------
if not config.FAISS_INDEX_PATH.exists():
    st.error(
        "No vector index found. Run `ingest.py` then `embed_store.py` first "
        f"(expected {config.FAISS_INDEX_PATH})."
    )
    st.stop()

if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "GEMINI_API_KEY is not set. Locally: copy `.env.example` to `.env` and add your key. "
        "On Streamlit Cloud: add it under Settings -> Secrets. "
        "Get a free key at https://aistudio.google.com/apikey."
    )
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploaded_chunks", [])

index, metadata = active_index_and_metadata()
stats = corpus_stats(metadata)
session_papers = {c["source_paper"] for c in st.session_state.uploaded_chunks}

# --- Header ------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>⚽ Sports Event-Spotting Research Assistant</h1>'
    "<p>Ask questions across research papers on sports video event-spotting and "
    "temporal action localization. Every answer is grounded in retrieved passages and cited.</p></div>",
    unsafe_allow_html=True,
)

chat_tab, library_tab = st.tabs(["💬  Ask", f"📚  Library ({len(stats)})"])

with library_tab:
    total_chunks = sum(s["chunks"] for s in stats.values())
    total_pages = sum(s["max_page"] for s in stats.values())
    st.markdown(
        f"""
        <div class="stat-row">
          <div class="stat"><div class="num">{len(stats)}</div><div class="lbl">Papers</div></div>
          <div class="stat"><div class="num">{total_pages:,}</div><div class="lbl">Pages</div></div>
          <div class="stat"><div class="num">{total_chunks:,}</div><div class="lbl">Searchable chunks</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Add papers")
    uploaded = st.file_uploader(
        "Drop PDFs here to query them right away",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.caption(
        "Uploaded papers are searchable **for this browser session only** — they are not saved to the "
        "server. To add papers permanently, commit them to `data/papers/` and re-run the pipeline "
        "(see the README)."
    )

    known = {c["source_paper"] for c in st.session_state.uploaded_chunks} | set(stats)
    new_files = [f for f in (uploaded or []) if f.name not in known]
    if new_files:
        with st.spinner(f"Chunking and embedding {len(new_files)} paper(s)..."):
            added = []
            for file in new_files:
                try:
                    chunks = process_pdf_bytes(file.name, file.getvalue())
                except Exception as exc:
                    st.error(f"Could not read **{file.name}**: {exc}")
                    continue
                if not chunks:
                    st.warning(f"No extractable text in **{file.name}** (scanned image PDF?). Skipped.")
                    continue
                st.session_state.uploaded_chunks.extend(chunks)
                added.append(f"{pretty_title(file.name)} ({len(chunks)} chunks)")
        if added:
            st.success("Added " + "; ".join(added))
            st.rerun()

    st.subheader("In the corpus")
    for paper, s in sorted(stats.items()):
        pill = '<span class="pill">THIS SESSION</span>' if paper in session_papers else ""
        st.markdown(
            f'<div class="paper"><div class="title">{pretty_title(paper)}{pill}</div>'
            f'<div class="meta">{s["max_page"]} pages · {s["chunks"]} chunks</div></div>',
            unsafe_allow_html=True,
        )

with chat_tab:
    controls, _ = st.columns([1, 2])
    with controls:
        st.slider(
            "Passages to retrieve per question",
            min_value=1, max_value=10, value=config.DEFAULT_TOP_K,
            key="top_k",  # read below via session_state, since chat_input lives outside this tab
            help="How many chunks are pulled from the index and shown to the model as context.",
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Sources ({len(message['sources'])})"):
                    for chunk in message["sources"]:
                        st.markdown(
                            f"**{pretty_title(chunk['source_paper'])}** — p.{chunk['page_start']}-"
                            f"{chunk['page_end']} · similarity {chunk['score']:.3f}"
                        )
                        st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

question = st.chat_input("Ask a question about the papers...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Retrieving passages and generating a grounded answer..."):
        sources = search(question, index, metadata, top_k=st.session_state.get("top_k", config.DEFAULT_TOP_K))
        answer = generate_answer_from_chunks(question, sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()
