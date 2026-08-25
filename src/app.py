"""
Step 6 (GUI variant): Streamlit web interface for the RAG pipeline.

This is a thin UI layer over generate.generate_answer() -- all retrieval and
generation logic stays in retrieve.py/generate.py so the "brains" of the
project remain framework-agnostic and reusable by cli.py, evaluate.py, and
this app alike. app.py's only job is presentation: chat history, a sidebar
with corpus info, and a readable sources display per answer.

Run with:  streamlit run app.py
"""

import os

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

from generate import generate_answer  # noqa: E402  (must follow the secrets bridge above)

st.set_page_config(
    page_title="Sports Event-Spotting Research Assistant",
    page_icon="⚽",
    layout="wide",
)

# --- Sidebar: what this app is, and what's loaded --------------------------
with st.sidebar:
    st.title("⚽ Research Assistant")
    st.caption("RAG chatbot over papers on sports video event-spotting / temporal action localization.")

    st.subheader("How it works")
    st.markdown(
        "1. Your question is embedded (`all-MiniLM-L6-v2`)\n"
        "2. The most relevant chunks are retrieved from a FAISS index\n"
        "3. Gemini answers **using only those chunks**, and cites them"
    )

    st.subheader(f"Loaded papers ({len(list(config.PAPERS_DIR.glob('*.pdf')))})")
    for pdf_path in sorted(config.PAPERS_DIR.glob("*.pdf")):
        st.markdown(f"- {pdf_path.stem}")

    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=config.DEFAULT_TOP_K)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Startup checks: fail with a readable message, not a stack trace -------
if not config.FAISS_INDEX_PATH.exists():
    st.error(
        "No vector index found. Run `ingest.py` then `embed_store.py` first "
        f"(expected {config.FAISS_INDEX_PATH})."
    )
    st.stop()

if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "GEMINI_API_KEY not set. Copy .env.example to .env and add your key "
        "(get one free at https://aistudio.google.com/apikey), then restart the app."
    )
    st.stop()

# --- Chat state --------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources?}

st.title("Ask about the papers")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for chunk in message["sources"]:
                    st.markdown(
                        f"- **{chunk['source_paper']}** "
                        f"(p.{chunk['page_start']}-{chunk['page_end']}, score={chunk['score']:.3f})"
                    )

question = st.chat_input("Ask a question about the loaded papers...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving chunks and generating answer..."):
            answer, sources = generate_answer(question, top_k=top_k)
        st.markdown(answer)
        with st.expander("Sources"):
            for chunk in sources:
                st.markdown(
                    f"- **{chunk['source_paper']}** "
                    f"(p.{chunk['page_start']}-{chunk['page_end']}, score={chunk['score']:.3f})"
                )

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
