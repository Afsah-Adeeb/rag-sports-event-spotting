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

THREE TABS
------------
Ask      -- the chat interface.
Library  -- what's in the corpus, plus the upload box.
Metrics  -- what the system actually did: latency per stage, retrieval
            confidence, flagged answers, and user ratings. Backed by
            telemetry.py; see that module for the logging design.
"""

import os
import uuid
from collections import defaultdict

import faiss
import pandas as pd
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

import telemetry  # noqa: E402  (must follow the secrets bridge)
from generate import answer_traced  # noqa: E402
from ingest import process_pdf_bytes  # noqa: E402
from retrieve import base_index_and_metadata, embed_texts  # noqa: E402

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
      .stat .sub { font-size: .74rem; opacity: .5; margin-top: .15rem; }

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
      .pill.warn { background: rgba(220,50,50,.16); color: #e2554f; }
      .pill.ok   { background: rgba(40,170,90,.16);  color: #2faa5f; }

      .flagged {
        padding: .8rem 1rem; border-radius: 12px; margin-bottom: .6rem;
        background: rgba(220,50,50,.07); border: 1px solid rgba(220,50,50,.22);
      }
      .flagged .q { font-weight: 600; margin-bottom: .3rem; }
      .flagged .meta { font-size: .8rem; opacity: .65; }
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


def stat_card(num, label, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="stat"><div class="num">{num}</div><div class="lbl">{label}</div>{sub_html}</div>'


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
st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])

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

chat_tab, library_tab, metrics_tab = st.tabs(
    ["💬  Ask", f"📚  Library ({len(stats)})", "📊  Metrics"]
)

# =============================================================================
# LIBRARY TAB
# =============================================================================
with library_tab:
    total_chunks = sum(s["chunks"] for s in stats.values())
    total_pages = sum(s["max_page"] for s in stats.values())
    st.markdown(
        '<div class="stat-row">'
        + stat_card(len(stats), "Papers")
        + stat_card(f"{total_pages:,}", "Pages")
        + stat_card(f"{total_chunks:,}", "Searchable chunks")
        + "</div>",
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

# =============================================================================
# METRICS TAB
# =============================================================================
with metrics_tab:
    records = telemetry.build_records()
    summary = telemetry.summarize(records)

    if not summary:
        st.info(
            "No questions logged yet. Ask something in the **Ask** tab and the "
            "monitoring data will appear here."
        )
    else:
        st.markdown(
            '<div class="stat-row">'
            + stat_card(summary["total_queries"], "Questions answered")
            + stat_card(
                f"{summary['median_total_ms'] / 1000:.1f}s",
                "Median latency",
                f"p95 {summary['p95_total_ms'] / 1000:.1f}s",
            )
            + stat_card(
                f"{summary['mean_top_score']:.2f}",
                "Mean retrieval score",
                f"flagged weak below {config.LOW_CONFIDENCE_THRESHOLD}",
            )
            + stat_card(
                f"{summary['satisfaction']:.0%}" if summary["satisfaction"] is not None else "—",
                "Satisfaction",
                f"{summary['up_count']}👍 {summary['down_count']}👎"
                if summary["rated_count"] else "no ratings yet",
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        # --- Where the time goes -------------------------------------------
        # Splitting these two is the point: retrieval is local CPU work in the
        # tens of milliseconds, generation is a network call to Gemini taking
        # seconds. A single "latency" number would hide that ~99% of the wait
        # is the LLM, and that optimizing FAISS would be wasted effort.
        st.subheader("Where the time goes")
        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                '<div class="stat-row">'
                + stat_card(f"{summary['median_retrieval_ms']:.0f} ms", "Retrieval (median)", "embed + FAISS search")
                + stat_card(f"{summary['median_generation_ms'] / 1000:.1f} s", "Generation (median)", "Gemini API call")
                + "</div>",
                unsafe_allow_html=True,
            )
            gen_share = (
                summary["median_generation_ms"]
                / max(summary["median_total_ms"], 1e-9) * 100
            )
            st.caption(
                f"Generation is **{gen_share:.1f}%** of total latency. Retrieval is local "
                "CPU work; the wait is almost entirely the LLM API call, so that is where "
                "any latency optimisation has to happen."
            )
        with right:
            recent = records[-30:]
            st.caption("Latency per question (last 30, seconds)")
            st.bar_chart(
                pd.DataFrame(
                    {
                        "retrieval": [r["retrieval_ms"] / 1000 for r in recent],
                        "generation": [r["generation_ms"] / 1000 for r in recent],
                    }
                ),
                height=220,
            )

        # --- Retrieval confidence -------------------------------------------
        st.subheader("Retrieval confidence")
        labels, counts = telemetry.score_histogram(records)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.bar_chart(
                pd.DataFrame({"questions": counts}, index=labels), height=240
            )
            st.caption(
                "Top-1 cosine similarity between the question and its best-matching chunk. "
                f"Below **{config.LOW_CONFIDENCE_THRESHOLD}** the index had nothing that really "
                "matched — see `measure_threshold.py` for how that cutoff was measured."
            )
        with c2:
            st.markdown(
                '<div class="stat-row" style="flex-direction:column">'
                + stat_card(summary["weak_retrieval_count"], "Weak retrieval", "nothing matched well")
                + stat_card(summary["refused_count"], "Model declined", "said it didn't know")
                + "</div>",
                unsafe_allow_html=True,
            )

        # --- The metric that matters ----------------------------------------
        st.subheader("Flagged for review")
        st.caption(
            "**Hallucination risk** = retrieval found nothing relevant, but the model "
            "answered confidently anyway instead of saying it didn't know. Those answers "
            "are built on irrelevant context and are the ones worth reading by hand."
        )

        flagged = [r for r in records if r.get("hallucination_risk")]
        over_refused = [r for r in records if r.get("over_refusal")]

        f1, f2 = st.columns(2)
        with f1:
            st.markdown(
                '<div class="stat-row">'
                + stat_card(
                    len(flagged),
                    "Hallucination risk",
                    "weak retrieval + confident answer",
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        with f2:
            st.markdown(
                '<div class="stat-row">'
                + stat_card(
                    len(over_refused),
                    "Over-refusal",
                    "good retrieval + declined anyway",
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        if flagged:
            for r in reversed(flagged[-10:]):
                st.markdown(
                    f'<div class="flagged"><div class="q">{r["question"]}</div>'
                    f'<div class="meta">top score {r["top_score"]:.3f} '
                    f'· {r["ts"]} · {r["surface"]}</div></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Answer it gave"):
                    st.markdown(r["answer"])
        else:
            st.success("No answers flagged — every low-confidence retrieval was declined correctly.")

        # --- Corpus coverage --------------------------------------------------
        # Which papers real questions actually reach. A paper that never gets
        # retrieved is either off-topic for what people ask, or chunked/extracted
        # badly enough that its embeddings never match anything.
        st.subheader("Corpus coverage")
        hits = summary["paper_hits"]
        used = [(pretty_title(p), n) for p, n in hits.most_common()]
        never = sorted(pretty_title(p) for p in stats if p not in hits)

        if used:
            st.bar_chart(
                pd.DataFrame({"times retrieved": [n for _, n in used]},
                             index=[p[:45] for p, _ in used]),
                height=max(200, 32 * len(used)),
                horizontal=True,
            )
        if never:
            st.warning(
                "**Never retrieved:** " + ", ".join(never)
                + ". Either no one asked about these, or their text extracted badly."
            )

        # --- Raw log -----------------------------------------------------------
        with st.expander(f"Query log ({len(records)} entries)"):
            table = pd.DataFrame([
                {
                    "time": r["ts"],
                    "question": r["question"][:70],
                    "top score": r["top_score"],
                    "latency (s)": round(r["total_ms"] / 1000, 2),
                    "in/out tokens": f'{r.get("input_tokens") or "?"}/{r.get("output_tokens") or "?"}',
                    "declined": "yes" if r["refused"] else "",
                    "flagged": "⚠" if r.get("hallucination_risk") else "",
                    "rating": {"up": "👍", "down": "👎"}.get(r.get("rating"), ""),
                }
                for r in reversed(records)
            ])
            st.dataframe(table, use_container_width=True, hide_index=True)

            if config.TELEMETRY_LOG_PATH.exists():
                st.download_button(
                    "Download raw event log (JSONL)",
                    data=config.TELEMETRY_LOG_PATH.read_bytes(),
                    file_name="rag_telemetry.jsonl",
                    mime="application/x-ndjson",
                )

        st.caption(
            "⚠️ On Streamlit Community Cloud the filesystem is ephemeral, so this log covers "
            "the current app instance only and resets on restart or redeploy. Running locally, "
            "it persists in `data/telemetry/events.jsonl`."
        )

# =============================================================================
# ASK TAB
# =============================================================================
with chat_tab:
    controls, _ = st.columns([1, 2])
    with controls:
        st.slider(
            "Passages to retrieve per question",
            min_value=1, max_value=10, value=config.DEFAULT_TOP_K,
            key="top_k",  # read below via session_state, since chat_input lives outside this tab
            help="How many chunks are pulled from the index and shown to the model as context.",
        )

    for i, message in enumerate(st.session_state.messages):
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

            # Feedback capture. The rating is written as its own telemetry event
            # keyed by query_id, rather than editing the already-written query
            # record -- see telemetry.py for why the log is append-only.
            if message.get("query_id"):
                if message.get("rating"):
                    st.caption(
                        f"Thanks — recorded {'👍' if message['rating'] == 'up' else '👎'}."
                    )
                else:
                    up, down, meta = st.columns([1, 1, 10])
                    if up.button("👍", key=f"up_{i}", help="This answer was useful"):
                        telemetry.log_feedback(message["query_id"], "up")
                        message["rating"] = "up"
                        st.rerun()
                    if down.button("👎", key=f"down_{i}", help="This answer was wrong or unhelpful"):
                        telemetry.log_feedback(message["query_id"], "down")
                        message["rating"] = "down"
                        st.rerun()
                    if message.get("timing"):
                        meta.caption(message["timing"])

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

question = st.chat_input("Ask a question about the papers...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    top_k = st.session_state.get("top_k", config.DEFAULT_TOP_K)

    with st.spinner("Retrieving passages and generating a grounded answer..."):
        result = answer_traced(question, top_k=top_k, index=index, metadata=metadata)

    query_id = telemetry.log_query(
        question=question,
        answer=result["answer"],
        retrieved_chunks=result["chunks"],
        retrieval_ms=result["retrieval_ms"],
        generation_ms=result["generation_ms"],
        top_k=top_k,
        usage=result,
        surface="app",
        session_id=st.session_state.session_id,
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["chunks"],
        "query_id": query_id,
        "rating": None,
        "timing": f"{result['retrieval_ms']:.0f} ms retrieval · "
                  f"{result['generation_ms'] / 1000:.1f} s generation",
    })
    st.rerun()
