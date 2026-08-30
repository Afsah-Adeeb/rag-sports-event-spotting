# CLAUDE.md
## Project

A Retrieval-Augmented Generation (RAG) Q&A chatbot over a personal collection of research papers on
sports video event-spotting / temporal action localization (SoccerNet, E2E-Spot, T-DEED, CALF, and
related work). Built as a resume/portfolio project to demonstrate RAG fundamentals for ML/DS interviews —
code is deliberately simple (no agents, no multi-hop retrieval, no complex chains) and every design
decision is meant to be explainable in an interview.

## Environment

- Python 3.11, virtualenv at `.venv/`.
- All commands below assume the venv is active or invoked directly via `.venv/Scripts/python.exe` (Windows).
- Secrets live in `.env` (git-ignored), based on `.env.example`. Requires `GEMINI_API_KEY` — free key from
  https://aistudio.google.com/apikey.

```bash
# Setup
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Run each pipeline stage from src/ (scripts use bare `import config`, i.e. relative to src/, not the project root)
cd src
../.venv/Scripts/python.exe ingest.py          # PDFs in data/papers/ -> data/chunks.jsonl
../.venv/Scripts/python.exe embed_store.py     # chunks.jsonl -> FAISS index in data/vector_store/
../.venv/Scripts/python.exe retrieve.py "question here"   # manual retrieval sanity check
../.venv/Scripts/python.exe generate.py "question here"   # full retrieve+generate, one-shot
../.venv/Scripts/python.exe cli.py             # interactive Q&A loop
../.venv/Scripts/python.exe evaluate.py        # runs eval/test_questions.json -> eval/results.md
../.venv/Scripts/python.exe telemetry.py       # terminal summary of the runtime monitoring log
../.venv/Scripts/python.exe measure_threshold.py  # re-derive config.LOW_CONFIDENCE_THRESHOLD
../.venv/Scripts/streamlit.exe run app.py      # web GUI (chat + library + metrics)
```

There is no test suite, linter, or build step - this is a linear script pipeline, not a package.

## Architecture

The pipeline is six sequential, independently-runnable stages, each a script in `src/` that reads the
previous stage's output file from `data/`. There's no orchestrator - you run them in order by hand.

```
data/papers/*.pdf
  -> ingest.py       -> data/chunks.jsonl
  -> embed_store.py  -> data/vector_store/index.faiss + chunk_metadata.jsonl
  -> retrieve.py      (library: retrieve(question, top_k) -> chunks, used by generate.py/evaluate.py)
  -> generate.py       (library + CLI: retrieve + Gemini call -> grounded, cited answer)
  -> cli.py            (thin interactive wrapper around generate.answer_traced)
  -> app.py             (Streamlit web GUI: Ask / Library / Metrics tabs)
  -> evaluate.py        reads eval/test_questions.json -> eval/results.md
  -> telemetry.py       both front ends -> data/telemetry/events.jsonl -> Metrics tab
```

`cli.py` and `app.py` are two interchangeable front ends over the same `generate.answer_traced()` --
all retrieval/generation logic stays framework-agnostic in `retrieve.py`/`generate.py`; the front ends
only handle presentation. Both log through `telemetry.py`, so monitoring covers the whole pipeline
rather than only whoever used the GUI.

`generate.py` exposes a small ladder of entry points, narrowest first: `generate_with_usage()` (chunks
in, answer + tokens + timing out), `answer_traced()` (adds retrieval and times both stages -- what the
front ends use), and the two original convenience wrappers `generate_answer_from_chunks()` /
`generate_answer()` kept for `evaluate.py` and backwards compatibility.

All tunable parameters (paths, chunk size/overlap, embedding model name, Gemini model name, top-k,
eval k-values) live in one place: `src/config.py`. Change knobs there, not inline in the scripts.

### Key design decisions (see docstrings in each file for the full reasoning)

- **Chunking** (`ingest.py`): fixed-size (~1000 chars, 200-char overlap), but paragraph-aware — chunks
  only break on blank-line boundaries, not mid-sentence. Chosen over semantic/embedding-based chunking
  for simplicity, since academic papers already have strong structural signal (headings, paragraphs).
  Each chunk carries `source_paper` filename and `page_start`/`page_end` for citation and eval.
- **Boilerplate stripping** (`ingest.py: find_boilerplate_lines`): PDF running headers/footers (paper
  title, page number) repeat across most pages without a blank-line separator from body text, so they'd
  otherwise get glued onto real sentences. Detected generically per-PDF: any line whose text (with digits
  normalized to `#`, so page numbers don't break the match) recurs on a large fraction of that PDF's pages
  is dropped.
- **Vector store** (`embed_store.py`): FAISS `IndexFlatIP` (exact brute-force, cosine similarity via
  normalized embeddings) over local `sentence-transformers/all-MiniLM-L6-v2` embeddings (free, CPU,
  384-dim). Chosen over Chroma deliberately — FAISS forces explicit management of the ID-to-metadata
  mapping (`chunk_metadata.jsonl`, written in the same order as vectors are added to the index), which is
  more instructive and easier to defend in an interview than a database client hiding that mechanism.
  Flat/exact search is fine at this corpus scale (thousands of chunks); would need IVF/HNSW well beyond that.
- **Generation** (`generate.py`): Gemini (`google-genai` SDK, model name in `config.GEMINI_MODEL_NAME`).
  The prompt in `build_prompt()` enforces answer-only-from-context, explicit "I don't know" when context
  is insufficient, and citation of source paper filenames — this is what makes faithfulness checkable at
  all in `evaluate.py`. Gemini model availability shifts over time; if `generate.py`/`evaluate.py` starts
  raising `404 NOT_FOUND` on the model name, list currently available models with
  `client.models.list()` (see `src/generate.py` for client setup) rather than guessing a replacement name.
- **Monitoring** (`telemetry.py`): every answered question, from both front ends, is appended to
  `data/telemetry/events.jsonl` as a JSON line; the app's Metrics tab reads it back. Three decisions
  worth knowing before editing it: (1) the log is **append-only** — feedback arrives after the answer
  was logged and is written as a separate `feedback` event keyed by `query_id`, then merged in
  `build_records()`, because concurrent Streamlit sessions appending single lines is safe while
  read-modify-rewrite races and loses records; (2) **raw signals are stored, judgement flags are
  recomputed on read** in `_derive_flags()`, so improving the refusal heuristic retroactively fixes
  historical records instead of leaving the dashboard showing verdicts from an older rule;
  (3) every public function **swallows its own exceptions** — monitoring must never break the thing it
  monitors. The headline metric is `hallucination_risk` (weak retrieval + a confident answer);
  `over_refusal` is its inverse. `looks_like_refusal()` is position-aware on purpose — matching refusal
  phrases anywhere in the answer misclassified answers that end with a caveat, so the marker must appear
  in the first ~150 chars. `config.LOW_CONFIDENCE_THRESHOLD` is measured by `measure_threshold.py`, not
  guessed; re-run it whenever the corpus or embedding model changes. `evaluate.py` deliberately does
  *not* log telemetry, so batch eval runs don't skew production metrics.
- **Evaluation** (`evaluate.py`): retrieval quality is automated (Hit Rate@k and Precision@k against
  hand-labeled `correct_papers` per question in `eval/test_questions.json`); answer faithfulness is
  deliberately *not* automated (no LLM-as-judge) — `evaluate.py` instead writes a report
  (`eval/results.md`) pairing each generated answer with its retrieved chunks for manual review. Note
  the terminology distinction preserved in the code comments: Hit Rate@k ("did we find the right paper
  at all") vs. strict Precision@k ("how much noise is in the top-k") are commonly conflated but different.

### Data flow contracts between stages

- `chunks.jsonl` / `chunk_metadata.jsonl` records: `{chunk_id, source_paper, chunk_index, page_start,
  page_end, text}`.
- `chunk_metadata.jsonl` line `i` must stay in the same order as vector `i` in `index.faiss` — FAISS
  returns bare integer positions, and that positional alignment is the only thing that maps a search hit
  back to a paper/page. If you ever rebuild the index, always regenerate both files together via
  `embed_store.py` in one run; don't hand-edit one without the other.
- `eval/test_questions.json`: list of `{question, correct_papers: [filename, ...]}`. `correct_papers` can
  list multiple filenames when more than one paper legitimately covers the same topic.

### Adding more papers

Drop additional PDFs into `data/papers/` and re-run `ingest.py` then `embed_store.py` — both are
idempotent full rebuilds (they regenerate `chunks.jsonl` / the FAISS index from scratch each run), not
incremental. No other code changes needed to grow the corpus.
