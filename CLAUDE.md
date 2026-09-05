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
../.venv/Scripts/python.exe export_text.py     # PDFs -> data/papers_text/*.txt (for agentic arm)
../.venv/Scripts/python.exe agent_rag.py "question"   # agentic (grep/read loop) answer
../.venv/Scripts/python.exe compare_rag.py     # semantic vs agentic -> eval/comparison.md
../.venv/Scripts/python.exe measure_threshold.py  # re-derive config.LOW_CONFIDENCE_THRESHOLD

# Evaluation suite. The first four make NO API calls (retrieval scoring is local).
../.venv/Scripts/python.exe evaluate.py        # retrieval: hit rate, MRR, coverage, per type
../.venv/Scripts/python.exe baselines.py       # vs BM25 and random -> eval/baselines.md
../.venv/Scripts/python.exe sweep.py           # chunk size x top-k grid -> eval/sweep.md
../.venv/Scripts/python.exe robustness.py      # typos/casual/keywords -> eval/robustness.md
../.venv/Scripts/python.exe evaluate.py --generate --pause 4.5   # + answers, facts, refusal
../.venv/Scripts/python.exe closed_book.py --pause 4.5           # -> eval/closed_book.md
../.venv/Scripts/python.exe judge.py           # LLM judge cross-check -> eval/judge.md

../.venv/Scripts/streamlit.exe run app.py      # web GUI (chat + library + metrics + evaluation)
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

The evaluation suite is a second, parallel structure -- every script imports its scoring
from `eval_core.py` and writes one markdown report plus its section of `eval/summary.json`:

```
eval/test_questions.json  (55 labelled questions: 40 answerable + 15 unanswerable)
  -> eval_core.py       metrics, Wilson/bootstrap intervals, fact matching, retry, caching
       -> evaluate.py     retrieval + (--generate) answers, facts, refusal -> results.md
       -> baselines.py    vs BM25 and random                               -> baselines.md
       -> closed_book.py  with vs without documents                        -> closed_book.md
       -> sweep.py        chunk size x top-k grid, in-memory indexes       -> sweep.md
       -> robustness.py   typos / casual / keyword-only variants           -> robustness.md
       -> judge.py        LLM judge cross-check, reads the answer cache    -> judge.md
  -> eval/summary.json  <- headline numbers, read by the app's Evaluation tab
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
- **Agentic comparison arm** (`export_text.py`, `agent_tools.py`, `agent_rag.py`, `compare_rag.py`):
  a second retrieval strategy with no index at all — the model drives `list_papers`/`grep_papers`/
  `read_paper` in a loop. Exists to *measure* the semantic-vs-agentic tradeoff rather than assert it.
  The Gemini SDK's automatic function calling is deliberately disabled and the loop written by hand,
  for the same reason FAISS was chosen over Chroma: the mechanism under study should be visible.
  Fairness constraints that must not be broken when editing: both arms run the same model
  (`config.BENCHMARK_MODEL_NAME`, separate from `GEMINI_MODEL_NAME` because free quota is
  per-model-per-day and a benchmark run would otherwise exhaust the live app's budget), and
  `export_text.py` reuses `ingest.py`'s extraction/cleaning so neither side gets cleaner text.
  Two tooling bugs were found by measurement and are worth not reintroducing: (1) tool output needs a
  **byte** budget, not just a count — after paragraph joining a single "line" ran to thousands of
  chars and one grep returned 127K chars (~113K tokens); (2) grep spends its budget **round-robin
  across papers**, because draining it in filename order meant a broad pattern never reached the
  alphabetically-later paper that actually answered the question, which looked like an agentic-retrieval
  failure but was a tool bug. `compare_rag.py` reports cost ratios and Hit Rate but deliberately not
  Precision@k (undefined comparably: semantic returns exactly k, agentic returns what it chose to open),
  and prints a sample-size warning below 15 questions.
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
- **Evaluation** (`eval_core.py` + six scripts): all scoring primitives live in `eval_core.py` so every
  arm scores identically — the comparisons between them are the whole point. Test set is 55 labelled
  questions in `eval/test_questions.json` (40 answerable + 15 deliberately unanswerable), each with
  `correct_papers`, `must_mention` facts, and a `type` tag. Things not to undo when editing:
  - **Two interval types, on purpose.** Proportions get a **Wilson** interval, means get a seeded
    **bootstrap**. Bootstrapping a proportion degenerates at the boundary (40/40 resamples to
    `[1.00, 1.00]`, claiming certainty 40 questions cannot support); Wilson gives `[0.91, 1.00]`.
    The bootstrap seed is fixed so intervals do not wobble between runs.
  - **Each required fact is a LIST of accepted spellings**, satisfied if any appears. Directly caused
    by the `Temporal-Discriminability` hyphen bug — one hyphen silently zeroed a score.
  - **Hit Rate@k must not be quoted for `multi_paper` questions.** They accept 3-4 of 9 papers, so
    five *random* chunks score the same 0.86 as the real system. `papers_covered()` is the metric
    that means something there (0.37, i.e. retrieval finds one paper and stops).
  - **Baselines are not strawmen** (`baselines.py`): BM25 proper, not word-overlap counting, plus a
    seeded random floor. Measured result: BM25 0.85 vs semantic 0.93 with *overlapping intervals* —
    the test set cannot establish the embeddings win overall; they clearly win on `paraphrase` only.
    Random uses `zlib.crc32`, not `hash()`, which is salted per process and made the "reproducible"
    floor move between runs.
  - **`sweep.py` never touches the committed index** — candidate indexes are built in memory. It also
    holds overlap at a constant *fraction* of chunk size, or the comparison confounds two variables.
  - **`closed_book.py` gets its own prompt**, not `build_prompt()` with empty context. The RAG prompt
    orders answering only from context, so with no context it would refuse everything and the
    experiment would measure the prompt rather than the model's memory.
  - **`judge.py` is a cross-check, never the headline.** It judges every answer twice and reports
    self-consistency *before* any verdict, because the 2026 literature finds judge-human correlation
    around 0.55 and far worse position bias in Flash-class models than Pro-class ones — and this
    project runs a Flash-class model for cost. Its real output is where it disagrees with the
    deterministic fact check.
  - Answers are cached to `eval/.answers_cache.json` so `judge.py` grades the *same* answers the fact
    checker graded; regenerating would make any disagreement possibly just resampling.
  - Headline numbers go to `eval/summary.json` via `ec.update_summary()`; the app's Evaluation tab
    reads that rather than parsing the markdown reports, so rewording a report cannot break the UI.
  - Terminology distinction preserved in the code comments: Hit Rate@k ("did we find the right paper
    at all") vs. strict Precision@k ("how much noise is in the top-k") are commonly conflated.
  - **Every number is corpus-specific.** Re-run all of these plus `measure_threshold.py` after adding
    papers; the random floor in particular falls as the corpus grows.

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
