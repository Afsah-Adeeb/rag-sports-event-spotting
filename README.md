# Sports Event-Spotting Research Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about a personal collection of
research papers on sports video event-spotting / temporal action localization, with cited, grounded answers.

## First-time setup

1. Install dependencies (only needed once, or after `requirements.txt` changes):
   ```
   "D:\RAG System\.venv\Scripts\python.exe" -m pip install -r "D:\RAG System\requirements.txt"
   ```
2. Get a free Gemini API key: https://aistudio.google.com/apikey
3. Copy `.env.example` to `.env` and paste your key in.
4. Put your PDFs in `data/papers/`.
5. Build the index (see "Adding more papers" below for exact commands).

## Day-to-day: asking questions

**Web GUI** (recommended):
```
"D:\RAG System\.venv\Scripts\streamlit.exe" run "D:\RAG System\src\app.py"
```
Opens in your browser at `localhost:8501`. Leave the terminal window open while using it; `Ctrl+C` in
that terminal stops the server.

**Command line** (no browser):
```
cd "D:\RAG System\src"
..\.venv\Scripts\python.exe cli.py
```
Type a question, press Enter. Type `exit` to quit.

Both do the exact same thing under the hood — same retrieval, same Gemini calls, same sources.

## Adding more papers

**Where:** Drop the new PDF files directly into `data/papers/`. No renaming, no folders — just the file itself.

**What to run:** From `src/`, run these two, in this order:
```
cd "D:\RAG System\src"
..\.venv\Scripts\python.exe ingest.py
..\.venv\Scripts\python.exe embed_store.py
```

**What actually happens:**
- `ingest.py` re-reads **every** PDF currently in `data/papers/` (old ones + new ones) and rewrites
  `data/chunks.jsonl` from scratch — it's a full rebuild, not an incremental add.
- `embed_store.py` re-reads `chunks.jsonl` and rewrites the FAISS index (`data/vector_store/`) from
  scratch, same reasoning.
- Nothing here is "training" a model — it's just re-chunking text and re-computing embeddings with the
  same fixed local model each time. No cost, no waiting on a training job. For ~30 papers this takes
  well under a minute total.

**What to refresh:** If `app.py` (Streamlit) or `cli.py` was already running, it's holding the *old*
index in memory — restart it (stop with `Ctrl+C`, run the launch command again) so it picks up the new
index. Just refreshing the browser tab is not enough.

## Evaluation

Replace the example questions in `eval/test_questions.json` with your own (same
`{"question": ..., "correct_papers": [...]}` format), then:
```
cd "D:\RAG System\src"
..\.venv\Scripts\python.exe evaluate.py
```
Prints Hit Rate@k / Precision@k to the terminal and writes `eval/results.md` with every generated
answer next to its retrieved sources, for manual faithfulness review.

## Troubleshooting

- **`'D:\RAG' is not recognized...`** — you're in `cmd.exe` and a path with a space wasn't quoted.
  Quote the *entire* command from the start: `"D:\RAG System\.venv\Scripts\python.exe" script.py`.
- **`&&` or `;` throws a parser error** — you're mixing shell syntax. In `cmd.exe`, `&&` chains commands
  and `cd` + `run` need to be separate lines or already-quoted; PowerShell's separator is also `;`, not `&&`.
- **Streamlit asks for an email on first run** — that's Streamlit's own one-time onboarding prompt, not
  part of this app. Leave it blank and press Enter; it won't ask again.
- **`GEMINI_API_KEY not set`** — `.env` is missing or empty. Copy `.env.example` to `.env` and add your key.
- **`404 NOT_FOUND` on the Gemini model name** — Google occasionally deprecates model versions. List what's
  currently available and update `GEMINI_MODEL_NAME` in `src/config.py` (see comment there for the exact
  snippet).
