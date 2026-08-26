# Sports Event-Spotting Research Assistant

**Live demo: [rag-event-spotting.streamlit.app](https://rag-event-spotting.streamlit.app/)**

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

There are two routes, and they do different things.

### Route 1 — Upload in the browser (instant, temporary)

Open the **Library** tab in the app and drop PDFs into the upload box. They're chunked and embedded
on the spot and become answerable within a few seconds.

**These uploads live only in your browser session.** They are not written to the server, and they
disappear when the app restarts or you reload. That's deliberate, for two reasons:
- Streamlit Community Cloud's filesystem is ephemeral — anything written at runtime is wiped on the
  next restart or redeploy, so "saving" it would silently lose your papers.
- The hosted app is public. Persisting uploads server-side would let any visitor permanently change
  the corpus everyone else queries.

Use this for a quick "what does this new paper say?" — or to demo the pipeline on someone else's PDF.

### Route 2 — Commit to the repo (permanent)

This is how the corpus actually grows.

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

**To update the live demo:** commit and push the new PDFs *and* the regenerated `data/vector_store/`,
then Streamlit Cloud auto-redeploys within a minute or two:
```
git add data/ && git commit -m "Add papers" && git push
```

## Evaluation

Replace the example questions in `eval/test_questions.json` with your own (same
`{"question": ..., "correct_papers": [...]}` format), then:
```
cd "D:\RAG System\src"
..\.venv\Scripts\python.exe evaluate.py
```
Prints Hit Rate@k / Precision@k to the terminal and writes `eval/results.md` with every generated
answer next to its retrieved sources, for manual faithfulness review.

## What's committed and why

Unusually for a repo, the PDFs (`data/papers/`) and the prebuilt FAISS index (`data/vector_store/`)
are committed rather than ignored. That's deliberate: it makes the repo self-contained, so a hosted
demo can answer questions immediately without a rebuild step, and anyone cloning it gets a working
system without supplying their own corpus. The cost is repo size (~56MB) and the fact that the full
extracted text of copyrighted papers lives in the history. The only thing never committed is `.env`
(and `.streamlit/secrets.toml`) — the API key.

## Deployment

### Do I need to keep a server running forever?

Not on your own machine, no. Running `streamlit run app.py` locally only works while that terminal is
open and only on your computer — that's fine for demos you drive yourself, but it's not a link you can
put on a resume. To get a permanent public URL, you host it somewhere that runs the server for you.

### Option 1: Streamlit Community Cloud (free, recommended)

Free, purpose-built for exactly this, and connects straight to a GitHub repo.

- **How it works:** you point it at this repo and `src/app.py`; it installs `requirements.txt` and runs
  the app on their infrastructure. You get a permanent `*.streamlit.app` URL.
- **Auto-deploys:** every push to `main` redeploys automatically. No manual redeploy step.
- **Sleeping:** free apps sleep after ~7 days with no visitors. They wake automatically on the next
  visit (a ~30s cold start). The URL never dies — it's dormant, not deleted. Perfectly fine for a
  resume link.
- **Your API key:** do NOT commit it. In the app's Settings → Secrets, add:
  ```toml
  GEMINI_API_KEY = "your_actual_key"
  ```
  `app.py` already reads this automatically (see the `st.secrets` bridge at the top of the file).
- **Resource limits:** ~1GB RAM on the free tier. This app loads a 384-dim MiniLM model plus a small
  FAISS index, so it fits comfortably.

Steps: push to GitHub → sign in at share.streamlit.io with GitHub → "New app" → pick the repo, branch
`main`, main file path `src/app.py` → add the secret above → Deploy.

### Option 2: Hugging Face Spaces (free)

Also free and always-on (no sleeping), and arguably a better fit culturally for an ML portfolio since
recruiters in ML already browse HF profiles. Requires a small `app.py`-at-root convention change or a
Space config file, and secrets go in Space Settings → Variables and secrets.

### Option 3: Keep it local

Entirely legitimate. Record a short screen capture of the app answering a few questions and put the GIF
in this README — many interviewers prefer a 20-second demo video over clicking a live link anyway, and
it costs nothing and never breaks.

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
