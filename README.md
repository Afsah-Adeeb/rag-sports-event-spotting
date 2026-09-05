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

The system is measured against a hand-labelled test set of **55 questions** in
`eval/test_questions.json`: 41 the corpus can answer, and 14 it deliberately cannot.
Each question carries the paper(s) that count as a correct retrieval, the facts a good
answer must contain, and a type tag (`simple`, `paraphrase`, `comparison`, `multi_paper`,
`unanswerable`).

Everything below is reproducible from that one file. The **🧪 Evaluation** tab in the app
shows the same numbers without running anything.

### Headline

| Metric | Score [95% CI] |
|---|---|
| Hit Rate@5 | 0.93 [0.81–0.97] |
| Hit Rate@1 | 0.63 [0.48–0.76] |
| MRR | 0.72 [0.61–0.83] |
| Precision@5 | 0.52 [0.41–0.62] |
| **Correctly refused unanswerable questions** | **1.00 [0.78–1.00]** |

Every figure carries a confidence interval, and the interval type is chosen per metric.
Proportions use a **Wilson** interval; means use a **seeded bootstrap**. This is not
decoration: bootstrapping a proportion collapses at the boundary — 40 hits out of 40
resamples to 40/40 every time and reports `[1.00, 1.00]`, claiming certainty no 40-question
test set can support. Wilson gives `[0.91, 1.00]`, which is the honest answer.

### Against baselines, because a number alone is not a result

`baselines.py` runs the same questions through two alternatives with the same scoring code:

| Metric | semantic | BM25 | random |
|---|---|---|---|
| Hit Rate@5 | 0.93 | 0.85 | 0.46 |
| Hit Rate@1 | 0.63 | 0.54 | 0.20 |
| MRR | 0.724 | 0.652 | 0.296 |

BM25 is the real keyword-ranking algorithm, not a word-overlap strawman — beating a
strawman would prove nothing. And the honest reading of that table is that **the
intervals overlap: this test set cannot establish that the embeddings beat keyword
search overall.**

Where they *do* clearly win is on reworded questions, which is exactly what embeddings
are for:

| Type | semantic | BM25 |
|---|---|---|
| simple | 0.94 | 0.89 |
| **paraphrase** | **0.88** | **0.75** |
| comparison | 1.00 | 0.86 |

The specific cases are more convincing than the average. *"Is there a benchmark for a
racket sport played indoors on a small table?"* never says "table tennis"; *"what happens
when the people labelling the videos put the timestamp slightly in the wrong place?"*
never says "temporal misalignment". Semantic retrieval finds both. Character matching
cannot.

### What retrieval is actually worth

Every other number here measures the system *with* retrieval. None of them prove retrieval
is what produced the answer — and that matters, because these are nine well-known arXiv
papers that were almost certainly in the model's training data.

So `closed_book.py` asks all 55 questions twice with the same model: once through the full
pipeline, once with **no documents at all**.

| | Full RAG | No documents | Retrieval buys |
|---|---|---|---|
| Required facts present | 0.71 [0.62–0.79] | 0.58 [0.47–0.69] | **+0.13** |
| **Refused the unanswerable questions** | **0.87** [0.62–0.96] | **0.00** [0.00–0.20] | **+0.87** |

Read those two rows together, because they say opposite things and the second one is the
point.

**On knowledge, retrieval barely helps.** It buys 13 points. On 27 of the 40 answerable
questions the model produced at least half the required facts *with nothing supplied* —
it already knew them. Retrieval won 12 questions, tied 22, and actively lost 6. Anyone
quoting 0.71 as what this pipeline achieves is claiming credit the model earned in
pre-training.

**On honesty, retrieval is the entire system.** Handed no documents, the model answered
**every single one** of the 15 questions the corpus cannot answer — a 0% refusal rate, 15
confident fabrications. Handed retrieved chunks that visibly fail to contain the answer,
it declined 13 of 15.

That inverts the usual pitch for RAG on a well-known corpus. The value here is not that
the model knows more. It is that it knows **when to stop** — and that is a property
retrieval creates rather than improves. It is also the honest answer to "why not just ask
Gemini directly?": you would get answers of roughly similar quality, and fifteen
fabrications you had no way to detect.

*Measurement note: this table was produced on the pre-relabel test set (15 unanswerable
questions, before `U02` was found to be mislabelled and reclassified). The fact-coverage
figures move only marginally; the 0.87 RAG refusal rate is understated -- on the corrected
set the same answers give 1.00. The refresh run hit the 500/day free-tier cap, so the
table stays as measured rather than being quietly adjusted by hand.*

The closed-book arm is given its own prompt rather than the RAG prompt with an empty
context block. The RAG prompt orders the model to answer only from provided context, so
with no context it would refuse everything and score zero — measuring the prompt instead
of the model's memory.

### The two flagged hallucinations were both bugs in the evaluation

The first scored run flagged two answers as hallucinations — the system answering
questions the corpus supposedly cannot answer. Reading both by hand, **neither was the
system's fault.**

**`U02` — "How does T-DEED perform on the SoccerNet Ball Action Spotting benchmark?"**
Written as a deliberate trap on the reasoning that the T-DEED paper mentions that
benchmark without evaluating on it. True — but the *SoccerNet* chapter covers T-DEED's
win in detail and prints its scores in a results table. The answer was in the corpus, in
a different paper. The system answered correctly and was marked wrong because the label
was wrong. It is now `S19`, an answerable question, with the note kept.

**`U12` — "Do any of these methods support live streaming input?"**
The answer began *"The provided text does not explicitly confirm whether..."*, which is a
refusal. `telemetry.looks_like_refusal()` matches a fixed list of phrases, and every entry
was a blunt form — "does not contain", "cannot answer". None matched the hedged form. The
marker list now includes them.

With both fixed, correct refusal goes from 0.87 to **1.00 [0.78–1.00]** — the system
declined every one of the 14 questions it should have, and hallucinated on none.

Three things worth taking from that:

- **A wrong label looks exactly like a model failure.** Both flags were indistinguishable
  from real hallucinations in the aggregate. Only reading the individual cases separated
  them, which is the argument for reports that name specific questions rather than just
  reporting a rate.
- **Re-scoring must be free, or you won't do it.** `evaluate.py --from-cache` re-applies
  the scoring to cached answers with no API calls, so fixing the heuristic and seeing the
  corrected number took seconds instead of 55 more generations. Same principle as the
  monitoring layer, which stores raw signals and recomputes judgement on read.
- **The interval still refuses to overclaim.** 14 out of 14 reports as `[0.78, 1.00]`, not
  `[1.00, 1.00]`. Fourteen questions cannot establish perfection, and the metric says so.

### Three findings worth reading

**1. Retrieval confidence cannot detect unanswerable questions.**
`config.LOW_CONFIDENCE_THRESHOLD` was measured against deliberately off-topic questions
("What is the capital of Brazil?") and separated them cleanly. Against questions that are
*on-topic in wording but unanswerable from the corpus*, it fails completely: **14 of the
15 score above the threshold**, the highest at 0.706, while genuine questions drop as low
as 0.332. Cosine similarity measures whether a passage is *about* your question, not
whether it *answers* it, and no threshold value recovers that difference. The
`hallucination_risk` flag on the Metrics tab inherits the blind spot. Documented in
`config.py` rather than tuned away.

**2. Hit Rate@k is the wrong metric for multi-paper questions.**
On questions that accept three or four papers out of nine, five *random* chunks almost
always touch one, so random retrieval scores the same 0.86 as the real system. Paper
coverage — the fraction of *all* correct papers reached — tells the real story: 0.37.
Top-k retrieval concentrates on whichever single paper matches best, and cosine similarity
has no term for diversity. That is a fixable property of the retriever, not a limit of
embeddings.

**3. Typos hurt more than bad grammar.**
`robustness.py` re-asks every question in three degraded forms, generated deterministically:

| Variant | Hit Rate@5 | Same top paper as original |
|---|---|---|
| original | 0.93 | — |
| typo | 0.76 | 0.46 |
| casual | 0.80 | 0.85 |
| keywords only | 0.90 | 0.80 |

Stripping stopwords barely matters. Typos cost 17 points and change the top-ranked paper
more than half the time. Stripping punctuation is its own hazard: it turns `T-DEED` into
`tdeed` and `E2E-Spot` into `e2espot`, which is the same hyphen failure that broke a
lexical search for `Temporal-Discriminability` earlier in this project.

### The LLM judge, and why it is not the headline metric

`judge.py` grades every answer for whether the retrieved context actually supports it —
the standard LLM-as-judge setup. It runs against the *same cached answers* the
deterministic fact check scored, so any difference between the two is a difference in
judgement rather than in which answers were sampled.

| | Result |
|---|---|
| Judge agrees with **itself** (same answer, judged twice) | 0.95 [0.85–0.98] |
| Judge agrees with the **deterministic fact check** | 0.51 [0.36–0.66] |
| Judge agrees with the **refusal heuristic** | 0.75 [0.62–0.84] |
| Answers the judge flagged that fact-checking passed | **0** |

Read the first two rows together. The judge is **highly repeatable and agrees with almost
nothing else.** That is precisely the failure mode the 2026 literature warns about:
judges with test-retest reliability above 0.95 have been measured carrying severe
systematic bias at the same time. Repeatability is a precondition for trust, not evidence
of it — and a project that ran the judge once and reported "0.95 consistency" would have
looked rigorous while measuring nothing.

The last row is the one that decides its usefulness here. The whole argument for a judge
is that it catches fluent, confident, unsupported answers a string-matching metric cannot
see. **It found zero.** Every disagreement ran the other way — 20 cases where required
facts were missing but the judge passed the answer as well-grounded, which is a retrieval
gap the judge is not equipped to notice.

It also under-detects refusals badly: it returned `REFUSED` on 5 answers when the system
actually declined roughly 19.

One more detail worth keeping: self-consistency measured **0.98 on one run and 0.95 on
the next**, same code, same answers, different calls. The stability metric is itself
unstable at this sample size.

So the judge stays in the repo, wired up and reported — but the deterministic fact check
stays the headline. Being able to build one, measure it, and conclude it did not earn
promotion is a better answer than either using it uncritically or avoiding it.

### The settings are measured, not guessed

`sweep.py` re-chunks and re-embeds the whole corpus at every chunk size and scores every
top-k, building each candidate index **in memory** so the committed index is never
touched. Overlap is held at a constant *fraction* of chunk size, so the comparison
measures chunk size rather than chunk size and redundancy together.

MRR across the grid:

| chunk size | k=3 | k=5 | k=10 |
|---|---|---|---|
| 500 | 0.696 | 0.702 | 0.715 |
| **1000** | 0.683 | **0.717** | 0.730 |
| 1500 | 0.708 | 0.708 | 0.724 |

Published ablations suggested ~500 characters might beat 1000, which is why it was worth
running. On this corpus it does not: every cell sits within measurement error of every
other. The defensible answer to "why 1000?" is therefore not "it's optimal" but **"it
doesn't matter much, and here is the grid that shows it"** — and k=10 buys +0.013 MRR for
double the prompt tokens on every query, which is not a trade worth making.

### Running it

```
cd "D:\RAG System\src"

..\.venv\Scripts\python.exe evaluate.py        # retrieval scores          (free)
..\.venv\Scripts\python.exe baselines.py       # vs BM25 and random        (free)
..\.venv\Scripts\python.exe sweep.py           # chunk size / top-k grid   (free)
..\.venv\Scripts\python.exe robustness.py      # typos, casual, keywords   (free)

..\.venv\Scripts\python.exe evaluate.py --generate --pause 4.5  # answers + refusal
..\.venv\Scripts\python.exe closed_book.py --pause 4.5          # what retrieval is worth
..\.venv\Scripts\python.exe judge.py                            # LLM judge cross-check
```

The first four make **no API calls at all** — retrieval scoring is local. That is what
makes the sweep affordable and what keeps the default `evaluate.py` run free. The last
three generate answers and are behind explicit flags.

They run on `config.BENCHMARK_MODEL_NAME`, not the model the live app uses, because
free-tier quota is per-model-per-day and a 55-question run on the app's model would
exhaust the deployed demo's budget. `--pause 4.5` respects the measured 15-requests-minute
free-tier limit; the retry logic is the safety net, not the plan.

Reports land in `eval/` and are committed, so the Evaluation tab loads without re-running
anything. **All of these numbers are corpus-specific — re-run every script after adding
papers**, including `measure_threshold.py`.

## Semantic vs Agentic RAG (the comparison experiment)

The main pipeline is **semantic RAG**: pre-embed every chunk, embed the question, return the nearest
5. One retrieval step, decided by vector similarity.

`agent_rag.py` implements the alternative, **agentic RAG** — the pattern behind coding agents like
Claude Code and Cursor. There is no index at all. The model gets three tools (`list_papers`,
`grep_papers`, `read_paper`) and searches the papers itself, reading results and deciding what to look
at next, until it can answer.

`compare_rag.py` runs both over the same labelled questions and writes `eval/comparison.md`:
```
cd "D:\RAG System\src"
..\.venv\Scripts\python.exe export_text.py     # one-time: PDFs -> greppable text
..\.venv\Scripts\python.exe compare_rag.py
```

**What is held constant**, so the result measures retrieval strategy and nothing else: the same model
for both arms (`config.BENCHMARK_MODEL_NAME`), the same extracted text (`export_text.py` reuses
`ingest.py`'s extraction and cleaning), and the same questions and ground truth that `evaluate.py` uses.

Current result on 3 questions — cost ratios are meaningful, the hit rate is **not** (see the sample-size
warning the report prints; ~15-20 questions are needed before a pass/fail proportion is worth quoting):

| Metric | Semantic | Agentic |
|---|---|---|
| Median latency | 6.1s | 12.1s (2.0x) |
| Mean input tokens | 1,357 | 5,296 (3.9x) |
| API round trips | 1.0 | 2.3 |

The cost gap is structural, not incidental: every turn of the agentic loop resends the entire
conversation *including every tool result*, so tokens compound with each search.

**Why agentic retrieval struggles on papers specifically.** Grep matches characters, embeddings match
meaning. Asked about "temporal discriminability", a lexical search misses the T-DEED paper entirely —
that paper writes `Temporal-Discriminability`, hyphenated, and never as two plain words. A hyphen
defeats the match; an embedding does not notice it. The agentic loop can recover by retrying with a
shorter pattern, which is its real advantage, but it has to spend turns doing so.

## Monitoring

Evaluation tells you if the system is good *on questions you chose, offline*. Monitoring tells you
what it actually did *on real questions, in production*. Both are needed; they answer different things.

Every answered question — from the web app **and** the CLI — is recorded to an append-only event log
at `data/telemetry/events.jsonl`. The **Metrics** tab in the app reads that log and shows:

- **Latency split into retrieval vs generation.** These are reported separately on purpose. Retrieval
  is local CPU work at ~30ms; generation is a network call to Gemini at ~6-12s. Roughly 90% of the
  wait is the LLM, so optimising FAISS would be wasted effort — that conclusion is only visible
  because the two stages are timed separately.
- **Retrieval confidence**, as a histogram of top-1 cosine similarity per question.
- **Hallucination risk** — the metric that matters most here. It flags answers where retrieval found
  nothing relevant (top score below `LOW_CONFIDENCE_THRESHOLD`) *but the model answered confidently
  anyway* instead of saying it didn't know. Those answers are built on irrelevant context and are
  exactly the ones a human should read. The inverse, **over-refusal** (good retrieval, model declined
  anyway), is tracked too — that one points at an over-strict prompt.
- **Corpus coverage** — which papers real questions actually reach. A paper that is never retrieved is
  either off-topic for what people ask or extracted/chunked badly.
- **User ratings** — 👍/👎 on each answer, written as separate feedback events.
- The raw log, as a table and a JSONL download.

**The confidence threshold is measured, not guessed.** `measure_threshold.py` runs a batch of
on-topic questions and a batch of deliberately off-topic ones and compares the score distributions:
```
cd "D:\RAG System\src"
..\.venv\Scripts\python.exe measure_threshold.py
```
On the current 9-paper corpus the two groups separate cleanly (on-topic top-1 never below 0.394,
off-topic never above 0.231), so `config.LOW_CONFIDENCE_THRESHOLD` sits at the midpoint, 0.32.
Re-run this after changing the corpus or the embedding model — the number is corpus-specific.

**And it has a measured blind spot, which is worth knowing before trusting this dashboard.**
That clean separation only holds for questions that are off-topic in *vocabulary*. The
evaluation suite includes 15 questions that are on-topic in wording but genuinely
unanswerable from the corpus ("What is the carbon footprint of training these event
spotting models?"), and **14 of the 15 score above the threshold** — the highest at 0.706,
while real answerable questions drop as low as 0.332. The two groups overlap completely.

Raising the threshold does not fix it: at 0.71 it would flag nearly every genuine question
too. Cosine similarity measures whether a passage is *about* a question, not whether it
*answers* it, and those are different properties. So `hallucination_risk` here reliably
catches "the user asked about something else entirely" and does not catch "the corpus
cannot answer this". The defence that does work is the model's own refusal behaviour,
which `evaluate.py --generate` measures directly against those 15 questions.

Terminal summary without opening the app:
```
cd "D:\RAG System\src"
..\.venv\Scripts\python.exe telemetry.py
```

**Storage caveat, stated plainly:** the log is a local file, so on Streamlit Community Cloud (whose
filesystem is ephemeral) it covers the current app instance only and resets on restart or redeploy.
Locally it persists. Moving to a database means reimplementing two functions in `telemetry.py`
(`_append` and `load_events`) and nothing else. The log is git-ignored — it is per-deployment runtime
data, and committing it would publish whatever visitors typed into the public demo.

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
