# Retrieval Baselines

Three retrievers over the same 41 answerable questions, the same chunks, and the same scoring code. The only thing that differs is how candidates are ranked.

| Arm | What it does |
|---|---|
| **semantic** | The real system. Embeds the question, returns the nearest chunks by cosine similarity. |
| **bm25** | Classical keyword search. Matches words, weighted by rarity and chunk length. No embeddings. |
| **random** | Picks chunks at random. The floor. |

## Results

| Metric | semantic | bm25 | random |
|---|---|---|---|
| Hit Rate@1 | 0.63 [0.48-0.76] | 0.54 [0.39-0.68] | 0.20 [0.10-0.34] |
| Hit Rate@5 | 0.93 [0.81-0.97] | 0.85 [0.72-0.93] | 0.46 [0.32-0.61] |
| MRR | 0.724 [0.613-0.831] | 0.652 [0.528-0.772] | 0.296 [0.177-0.415] |
| Precision@5 | 0.52 [0.41-0.62] | 0.47 [0.38-0.58] | 0.18 [0.11-0.25] |
| Paper coverage | 0.78 [0.67-0.88] | 0.75 [0.63-0.86] | 0.34 [0.22-0.47] |

Semantic retrieval scores **+0.46** over random and **+0.07** over keyword search on Hit Rate@5.

The gap over BM25 is what the embeddings actually buy. The gap over random is what any working retrieval buys, and quoting only that second number would flatter the system.

## By question type

### Hit Rate@5

| Type | n | semantic | bm25 | random |
|---|---|---|---|---|
| simple | 19 | 0.95 [0.75-0.99] | 0.89 [0.69-0.97] | 0.37 [0.19-0.59] |
| paraphrase | 8 | 0.88 [0.53-0.98] | 0.75 [0.41-0.93] | 0.12 [0.02-0.47] |
| comparison | 7 | 1.00 [0.65-1.00] | 0.86 [0.49-0.97] | 0.71 [0.36-0.92] |
| multi_paper | 7 | 0.86 [0.49-0.97] | 0.86 [0.49-0.97] | 0.86 [0.49-0.97] |

**The `paraphrase` row is the one that matters.** Those questions are deliberately worded so the paper's own vocabulary never appears -- "a racket sport played indoors on a small table" instead of "table tennis". Keyword search has nothing to match on. If embeddings are worth having, this is the row where it shows.

### Paper coverage (fraction of ALL correct papers reached)

| Type | n | semantic | bm25 | random |
|---|---|---|---|---|
| simple | 19 | 0.95 [0.84-1.00] | 0.89 [0.74-1.00] | 0.37 [0.16-0.58] |
| paraphrase | 8 | 0.81 [0.56-1.00] | 0.69 [0.38-0.94] | 0.06 [0.00-0.19] |
| comparison | 7 | 0.71 [0.57-0.93] | 0.79 [0.50-1.00] | 0.50 [0.21-0.79] |
| multi_paper | 7 | 0.37 [0.19-0.55] | 0.42 [0.26-0.58] | 0.43 [0.24-0.60] |

**Read those two tables together, because the `multi_paper` row is the worst result in this report.** Random retrieval scores 0.86 on Hit Rate@5 against the real system's 0.86, and 0.43 on paper coverage against 0.37. The intervals overlap almost completely, so the honest reading is not "random wins" -- it is that **on questions needing several papers at once, semantic retrieval is statistically indistinguishable from picking chunks at random.**

Two separate things are going on, and they should not be conflated:

1. **Hit Rate@5 is the wrong metric here.** These questions accept three or four papers out of nine, so five random chunks are almost guaranteed to touch one, and Hit Rate scores that as a full success. It should not be quoted for multi-paper questions at all.
2. **Top-k retrieval concentrates, and breadth questions need the opposite.** The five nearest chunks to a question tend to come from whichever single paper matches best, so semantic search reaches only 0.37 of the papers a complete answer needs. Random spreads across the corpus by construction, which is why it is not obviously worse here. Similarity ranking has no term for diversity, and that is a fixable property of the retriever (cap chunks per paper, or re-rank for spread) rather than a limit of embeddings.

## Where they disagree

Questions one arm found and the other missed. More useful than the averages: it names the specific failure.

**Semantic found, BM25 missed (6):**
- `S08` *(simple)* What is the Table Tennis Australia dataset and how large is it?
- `S15` *(simple)* What is the difference between Temporal Action Localization, Action Spotting, and Precise Event Spotting?
- `P03` *(paraphrase)* What happens when the people labelling the videos put the timestamp slightly in the wrong place?
- `P08` *(paraphrase)* Is there a benchmark for a racket sport played indoors on a small table?
- `C07` *(comparison)* How does the survey's definition of Action Spotting compare with how the SoccerNet paper uses the term?
- `M05` *(multi_paper)* Which sports are covered across this collection of papers?

**BM25 found, semantic missed (3):**
- `S14` *(simple)* What is the difference between tight and loose average-mAP in SoccerNet?
- `P05` *(paraphrase)* What can you do if you do not have enough labelled training footage?
- `M07` *(multi_paper)* Across these papers, what are the main open challenges in precise event spotting?
