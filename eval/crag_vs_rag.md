# Plain RAG vs Corrective RAG

Both arms, same 41 answerable + 14 unanswerable questions, same model (`gemini-3.1-flash-lite`), same scoring code.

**Both are scored on the context the generator actually received**, not on everything retrieval returned. Plain RAG sends the five chunks it retrieved; CRAG grades those five, may look at fifteen, and sends its best five. Scoring CRAG on all fifteen would raise its hit rate for free -- more chunks cannot lose a paper a smaller set already found -- and would measure search depth instead of judgement. Given the same budget of five passages, which arm fills them better?

## The headline: catching out-of-scope questions

`config.LOW_CONFIDENCE_THRESHOLD` was measured and documented as unable to do this job -- 14 of 15 unanswerable questions score above it, because cosine similarity measures whether a passage is *about* a question, not whether it *answers* it. CRAG exists to replace that judgement with one made by a model that reads the passage.

| | Plain RAG | CRAG |
|---|---|---|
| Correctly refused (14 unanswerable) | 0.86 [0.60-0.96] | 0.93 [0.69-0.99] |
| Wrongly refused (answerable) | 0.15 [0.07-0.28] | 0.07 [0.03-0.19] |

Both directions, because a system that refuses everything scores perfectly on the first row and terribly on the second.

- **Plain RAG** answered anyway: `U12`, `U13`
- **Plain RAG** refused a question it could have answered: `S02`, `S10`, `C01`, `C02`, `C07`, `M03`
- **CRAG** answered anyway: `U12`
- **CRAG** refused a question it could have answered: `C01`, `C02`, `M03`

## Retrieval, on what the model was given

| Metric | Plain RAG | CRAG |
|---|---|---|
| Hit Rate@5 | 0.93 [0.81-0.97] | 0.95 [0.84-0.99] |
| MRR | 0.724 [0.613-0.831] | 0.744 [0.646-0.837] |
| Paper coverage | 0.78 [0.67-0.88] | 0.83 [0.74-0.92] |
| Chunks sent to the model | 5.000 [5.000-5.000] | 2.964 [2.436-3.455] |

Stage 1 CRAG can only drop chunks or look further down the same ranked list -- it cannot invent a better query. Large movement here was not expected and its absence is not a failure; it is the boundary between Stage 1 and query rewriting.

## Answers

| Metric | Plain RAG | CRAG |
|---|---|---|
| Required facts present | 0.72 [0.63-0.80] | 0.80 [0.73-0.88] |

## Is the grader any good?

Judged against labels already in the test set, over 655 graded chunks.

| Check | Result |
|---|---|
| Chunks for *unanswerable* questions marked irrelevant | 0.97 [0.93-0.98] |
| Chunks from a *wrong* paper marked irrelevant | 0.63 [0.57-0.69] |
| Chunks from a *correct* paper kept | 0.49 [0.43-0.56] |

The first two rows are the meaningful ones. A chunk from a paper that cannot answer the question almost certainly does not contain the answer, so the grader should reject it. The third row is looser: a chunk from the right paper might be that paper's reference list, and marking it irrelevant is correct. A low number there is not automatically an error.

## What the correction cost

| | Plain RAG | CRAG |
|---|---|---|
| API calls per question | 1.000 [1.000-1.000] | 2.509 [2.382-2.636] |
| Input tokens per question | 1473 | 4503 |
| Latency (s) | 2.885 [2.520-3.291] | 7.605 [6.921-8.294] |

CRAG costs **3.1x** the input tokens. It deepened on 38 question(s) and refused on 10, and a refusal skips generation entirely -- so the extra grading call is partly paid back on exactly the questions plain RAG would have spent a generation call getting wrong.

## Per question

| id | type | plain | CRAG | CRAG decision |
|---|---|---|---|---|
| `S01` | simple | 0.00 | 0.67 | answer (deepened) |
| `S02` | simple | 0.00 | 1.00 | answer (deepened) |
| `S03` | simple | 1.00 | 1.00 | answer |
| `S04` | simple | 0.67 | 0.67 | answer (deepened) |
| `S05` | simple | 0.67 | 1.00 | answer (deepened) |
| `S06` | simple | 1.00 | 1.00 | answer |
| `S07` | simple | 0.50 | 1.00 | answer (deepened) |
| `S08` | simple | 0.50 | 0.50 | answer (deepened) |
| `S09` | simple | 1.00 | 1.00 | answer |
| `S10` | simple | 0.33 | 0.33 | answer |
| `S11` | simple | 1.00 | 1.00 | answer |
| `S12` | simple | 0.67 | 0.67 | answer |
| `S13` | simple | 0.33 | 0.33 | answer (deepened) |
| `S14` | simple | 1.00 | 1.00 | answer (deepened) |
| `S15` | simple | 0.67 | 0.67 | answer |
| `S16` | simple | 0.50 | 1.00 | answer (deepened) |
| `S17` | simple | 0.50 | 0.50 | answer (deepened) |
| `S18` | simple | 1.00 | 1.00 | answer |
| `P01` | paraphrase | 0.50 | 0.50 | answer |
| `P02` | paraphrase | 0.50 | 0.50 | answer (deepened) |
| `P03` | paraphrase | 1.00 | 1.00 | answer |
| `P04` | paraphrase | 1.00 | 1.00 | answer |
| `P05` | paraphrase | 0.50 | 1.00 | answer (deepened) |
| `P06` | paraphrase | 1.00 | 0.50 | answer (deepened) |
| `P07` | paraphrase | 0.67 | 0.67 | answer (deepened) |
| `P08` | paraphrase | 1.00 | 1.00 | answer |
| `C01` | comparison | 0.67 | 1.00 | answer (deepened) |
| `C02` | comparison | 0.67 | 0.33 | answer (deepened) |
| `C03` | comparison | 0.50 | 0.50 | answer (deepened) |
| `C04` | comparison | 1.00 | 1.00 | answer (deepened) |
| `C05` | comparison | 1.00 | 1.00 | answer (deepened) |
| `C06` | comparison | 1.00 | 1.00 | answer |
| `C07` | comparison | 1.00 | 1.00 | answer |
| `M01` | multi_paper | 0.67 | 1.00 | answer (deepened) |
| `M02` | multi_paper | 1.00 | 1.00 | answer |
| `M03` | multi_paper | 0.67 | 0.67 | answer |
| `M04` | multi_paper | 1.00 | 1.00 | answer |
| `M05` | multi_paper | 0.67 | 1.00 | answer (deepened) |
| `M06` | multi_paper | 0.67 | 0.67 | answer (deepened) |
| `M07` | multi_paper | 0.33 | 0.33 | answer (deepened) |
| `U01` | unanswerable | - | - | answer (deepened) |
| `S19` | simple | 1.00 | 1.00 | answer (deepened) |
| `U03` | unanswerable | - | - | refuse (deepened) |
| `U04` | unanswerable | - | - | answer (deepened) |
| `U05` | unanswerable | - | - | refuse (deepened) |
| `U06` | unanswerable | - | - | refuse (deepened) |
| `U07` | unanswerable | - | - | answer (deepened) |
| `U08` | unanswerable | - | - | refuse (deepened) |
| `U09` | unanswerable | - | - | refuse (deepened) |
| `U10` | unanswerable | - | - | refuse (deepened) |
| `U11` | unanswerable | - | - | refuse (deepened) |
| `U12` | unanswerable | - | - | answer (deepened) |
| `U13` | unanswerable | - | - | refuse (deepened) |
| `U14` | unanswerable | - | - | refuse (deepened) |
| `U15` | unanswerable | - | - | refuse (deepened) |
