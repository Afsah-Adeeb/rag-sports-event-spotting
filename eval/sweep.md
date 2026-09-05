# Settings Sweep: chunk size and top-k

Every combination of chunk size [500, 1000, 1500] and top-k [3, 5, 10], scored on the same labelled questions. The corpus is re-chunked and re-embedded from the PDFs for each chunk size.

**Currently committed:** `CHUNK_SIZE = 1000`, `DEFAULT_TOP_K = 5`.

Free to run -- retrieval scoring makes no API calls. Overlap is held at a constant fraction of chunk size (20%), so this measures chunk size rather than chunk size and redundancy together. Nothing under `data/vector_store/` is modified: every candidate index is built in memory and discarded.

## Corpus shape at each chunk size

| Chunk size | Overlap | Chunks | Mean chars | Build time |
|---|---|---|---|---|
| 500 | 100 | 1581 | 566 | 93.0s |
| 1000 | 200 | 827 | 1073 | 43.7s |
| 1500 | 300 | 580 | 1522 | 29.3s |

## Hit Rate@k

| chunk size \ top-k | 3 | 5 | 10 |
|---|---|---|---|
| **500** | 0.85 [0.71-0.93] | 0.88 [0.74-0.95] | 0.97 [0.87-1.00] |
| **1000** | 0.78 [0.62-0.88] | 0.93 [0.80-0.97] | 1.00 [0.91-1.00] |
| **1500** | 0.88 [0.74-0.95] | 0.88 [0.74-0.95] | 0.97 [0.87-1.00] |

## MRR

| chunk size \ top-k | 3 | 5 | 10 |
|---|---|---|---|
| **500** | 0.696 [0.567-0.812] | 0.702 [0.573-0.817] | 0.715 [0.594-0.821] |
| **1000** | 0.683 [0.546-0.812] | 0.717 [0.597-0.834] | 0.730 [0.618-0.839] |
| **1500** | 0.708 [0.592-0.821] | 0.708 [0.592-0.821] | 0.724 [0.618-0.828] |

## Paper coverage

| chunk size \ top-k | 3 | 5 | 10 |
|---|---|---|---|
| **500** | 0.73 [0.60-0.85] | 0.75 [0.64-0.87] | 0.85 [0.76-0.93] |
| **1000** | 0.65 [0.52-0.78] | 0.78 [0.67-0.88] | 0.90 [0.83-0.96] |
| **1500** | 0.74 [0.62-0.86] | 0.75 [0.63-0.86] | 0.88 [0.80-0.95] |

## Precision@k

| chunk size \ top-k | 3 | 5 | 10 |
|---|---|---|---|
| **500** | 0.56 [0.45-0.67] | 0.53 [0.42-0.64] | 0.50 [0.41-0.59] |
| **1000** | 0.51 [0.39-0.62] | 0.50 [0.40-0.61] | 0.49 [0.40-0.58] |
| **1500** | 0.56 [0.46-0.66] | 0.49 [0.40-0.59] | 0.47 [0.38-0.55] |

## What the sweep says

- Best Hit Rate: **chunk 1000, top-k 10** at 1.00 [0.91-1.00] -- but note this will almost always pick the largest k in the grid, because Hit Rate cannot fall as k rises. It is not evidence that k=10 is the right choice.
- Best MRR: **chunk 1000, top-k 10** at 0.730 [0.618-0.839]
- Currently committed (1000/5): Hit Rate 0.93 [0.80-0.97], MRR 0.717 [0.597-0.834]

The best cell beats the committed one by +0.07 on Hit Rate, but the confidence intervals overlap, so this grid **cannot establish** that the difference is real. The honest conclusion is that retrieval quality is not very sensitive to these two knobs on this corpus -- which is itself the answer to "why 1000?": within measurement error, it does not matter much.

**Reading the top-k columns.** Hit Rate can only rise with larger k -- more slots cannot lose a paper that a smaller k already found -- so a higher Hit Rate at k=10 is not evidence that k=10 is better. MRR and precision are the columns that can genuinely fall, and every extra chunk is paid for in prompt tokens on every single query. Pick the smallest k whose MRR has stopped improving.

**This table is corpus-specific.** Re-run it after adding papers; the best chunk size for nine papers is not necessarily the best for fifty.
