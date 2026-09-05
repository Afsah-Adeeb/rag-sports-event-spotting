# Robustness: retrieval under badly-typed questions

Every answerable question re-asked in four forms and re-scored at top-k=5. All variants are generated deterministically from the original, so nothing here was hand-picked.

| Variant | What it does to the question |
|---|---|
| `original` | unchanged -- careful academic English |
| `typo` | one character-level typo per ~12 characters, inside words only |
| `casual` | lowercased, punctuation removed, polite question frame stripped |
| `keywords` | stopwords removed -- a bag of content words, search-box style |

## Results

| Variant | Hit Rate@5 | MRR | Same top paper as original | Mean confidence |
|---|---|---|---|---|
| `original` | 0.93 [0.81-0.97] | 0.724 [0.613-0.831] | - | 0.536 [0.501-0.573] |
| `typo` | 0.76 [0.61-0.86] | 0.561 [0.435-0.693] | 0.46 [0.32-0.61] | 0.407 [0.378-0.439] |
| `casual` | 0.80 [0.66-0.90] | 0.678 [0.551-0.805] | 0.85 [0.72-0.93] | 0.541 [0.498-0.587] |
| `keywords` | 0.90 [0.77-0.96] | 0.712 [0.592-0.826] | 0.80 [0.66-0.90] | 0.538 [0.496-0.582] |

**Accuracy and stability are different questions.** Hit Rate can hold steady while the *same top paper* column falls, because several papers may be acceptable for one question. That combination means the system is landing on a different-but-still-correct source -- fine today, fragile once the corpus grows and near-duplicate papers start competing.

- **`typo`**: Hit Rate +0.17 vs original (within measurement error); kept the same top paper on 46% of questions.
- **`casual`**: Hit Rate +0.12 vs original (within measurement error); kept the same top paper on 85% of questions.
- **`keywords`**: Hit Rate +0.02 vs original (within measurement error); kept the same top paper on 80% of questions.

## Per-type

| Type | `original` | `typo` | `casual` | `keywords` |
|---|---|---|---|---|
| comparison (7) | 1.00 | 1.00 | 0.86 | 1.00 |
| multi_paper (7) | 0.86 | 0.86 | 0.71 | 1.00 |
| paraphrase (8) | 0.88 | 0.50 | 0.88 | 0.75 |
| simple (19) | 0.95 | 0.74 | 0.79 | 0.89 |

## Questions that broke

Questions the original form got right and a degraded form got wrong. These name the actual failure rather than averaging it away.

**`S02`** (simple) -- broke under: `typo`
- original: Which datasets did the E2E-Spot authors add frame-accurate annotations to?
- typo: Which datasets did the E2E-pSot authors ad farme-accurrate annotaoins to?

**`S04`** (simple) -- broke under: `typo`, `casual`
- original: On which datasets is T-DEED evaluated, and by how much does it improve over E2E-Spot?
- typo: On hwich daatsets is T-DEED evaluaated, and by ohw much does it imprrove ovr E2E-Sopt?
- casual: on which datasets is tdeed evaluated and by how much does it improve over e2espot

**`S08`** (simple) -- broke under: `typo`
- original: What is the Table Tennis Australia dataset and how large is it?
- typo: What is the Table Tennnis Austraalia dasteat and how large is it?

**`S10`** (simple) -- broke under: `casual`, `keywords`
- original: What is Transformer Gate Shift?
- casual: transformer gate shift
- keywords: Transformer Gate Shift

**`S15`** (simple) -- broke under: `typo`, `casual`, `keywords`
- original: What is the difference between Temporal Action Localization, Action Spotting, and Precise Event Spotting?
- typo: What is the differeecnne beetween Temporal Action oLcaliation, Action Spottnig, and Preccise Event Spotting?
- casual: the difference between temporal action localization action spotting and precise event spotting
- keywords: difference Temporal Action Localization Action Spotting Precise Event Spotting

**`P02`** (paraphrase) -- broke under: `typo`
- original: How do these systems handle the fact that some events happen far more often than others?
- typo: How do these ysstems hande the fact htt some evnts hapen far more often than oters?

**`P04`** (paraphrase) -- broke under: `typo`
- original: Is there any work here that uses sound as well as pictures?
- typo: Is there any wrok hree that uses sond as well as pictrues?

**`P06`** (paraphrase) -- broke under: `keywords`
- original: How do researchers score whether a spotting system got the moment right?
- keywords: researchers score whether spotting system got moment right

**`P08`** (paraphrase) -- broke under: `typo`
- original: Is there a benchmark for a racket sport played indoors on a small table?
- typo: Is there a beenchmrak foor a racket sport plyed indoors on a small atbble?

**`C03`** (comparison) -- broke under: `casual`
- original: Compare how T-DEED and the Sony paper each try to make frame-level features more distinguishable.
- casual: how tdeed and the sony paper each try to make framelevel features more distinguishable

**`M03`** (multi_paper) -- broke under: `casual`
- original: Which papers build on or compare against E2E-Spot?
- casual: papers build on or compare against e2espot

**`S19`** (simple) -- broke under: `typo`
- original: How does T-DEED perform on the SoccerNet Ball Action Spotting benchmark?
- typo: How does T-DEED eprfrom on the SocceNet Ball Acttion Spottng benchmark?
