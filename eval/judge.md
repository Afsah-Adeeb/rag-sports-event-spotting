# LLM Judge: a cross-check, not a verdict

Every answer graded twice by `gemini-3.1-flash-lite` for whether it is supported by the context it was given. Run against the exact answers the deterministic fact-coverage metric scored, from a shared cache -- so any difference between the two is a difference in *judgement*, not in which answers were sampled.

**This is deliberately not the headline metric.** Published 2026 work finds automated RAG grader suites correlate with human judgement at around 0.55; that raw judge-human agreement overstates chance-corrected agreement by 34-41 points; and that position bias varies roughly 100-fold between models, with Flash-class models measuring far worse than Pro-class ones. This project runs a Flash-class model for cost reasons, which puts it on the wrong side of that split. So the judge is used to find disagreements worth a human's attention, not to hand out scores.

## First: can the judge agree with itself?

Each answer was judged twice in independent calls. The two passes agreed on **0.95 [0.85-0.98]** of answers.

High self-consistency. Note what this does and does not establish: it means the judge is *repeatable*, not that it is *right*. Judges with test-retest reliability above 0.95 have been measured carrying severe systematic bias at the same time. Repeatability is a precondition for trust, not evidence of it.

## Verdicts

| Verdict | Count |
|---|---|
| SUPPORTED | 47 |
| PARTIAL | 2 |
| UNSUPPORTED | 1 |
| REFUSED | 5 |

## Judge vs deterministic fact checking

On the 41 answerable questions, the two methods reached the same conclusion **0.51 [0.36-0.66]** of the time.

They are asking different questions, so perfect agreement was never the goal: fact coverage asks *did the answer contain the required facts*, the judge asks *is everything the answer said supported by the context*. An answer can pass one and fail the other for good reasons. The disagreements are the output.

### Judge passed it, required facts missing (20)

The answer is well-grounded in what it was given, but does not contain everything a complete answer needs. Usually a **retrieval** failure rather than a generation one: the model faithfully reported chunks that did not carry the missing fact.

| Question | Facts | Missing |
|---|---|---|
| `S01` What are the main components of the E2E-Spot architecture? | 0.00 | RegNet, GSM/Gate Shift, GRU/gated recurrent |
| `S02` Which datasets did the E2E-Spot authors add frame-accurate a | 0.00 | Tennis, Figure Skating/FigureSkating |
| `S05` What is the SoftIC loss and what problem does it address? | 0.67 | class imbalance/imbalance |
| `S07` What are the two limitations of the Gate Shift Module that M | 0.50 | receptive field/temporal range/adjacent |
| `S08` What is the Table Tennis Australia dataset and how large is  | 0.50 | 4,800/4800 |
| `S10` What is Transformer Gate Shift? | 0.33 | multi-scale/multi scale, Vision Transformer/ViT |
| `S12` How does the dynamic label assignment method work? | 0.67 | object detection |
| `S15` What is the difference between Temporal Action Localization, | 0.67 | frame-level/exact frame/frame level |
| `S16` What criticisms does the survey make of existing benchmark d | 0.50 | permissive/multi-label |
| `S17` How is audio processed in the multimodal soccer event detect | 0.50 | ResNet |
| `P01` Which model here is small enough to train on a single graphi | 0.50 | single GPU/one GPU |
| `P02` How do these systems handle the fact that some events happen | 0.50 | SoftIC/contrastive |
| `P05` What can you do if you do not have enough labelled training  | 0.00 | semi-supervised/label-efficient/distillation, 10%/unlabeled/unlabelled |
| `P07` Which method borrows an idea from object detection and appli | 0.67 | matching |
| `C01` How does T-DEED's approach to temporal resolution differ fro | 0.67 | encoder-decoder |
| `M01` Which papers in this collection introduce a new dataset or b | 0.67 | SoccerNet |
| `M03` Which papers build on or compare against E2E-Spot? | 0.67 | T-DEED |
| `M05` Which sports are covered across this collection of papers? | 0.67 | figure skating/figureskating/diving/finediving |
| `M06` What is the Gate Shift Module, and which papers here use or  | 0.67 | E2E-Spot |
| `M07` Across these papers, what are the main open challenges in pr | 0.67 | imbalance |

### Facts all present, judge flagged it (0)

None.

## Does the judge agree with the refusal heuristic?

`telemetry.looks_like_refusal()` decides by string matching in the first ~150 characters, and the Metrics dashboard's hallucination flag is built on it. The judge agrees with it on **0.75 [0.62-0.84]** of answers.

| Question | Heuristic says refused | Judge says | Answerable |
|---|---|---|---|
| `S02` | True | SUPPORTED | True |
| `C01` | True | SUPPORTED | True |
| `C07` | True | SUPPORTED | True |
| `M03` | True | SUPPORTED | True |
| `U01` | True | SUPPORTED | False |
| `U04` | True | SUPPORTED | False |
| `U05` | True | SUPPORTED | False |
| `U08` | True | SUPPORTED | False |
| `U09` | True | SUPPORTED | False |
| `U10` | True | SUPPORTED | False |
| `U11` | True | SUPPORTED | False |
| `U12` | True | SUPPORTED | False |
| `U13` | True | SUPPORTED | False |
| `U14` | True | SUPPORTED | False |

Each row is worth reading by hand. A cheap heuristic disagreeing with an unreliable judge does not tell you which one is wrong -- it tells you where to look.

## The unanswerable questions, as the judge sees them

| Verdict | Count |
|---|---|
| SUPPORTED | 10 |
| PARTIAL | 0 |
| UNSUPPORTED | 0 |
| REFUSED | 4 |

`REFUSED` is the correct outcome for all of these. Anything scored `SUPPORTED` is the judge claiming the context backs an answer to a question the corpus cannot answer -- which would be the judge failing, the system failing, or both.
