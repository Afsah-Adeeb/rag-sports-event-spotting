# Evaluation Report

Test set: 41 answerable + 14 unanswerable questions. Retrieval at top-k=5.

All figures carry a 95% confidence interval in brackets. Proportions use a Wilson interval, means use a seeded bootstrap -- see `eval_core.py` for why the two differ.

## Retrieval

| Metric | Score [95% CI] |
|---|---|
| Hit Rate@1 | 0.63 [0.48-0.76] |
| Hit Rate@3 | 0.78 [0.63-0.88] |
| Hit Rate@5 | 0.93 [0.81-0.97] |
| Precision@1 | 0.63 [0.49-0.78] |
| Precision@3 | 0.52 [0.41-0.63] |
| Precision@5 | 0.52 [0.41-0.62] |
| MRR | 0.724 [0.613-0.831] |
| Paper coverage | 0.78 [0.67-0.88] |

- **Hit Rate@k** -- did at least one of the top-k chunks come from a correct paper? *Did we find it at all.*
- **Precision@k** -- what fraction of the top-k came from a correct paper? *How much noise came with it.*
- **MRR** -- 1/(position of the first correct chunk). *Where we found it.* Hit Rate scores position 1 and position 5 identically; this does not.
- **Paper coverage** -- fraction of ALL correct papers reached, which matters for the multi-paper questions that need three or four at once.

### Where the first correct chunk lands

| Rank | Questions |
|---|---|
| 1 | 26 |
| 2 | 2 |
| 3 | 4 |
| 4 | 3 |
| 5 | 3 |
| not in top-5 | 3 |

If most of the mass sits at rank 4-5, the system depends on every slot and lowering `DEFAULT_TOP_K` would break it. If it sits at rank 1, a smaller k is free.

## Confidence control: answerable vs unanswerable

The 15 unanswerable questions have no correct paper to find, so they are not scored for accuracy. They serve as a control: retrieval confidence on questions the corpus *can* answer should sit clearly above confidence on questions it cannot.

| Group | Mean top-1 similarity [95% CI] |
|---|---|
| Answerable | 0.536 [0.501-0.573] |
| Unanswerable | 0.476 [0.425-0.527] |

The two groups **overlap** by 0.326. No single similarity threshold can separate 'found it' from 'found nothing' on this corpus, which means the hallucination flag in the monitoring dashboard cannot be fully trusted. Worth saying out loud rather than tuning the threshold until it looks clean.

## By question type

| Type | n | Hit@5 | MRR | Paper coverage | Mean confidence |
|---|---|---|---|---|---|
| simple | 19 | 0.95 [0.75-0.99] | 0.831 [0.651-0.961] | 0.95 [0.84-1.00] | 0.578 [0.520-0.635] |
| paraphrase | 8 | 0.88 [0.53-0.98] | 0.635 [0.375-0.875] | 0.81 [0.56-1.00] | 0.491 [0.448-0.539] |
| comparison | 7 | 1.00 [0.65-1.00] | 0.790 [0.561-1.000] | 0.71 [0.57-0.93] | 0.503 [0.423-0.603] |
| multi_paper | 7 | 0.86 [0.49-0.97] | 0.469 [0.214-0.726] | 0.37 [0.19-0.55] | 0.508 [0.459-0.555] |
| unanswerable | 14 | - | - | - | 0.476 [0.425-0.527] |

This table is where the findings are. A single average across all questions can be 100% on simple lookups and 0% on multi-paper ones and still print as a healthy number.

## Per-question detail

### S01 (simple) -- What are the main components of the E2E-Spot architecture?
*Why this question is here: Core architecture fact. Baseline sanity check.*

**Correct paper(s):** Spotting Temporally Precise, Fine-Grained Events in Video.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.14-14, score=0.520)
2. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.485)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.14-14, score=0.445)
4. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.3-3, score=0.432)
5. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.6-6, score=0.426)

---

### S02 (simple) -- Which datasets did the E2E-Spot authors add frame-accurate annotations to?
*Why this question is here: Dataset contribution, stated in the abstract and again in section 4.*

**Correct paper(s):** Spotting Temporally Precise, Fine-Grained Events in Video.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.24-24, score=0.567)
2. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.6-6, score=0.546)
3. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.3-3, score=0.522)
4. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.14-14, score=0.507)
5. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.500)

---

### S03 (simple) -- What is the SGP-Mixer layer in T-DEED and what problem does it solve?
*Why this question is here: Tests whether retrieval reaches the contributions list, not just the abstract.*

**Correct paper(s):** T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.4-4, score=0.560)
2. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.5-5, score=0.501)
3. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.1-1, score=0.489)
4. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.4-4, score=0.485)
5. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.11-11, score=0.471)

---

### S04 (simple) -- On which datasets is T-DEED evaluated, and by how much does it improve over E2E-Spot?
*Why this question is here: Requires a specific number, not just a topic match. Catches vague hand-waving answers.*

**Correct paper(s):** T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.6-6, score=0.538)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.3-3, score=0.494)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.4-4, score=0.474)
4. [ ] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.6-6, score=0.463)
5. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.4-4, score=0.459)

---

### S05 (simple) -- What is the SoftIC loss and what problem does it address?
*Why this question is here: Named loss function with a stated motivation.*

**Correct paper(s):** Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.4-4, score=0.425)
2. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.4-4, score=0.345)
3. [x] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.5-5, score=0.343)
4. [ ] Deep learning for action spotting in association football videos.pdf (p.26-26, score=0.330)
5. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.5-5, score=0.313)

---

### S06 (simple) -- What does the ASTRM module do?
*Why this question is here: Acronym-only question: tests whether a bare acronym retrieves the right paper.*

**Correct paper(s):** Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.7-7, score=0.433)
2. [ ] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.4-4, score=0.331)
3. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.4-4, score=0.307)
4. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.13-13, score=0.299)
5. [x] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.2-2, score=0.295)

---

### S07 (simple) -- What are the two limitations of the Gate Shift Module that MFS is designed to fix?
*Why this question is here: Two-part answer, so a partial answer should score partially rather than pass or fail.*

**Correct paper(s):** Multi-Focus Temporal Shifting for Precise Event.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.439)
2. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.413)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.29-29, score=0.386)
4. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.3-3, score=0.383)
5. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.376)

---

### S08 (simple) -- What is the Table Tennis Australia dataset and how large is it?
*Why this question is here: Requires a number that appears exactly once in the whole corpus.*

**Correct paper(s):** Multi-Focus Temporal Shifting for Precise Event.pdf
**First correct chunk at rank:** 3

**Retrieved:**
1. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.17-17, score=0.575)
2. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.17-17, score=0.517)
3. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.4-4, score=0.481)
4. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.6-6, score=0.475)
5. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.25-25, score=0.462)

---

### S09 (simple) -- Why does applying DINO directly to precise event spotting not work?
*Why this question is here: A negative result. Exactly the kind of finding a careless model inverts into a positive one.*

**Correct paper(s):** Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.2-2, score=0.541)
2. [x] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.5-5, score=0.507)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.4-4, score=0.503)
4. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.5-5, score=0.485)
5. [x] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.1-1, score=0.484)

---

### S10 (simple) -- What is Transformer Gate Shift?
*Why this question is here: Name collides with 'Gate Shift Module' in three other papers. Tests disambiguation.*

**Correct paper(s):** Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf
**First correct chunk at rank:** 4

**Retrieved:**
1. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.3-3, score=0.427)
2. [ ] Deep learning for action spotting in association football videos.pdf (p.29-29, score=0.401)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.29-29, score=0.379)
4. [x] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.4-4, score=0.340)
5. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.14-14, score=0.335)

---

### S11 (simple) -- What is temporal misalignment in action spotting labels, and what causes it?
*Why this question is here: Definitional question that also demands the stated cause.*

**Correct paper(s):** Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.2-2, score=0.718)
2. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.2-2, score=0.681)
3. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.1-1, score=0.676)
4. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.2-2, score=0.661)
5. [ ] Deep learning for action spotting in association football videos.pdf (p.11-11, score=0.646)

---

### S12 (simple) -- How does the dynamic label assignment method work?
*Why this question is here: Mechanism question. A good answer names the idea it borrowed and from where.*

**Correct paper(s):** Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.3-3, score=0.617)
2. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.6-6, score=0.561)
3. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.6-6, score=0.492)
4. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.2-2, score=0.486)
5. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.5-5, score=0.477)

---

### S13 (simple) -- How many games does SoccerNet Action Spotting contain, and how many annotated actions are in SoccerNet-v2?
*Why this question is here: Three numbers from two different sections, so one chunk is not enough.*

**Correct paper(s):** Deep learning for action spotting in association football videos.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Deep learning for action spotting in association football videos.pdf (p.6-6, score=0.851)
2. [x] Deep learning for action spotting in association football videos.pdf (p.4-4, score=0.797)
3. [x] Deep learning for action spotting in association football videos.pdf (p.6-6, score=0.795)
4. [x] Deep learning for action spotting in association football videos.pdf (p.5-5, score=0.794)
5. [x] Deep learning for action spotting in association football videos.pdf (p.23-23, score=0.768)

---

### S14 (simple) -- What is the difference between tight and loose average-mAP in SoccerNet?
*Why this question is here: Evaluation-protocol detail buried deep in a metrics section.*

**Correct paper(s):** Deep learning for action spotting in association football videos.pdf
**First correct chunk at rank:** not in top-5

**Retrieved:**
1. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.20-20, score=0.473)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.13-13, score=0.464)
3. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.20-20, score=0.448)
4. [ ] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.6-6, score=0.433)
5. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.19-19, score=0.426)

---

### S15 (simple) -- What is the difference between Temporal Action Localization, Action Spotting, and Precise Event Spotting?
*Why this question is here: The survey's headline contribution. Three-way distinction, easy to blur.*

**Correct paper(s):** Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf
**First correct chunk at rank:** 5

**Retrieved:**
1. [ ] Deep learning for action spotting in association football videos.pdf (p.5-5, score=0.708)
2. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.2-2, score=0.682)
3. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.1-1, score=0.678)
4. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.5-5, score=0.674)
5. [x] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.4-4, score=0.652)

---

### S16 (simple) -- What criticisms does the survey make of existing benchmark datasets and evaluation protocols?
*Why this question is here: A critique rather than a fact, which is harder to keep grounded.*

**Correct paper(s):** Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.14-14, score=0.424)
2. [ ] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.6-6, score=0.412)
3. [ ] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.6-6, score=0.412)
4. [x] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.23-23, score=0.399)
5. [ ] Deep learning for action spotting in association football videos.pdf (p.20-20, score=0.376)

---

### S17 (simple) -- How is audio processed in the multimodal soccer event detection paper?
*Why this question is here: The only paper in the corpus that deals with audio at all.*

**Correct paper(s):** Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.24-24, score=0.748)
2. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.3-3, score=0.748)
3. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.5-5, score=0.698)
4. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.2-2, score=0.688)
5. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.25-25, score=0.684)

---

### S18 (simple) -- Does adding audio always improve soccer event detection?
*Why this question is here: Nuanced yes-and-no. A model that skims the abstract answers a flat 'yes'.*

**Correct paper(s):** Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.2-2, score=0.705)
2. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.24-24, score=0.693)
3. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.24-24, score=0.663)
4. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.6-6, score=0.662)
5. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.3-3, score=0.652)

---

### P01 (paraphrase) -- Which model here is small enough to train on a single graphics card?
*Why this question is here: Says 'graphics card'; the paper says 'GPU'. Keyword search should struggle here.*

**Correct paper(s):** Spotting Temporally Precise, Fine-Grained Events in Video.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.15-15, score=0.417)
2. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.12-12, score=0.392)
3. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.15-15, score=0.385)
4. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.4-4, score=0.382)
5. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.12-12, score=0.376)

---

### P02 (paraphrase) -- How do these systems handle the fact that some events happen far more often than others?
*Why this question is here: Describes class imbalance without ever using the term. Pure semantic test.*

**Correct paper(s):** Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.2-2, score=0.449)
2. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.6-6, score=0.408)
3. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.11-11, score=0.407)
4. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.14-14, score=0.406)
5. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.4-4, score=0.392)

---

### P03 (paraphrase) -- What happens when the people labelling the videos put the timestamp slightly in the wrong place?
*Why this question is here: Everyday phrasing of 'temporal misalignment in ground-truth labels'.*

**Correct paper(s):** Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf
**First correct chunk at rank:** 3

**Retrieved:**
1. [ ] Deep learning for action spotting in association football videos.pdf (p.11-11, score=0.460)
2. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.5-5, score=0.443)
3. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.1-1, score=0.439)
4. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.2-2, score=0.407)
5. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.1-1, score=0.399)

---

### P04 (paraphrase) -- Is there any work here that uses sound as well as pictures?
*Why this question is here: 'Sound' and 'pictures' never appear in the corpus; 'audio' and 'visual' do.*

**Correct paper(s):** Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.20-20, score=0.423)
2. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.14-14, score=0.395)
3. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.3-3, score=0.394)
4. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.6-6, score=0.382)
5. [x] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.2-2, score=0.375)

---

### P05 (paraphrase) -- What can you do if you do not have enough labelled training footage?
*Why this question is here: Casual phrasing of the label-efficiency problem.*

**Correct paper(s):** Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf
**First correct chunk at rank:** not in top-5

**Retrieved:**
1. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.23-23, score=0.526)
2. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.5-5, score=0.512)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.14-14, score=0.508)
4. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.8-8, score=0.503)
5. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.15-15, score=0.496)

---

### P06 (paraphrase) -- How do researchers score whether a spotting system got the moment right?
*Why this question is here: Plain-language question about the evaluation metric.*

**Correct paper(s):** Deep learning for action spotting in association football videos.pdf, Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Deep learning for action spotting in association football videos.pdf (p.11-11, score=0.602)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.19-19, score=0.583)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.3-3, score=0.570)
4. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.4-4, score=0.568)
5. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.9-9, score=0.549)

---

### P07 (paraphrase) -- Which method borrows an idea from object detection and applies it to time instead of space?
*Why this question is here: Describes the mechanism instead of naming it.*

**Correct paper(s):** Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf
**First correct chunk at rank:** 2

**Retrieved:**
1. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.5-5, score=0.476)
2. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.9-9, score=0.460)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.2-2, score=0.457)
4. [ ] Deep learning for action spotting in association football videos.pdf (p.11-11, score=0.456)
5. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.4-4, score=0.453)

---

### P08 (paraphrase) -- Is there a benchmark for a racket sport played indoors on a small table?
*Why this question is here: Describes table tennis without naming it. Essentially impossible for keyword search.*

**Correct paper(s):** Multi-Focus Temporal Shifting for Precise Event.pdf
**First correct chunk at rank:** 4

**Retrieved:**
1. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.17-17, score=0.576)
2. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.17-17, score=0.561)
3. [ ] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.8-8, score=0.503)
4. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.4-4, score=0.501)
5. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.487)

---

### C01 (comparison) -- How does T-DEED's approach to temporal resolution differ from E2E-Spot's?
*Why this question is here: Needs both papers, or one paper's related-work section.*

**Correct paper(s):** T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf, Spotting Temporally Precise, Fine-Grained Events in Video.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.6-6, score=0.478)
2. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.12-12, score=0.473)
3. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.14-14, score=0.471)
4. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.8-8, score=0.453)
5. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.4-4, score=0.447)

---

### C02 (comparison) -- Both MFS and Transformer Gate Shift extend the Gate Shift Module. How do they differ?
*Why this question is here: Two papers with overlapping authors and near-identical module names. Confusion trap.*

**Correct paper(s):** Multi-Focus Temporal Shifting for Precise Event.pdf, Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.3-3, score=0.414)
2. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.351)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.29-29, score=0.334)
4. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.4-4, score=0.333)
5. [ ] Deep learning for action spotting in association football videos.pdf (p.29-29, score=0.329)

---

### C03 (comparison) -- Compare how T-DEED and the Sony paper each try to make frame-level features more distinguishable.
*Why this question is here: Same goal, different mechanisms, two papers that never cite each other on this point.*

**Correct paper(s):** T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf, Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf
**First correct chunk at rank:** 5

**Retrieved:**
1. [ ] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.5-5, score=0.486)
2. [ ] Deep learning for action spotting in association football videos.pdf (p.16-16, score=0.458)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.16-16, score=0.438)
4. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.5-5, score=0.436)
5. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.2-2, score=0.432)

---

### C04 (comparison) -- How computationally efficient is MFS compared with the heavier architectures it is measured against?
*Why this question is here: Requires reading a specific efficiency comparison rather than the general claim.*

**Correct paper(s):** Multi-Focus Temporal Shifting for Precise Event.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.603)
2. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.557)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.7-7, score=0.548)
4. [ ] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.8-8, score=0.491)
5. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.11-11, score=0.420)

---

### C05 (comparison) -- Do E2E-Spot and T-DEED use the same backbone sizes?
*Why this question is here: Answer lives inside a results table. Tests whether table text survived PDF extraction.*

**Correct paper(s):** T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf, Spotting Temporally Precise, Fine-Grained Events in Video.pdf
**First correct chunk at rank:** 3

**Retrieved:**
1. [ ] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.6-6, score=0.332)
2. [ ] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.5-5, score=0.309)
3. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.306)
4. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.3-3, score=0.304)
5. [ ] Deep learning for action spotting in association football videos.pdf (p.14-14, score=0.299)

---

### C06 (comparison) -- Which papers here tackle problems with the training labels rather than the model architecture?
*Why this question is here: Requires categorising papers, not just recalling one of them.*

**Correct paper(s):** Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf, Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.5-5, score=0.462)
2. [x] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.6-6, score=0.427)
3. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.3-3, score=0.402)
4. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.12-12, score=0.383)
5. [x] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.6-6, score=0.369)

---

### C07 (comparison) -- How does the survey's definition of Action Spotting compare with how the SoccerNet paper uses the term?
*Why this question is here: Two sources, same term, slightly different framing.*

**Correct paper(s):** Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf, Deep learning for action spotting in association football videos.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Deep learning for action spotting in association football videos.pdf (p.4-4, score=0.744)
2. [x] Deep learning for action spotting in association football videos.pdf (p.6-6, score=0.743)
3. [x] Deep learning for action spotting in association football videos.pdf (p.8-8, score=0.734)
4. [x] Deep learning for action spotting in association football videos.pdf (p.4-4, score=0.730)
5. [x] Deep learning for action spotting in association football videos.pdf (p.23-23, score=0.714)

---

### M01 (multi_paper) -- Which papers in this collection introduce a new dataset or benchmark, and what are they?
*Why this question is here: Needs at least three different papers represented in the top-k.*

**Correct paper(s):** Multi-Focus Temporal Shifting for Precise Event.pdf, Spotting Temporally Precise, Fine-Grained Events in Video.pdf, Deep learning for action spotting in association football videos.pdf
**First correct chunk at rank:** 3

**Retrieved:**
1. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.8-8, score=0.563)
2. [ ] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.6-6, score=0.432)
3. [x] Deep learning for action spotting in association football videos.pdf (p.20-20, score=0.422)
4. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.17-17, score=0.399)
5. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.5-5, score=0.396)

---

### M02 (multi_paper) -- What are the different ways these papers capture long-range temporal context?
*Why this question is here: Breadth question. The top-5 must span several papers, not five chunks of one.*

**Correct paper(s):** Spotting Temporally Precise, Fine-Grained Events in Video.pdf, Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf, T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf, Multi-Focus Temporal Shifting for Precise Event.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.1-1, score=0.487)
2. [ ] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.2-2, score=0.486)
3. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.3-3, score=0.482)
4. [x] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.8-8, score=0.476)
5. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.7-7, score=0.468)

---

### M03 (multi_paper) -- Which papers build on or compare against E2E-Spot?
*Why this question is here: E2E-Spot is cited nearly everywhere. Tests whether breadth survives a very common term.*

**Correct paper(s):** T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf, Multi-Focus Temporal Shifting for Precise Event.pdf, Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf, Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf
**First correct chunk at rank:** 5

**Retrieved:**
1. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.14-14, score=0.416)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.394)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.3-3, score=0.385)
4. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.364)
5. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.6-6, score=0.353)

---

### M04 (multi_paper) -- What evaluation metric is used across most of these papers?
*Why this question is here: Easy fact, but only genuinely correct if drawn from several papers.*

**Correct paper(s):** T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf, Multi-Focus Temporal Shifting for Precise Event.pdf, Deep learning for action spotting in association football videos.pdf
**First correct chunk at rank:** 2

**Retrieved:**
1. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.13-13, score=0.458)
2. [x] Deep learning for action spotting in association football videos.pdf (p.18-18, score=0.409)
3. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.18-18, score=0.403)
4. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.19-19, score=0.396)
5. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.13-13, score=0.384)

---

### M05 (multi_paper) -- Which sports are covered across this collection of papers?
*Why this question is here: Corpus-coverage question. A good answer touches nearly every paper.*

**Correct paper(s):** Spotting Temporally Precise, Fine-Grained Events in Video.pdf, Multi-Focus Temporal Shifting for Precise Event.pdf, Deep learning for action spotting in association football videos.pdf, T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Deep learning for action spotting in association football videos.pdf (p.23-23, score=0.563)
2. [x] Deep learning for action spotting in association football videos.pdf (p.26-26, score=0.493)
3. [x] Deep learning for action spotting in association football videos.pdf (p.1-1, score=0.491)
4. [x] Deep learning for action spotting in association football videos.pdf (p.26-26, score=0.489)
5. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.16-16, score=0.477)

---

### M06 (multi_paper) -- What is the Gate Shift Module, and which papers here use or extend it?
*Why this question is here: One concept threaded through three papers.*

**Correct paper(s):** Spotting Temporally Precise, Fine-Grained Events in Video.pdf, Multi-Focus Temporal Shifting for Precise Event.pdf, Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf
**First correct chunk at rank:** 4

**Retrieved:**
1. [ ] Deep learning for action spotting in association football videos.pdf (p.29-29, score=0.454)
2. [ ] Deep learning for action spotting in association football videos.pdf (p.29-29, score=0.447)
3. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.2-2, score=0.420)
4. [x] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.4-4, score=0.405)
5. [x] Multi-Focus Temporal Shifting for Precise Event.pdf (p.3-3, score=0.372)

---

### M07 (multi_paper) -- Across these papers, what are the main open challenges in precise event spotting?
*Why this question is here: Open-ended synthesis. The single most likely question in the set to drift off-source.*

**Correct paper(s):** Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf, Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf, Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf
**First correct chunk at rank:** not in top-5

**Retrieved:**
1. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.2-2, score=0.612)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.5-5, score=0.610)
3. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.3-3, score=0.607)
4. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.26-26, score=0.605)
5. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.5-5, score=0.601)

---

### U01 (unanswerable) -- What mAP does E2E-Spot achieve on cricket boundary detection?
*Why this question is here: TRAP: cricket appears exactly once in the corpus, as a definition example in the Sony paper, with no results attached. Retrieval will surface it confidently.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.510)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.506)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.3-3, score=0.434)
4. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.12-12, score=0.406)
5. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.21-21, score=0.400)

---

### S19 (simple) -- How does T-DEED perform on the SoccerNet Ball Action Spotting benchmark?
*Why this question is here: MISLABELLED AT FIRST, and the evaluation caught it. This was written as an unanswerable trap on the reasoning that the T-DEED paper mentions SoccerNet Ball Action Spotting without evaluating on it -- which is true. But the SoccerNet chapter covers T-DEED's win in detail, with its scores in a results table, so the answer is in the corpus, just in a different paper. The system answered it correctly and was scored as hallucinating. Kept as a reminder that a wrong label looks exactly like a model failure until someone reads the flagged case.*

**Correct paper(s):** Deep learning for action spotting in association football videos.pdf
**First correct chunk at rank:** 1

**Retrieved:**
1. [x] Deep learning for action spotting in association football videos.pdf (p.4-4, score=0.706)
2. [x] Deep learning for action spotting in association football videos.pdf (p.16-16, score=0.705)
3. [x] Deep learning for action spotting in association football videos.pdf (p.31-31, score=0.691)
4. [x] Deep learning for action spotting in association football videos.pdf (p.8-8, score=0.686)
5. [x] Deep learning for action spotting in association football videos.pdf (p.6-6, score=0.684)

---

### U03 (unanswerable) -- What accuracy do these methods achieve on basketball free-throw spotting?
*Why this question is here: TRAP: basketball appears only as an illustrative example in the survey's task definitions.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.9-9, score=0.658)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.26-26, score=0.636)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.2-2, score=0.621)
4. [ ] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.6-6, score=0.599)
5. [ ] Deep learning for action spotting in association football videos.pdf (p.28-28, score=0.594)

---

### U04 (unanswerable) -- How many GPU-hours does it take to train MFS?
*Why this question is here: Never reported. A plausible-sounding efficiency question next to real GFLOP numbers.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.15-15, score=0.489)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.7-7, score=0.470)
3. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.15-15, score=0.457)
4. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.4-4, score=0.430)
5. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.428)

---

### U05 (unanswerable) -- What licence is the Table Tennis Australia dataset released under?
*Why this question is here: TRAP: 'CC BY' does appear in the corpus, but as the licence of a paper, not of that dataset.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.17-17, score=0.548)
2. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.17-17, score=0.486)
3. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.4-4, score=0.464)
4. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.6-6, score=0.443)
5. [ ] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.6-6, score=0.433)

---

### U06 (unanswerable) -- How many annotators labelled SoccerNet-v2, and what was their inter-annotator agreement?
*Why this question is here: Zero occurrences of inter-annotator agreement anywhere in the corpus.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.16-16, score=0.580)
2. [ ] Deep learning for action spotting in association football videos.pdf (p.4-4, score=0.576)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.6-6, score=0.575)
4. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.16-16, score=0.545)
5. [ ] Deep learning for action spotting in association football videos.pdf (p.5-5, score=0.530)

---

### U07 (unanswerable) -- What is the inference latency of the Sony ASTRM model in milliseconds?
*Why this question is here: TRAP: latency numbers do exist in the corpus, but only in the multimodal soccer paper.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.370)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.26-26, score=0.358)
3. [ ] Temporal Feature Distillation for Label - Efficient Precise Event Spotting in Sports Videos.pdf (p.2-2, score=0.352)
4. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.348)
5. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.3-3, score=0.345)

---

### U08 (unanswerable) -- How does T-DEED perform when tested on a sport it was never trained on?
*Why this question is here: Cross-sport generalisation is never evaluated for T-DEED.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.20-20, score=0.444)
2. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.18-18, score=0.443)
3. [ ] Towards Precise Action Spotting - Addressing Temporal Misalignment in Labels with Dynamic Label Assignment.pdf (p.5-5, score=0.439)
4. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.13-13, score=0.436)
5. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.2-2, score=0.432)

---

### U09 (unanswerable) -- Which of these methods can run on a mobile phone?
*Why this question is here: TRAP: 'mobile' appears once in the corpus, inside a MoViNets citation title only.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.5-5, score=0.298)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.4-4, score=0.295)
3. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.2-2, score=0.263)
4. [ ] Precise Event Spotting in Sports Videos - Solving Long-Range Dependency and Class Imbalance.pdf (p.7-7, score=0.247)
5. [ ] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.4-4, score=0.245)

---

### U10 (unanswerable) -- What is the carbon footprint of training these event spotting models?
*Why this question is here: Zero occurrences. A clean negative control -- retrieval confidence should be visibly low.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.19-19, score=0.592)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.5-5, score=0.575)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.14-14, score=0.542)
4. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.4-4, score=0.535)
5. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.9-9, score=0.535)

---

### U11 (unanswerable) -- Which commercial broadcasters have deployed E2E-Spot in production?
*Why this question is here: TRAP: commercial use and industry deployment are discussed generally, never for E2E-Spot.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.409)
2. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.13-13, score=0.403)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.14-14, score=0.371)
4. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.3-3, score=0.368)
5. [ ] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.360)

---

### U12 (unanswerable) -- Do any of these methods support live streaming input rather than recorded video?
*Why this question is here: Zero occurrences of 'streaming' in the corpus.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.6-6, score=0.413)
2. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.18-18, score=0.405)
3. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.11-11, score=0.400)
4. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.20-20, score=0.393)
5. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.14-14, score=0.386)

---

### U13 (unanswerable) -- What does it cost in US dollars to annotate one full SoccerNet game?
*Why this question is here: TRAP: dollar figures exist (global sports market revenue) but never annotation cost.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.16-16, score=0.527)
2. [ ] Deep learning for action spotting in association football videos.pdf (p.4-4, score=0.526)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.6-6, score=0.518)
4. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.12-12, score=0.512)
5. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.16-16, score=0.510)

---

### U14 (unanswerable) -- What privacy or ethical concerns do these papers raise about player tracking?
*Why this question is here: Zero occurrences of privacy or ethics. Highly plausible for the domain, absent from the corpus.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.23-23, score=0.373)
2. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.2-2, score=0.358)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.3-3, score=0.345)
4. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.2-2, score=0.343)
5. [ ] Deep learning for action spotting in association football videos.pdf (p.1-1, score=0.338)

---

### U15 (unanswerable) -- How does the multimodal soccer paper separate crowd noise between home and away stadiums?
*Why this question is here: TRAP: the right paper, the right topic area, at a level of detail that does not exist.*

**Expected behaviour:** refuse -- the corpus does not contain this answer.

**Retrieved:**
1. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.22-22, score=0.456)
2. [ ] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.14-14, score=0.450)
3. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.20-20, score=0.450)
4. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.24-24, score=0.443)
5. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.3-3, score=0.420)

---
