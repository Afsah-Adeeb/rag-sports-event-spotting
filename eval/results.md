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

## Answers

| Metric | Score [95% CI] |
|---|---|
| Fact coverage | 0.71 [0.63-0.80] |
| All required facts present | 0.41 [0.28-0.57] |
| Correct refusal (unanswerable) | 1.00 [0.78-1.00] |
| Over-refusal (answerable) | 0.12 [0.05-0.26] |

Refusal is reported in **both** directions on purpose. A system that declines everything scores 1.00 on correct refusal and 1.00 on over-refusal; neither number means anything alone.

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

**Answer:**

E2E-Spot is built on a compact, end-to-end architecture designed to be trained efficiently on a single GPU while handling large temporal windows. Its core design relies on three main principles:

*   **Task-specialized spatial-temporal features:** The model utilizes features specifically tailored for the action spotting task.
*   **Reasoning over sufficient temporal context:** The model functions as a sequence model capable of making stateful predictions across long temporal contexts (e.g., hundreds of frames).
*   **Efficient end-to-end learning:** The architecture integrates well-established components to simplify the pipeline and improve accuracy while maintaining a compact structure to manage memory constraints.

Additionally, the model may incorporate enhancements like larger CNNs or optical flow, and in specific configurations (such as for the FigureSkating dataset), it can employ dilation to improve performance with larger error tolerances.

**Required facts:** 0/3
**Missing:** RegNet, GSM/Gate Shift, GRU/gated recurrent

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

**Answer:**

The provided text does not contain information regarding which specific datasets the E2E-Spot authors added frame-accurate annotations to. It only mentions that the researchers utilized datasets such as FineGym and FineDiving to demonstrate the generality of the precise spotting task.

**Required facts:** 0/2
**Missing:** Tennis, Figure Skating/FigureSkating
**Refusal check:** REFUSED although the answer exists (over-refusal)

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

**Answer:**

The SGP-Mixer layer is a component designed to aggregate features from different temporal scales within the decoder blocks. It extends the original SGP layer to accommodate two input features: a feature $z$ from the preceding decoder layer and a feature $x$ from a skip connection, which have different temporal dimensions ($L/k^j$ and $L/k^{j-1}$, respectively).

The SGP-Mixer layer addresses two primary challenges:
1. **Multi-scale aggregation:** By incorporating the SGP-Mixer module, the layer effectively merges features across different temporal scales. It achieves this by using linear interpolation to upsample features from the previous layer to match the temporal dimensions of the skip connection.
2. **Token discriminability:** The module follows SGP principles by utilizing instant-level and window-level branches. The instant-level branch calculates the distance from the clip-level average token, while the window-level branch captures temporal information across multiple receptive fields. This structure enhances token discriminability while modeling temporal information, which is critical for precise frame-level predictions.

**Required facts:** 3/3

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

**Answer:**

Based on the provided documents, T-DEED is explicitly evaluated on the FineDiving dataset, where it outperforms state-of-the-art methods by +4.87 on the tight metric and +3.58 on the loose metric. 

The provided text mentions that T-DEED is compared with the E2E-Spot baseline on the Tennis and Figure Skating datasets; however, the specific performance improvement values for T-DEED over E2E-Spot on these specific datasets are not provided in the context.

**Required facts:** 2/3
**Missing:** 4.83/3.07/1.15

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

**Answer:**

The SoftIC loss is designed to make features more compact and increase their separability. It is inspired by the Instance Contrastive (IC) Loss and is specifically intended to be used in conjunction with mixup augmentation. Information regarding the specific mathematical formulation of SoftIC is not provided in the context.

**Required facts:** 2/3
**Missing:** class imbalance/imbalance

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

**Answer:**

The ASTRM module enhances backbone features by jointly incorporating spatial and temporal cues. It is composed of three primary components: local spatial, local temporal, and global temporal blocks. Research indicates that the local spatial component alone is insufficient for capturing temporal information, and that the global temporal component provides superior performance compared to the local temporal component. Following feature refinement by these blocks, the output is passed into a temporal module, consisting of a bidirectional GRU and an MLP, to produce per-frame event classifications.

**Required facts:** 3/3

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

**Answer:**

MFS is designed to address limitations in the Gate Shift Module (GSM) by expanding its restricted temporal context and improving its spatial selectivity.

**Required facts:** 1/2
**Missing:** receptive field/temporal range/adjacent

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

**Answer:**

Table Tennis Australia (TTA) is a table tennis performance event spotting benchmark designed to address gaps in existing datasets by capturing real-world recording conditions, such as non-ideal camera angles and frequent occlusions. It consists of 39 full-game videos recorded at 30 FPS with a resolution of 1280×720. The dataset contains 4,878 precisely annotated events categorized into eight classes—serve, bounce, forehand, and backhand, each distinguished by near or far table sides. The total event count is split into 3,759 training, 598 validation, and 521 test samples.

**Required facts:** 1/2
**Missing:** 4,800/4800

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

**Answer:**

Applying DINO directly to precise event spotting is ineffective because its projection-head alignment approach prioritizes representation invariance, which suppresses the subtle motion cues necessary for temporal discrimination. Specifically, DINO features remain near-constant (cosine similarity ≥0.999) across ground-truth event boundaries, indicating that the method discards the boundary-sensitive representations required to distinguish between semantically distinct, visually similar adjacent frames. In the absence of supervised guidance, these crucial motion cues are treated as noise and suppressed.

**Required facts:** 3/3

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

**Answer:**

Based on the provided documents, there is no definition or description of a method specifically titled "Transformer Gate Shift." The text discusses "Gate-Shift Networks" and "Gate-Shift-Fuse" for video action recognition, as well as TGS (which utilizes element-wise gating for temporal information flow), but it does not describe a method combining Transformer architectures with gate shifting.

**Required facts:** 1/3
**Missing:** multi-scale/multi scale, Vision Transformer/ViT

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

**Answer:**

Temporal misalignment in action spotting refers to the discrepancy between frames tagged with an action label and the actual timing of the event's occurrence. This issue arises primarily from two factors: human annotation errors and the inherent difficulty of precisely defining event boundaries across neighboring frames. Additionally, the process of manually identifying exact event frames is labor-intensive, often requiring annotators to repeatedly navigate between frames to verify timings.

**Required facts:** 2/2

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

**Answer:**

The dynamic label assignment method improves training consistency by moving away from static assignments based solely on ground-truth times. Instead, it employs a matching process that calculates costs between ground-truth labels and predictions based on two primary factors: predicted action class scores and temporal offsets from the tagged event times.

By incorporating these class scores into the matching process, the method identifies pairs that minimize the total cost. This allows the model to assign ground-truth labels to predictions that exhibit temporal offsets from the tagged event times, provided the predicted scores align with the ground-truth action class. This approach specifically addresses temporal misalignment in labels and reduces the need for post-processing. Performance improvements are observed when using this method, although its effectiveness can be limited in datasets where the definition of event frames is ambiguous or the sample size is insufficient for the transformer-based architecture.

**Required facts:** 2/3
**Missing:** object detection

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

**Answer:**

SoccerNet Action Spotting (v1) contains 500 full broadcast football matches. SoccerNet Action Spotting (v2) contains 110,458 annotated actions across 17 different action classes.

**Required facts:** 2/3
**Missing:** 550

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

**Answer:**

In the context of the SoccerNet challenge, the tight and loose average-mAP metrics refer to different temporal tolerances (δ) used for evaluation. The tight setting typically covers tolerances in the 1–5 second range, while the loose setting covers a broader range of 5–60 seconds.

A significant limitation of these metrics is that the evaluation of mAP at specific tolerances is sensitive to the input window size. Models that rely on small temporal windows (such as an 8-second window) may produce misleading values when evaluated at tolerances greater than the window size. Furthermore, the overall interpretation and comparability of these scores are complicated by inconsistencies in how evaluation toolkits handle spurious predictions, particularly when contradictory predictions are made at the same frame or when classes without ground-truth events are treated differently across implementations.

**Required facts:** 3/3

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

**Answer:**

The primary differences between these tasks lie in their output representation, temporal precision, and use cases:

*   **Temporal Action Localization (TAL):** Represents actions as temporal intervals (defined by start and end times). It is best suited for long-duration, continuous actions and typically allows for a wider tolerance window (approximately 1–5 seconds). It involves high annotation costs.
*   **Action Spotting (AS):** Represents actions using single keyframes rather than intervals. It is better suited for fast-paced, ambiguous actions common in sports. Compared to TAL, it has medium annotation costs and uses a tolerance window of 5–60 frames.
*   **Precise Event Spotting (PES):** An extension of AS that also uses single keyframes to represent events but demands significantly higher temporal precision. It requires frame-accurate detection with a very tight tolerance window of 0–2 frames, making it suitable for identifying fine-grained critical events. 

While TAL identifies temporal regions, AS and PES focus on identifying specific moments in time, with PES distinguishing itself through stricter evaluation tolerances.

**Required facts:** 2/3
**Missing:** frame-level/exact frame/frame level

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

**Answer:**

The current benchmarks are criticized for a significant gap between academic settings and real-world deployment. Existing datasets are primarily curated from professional broadcasts, which feature high-quality camera work, controlled lighting, and fixed angles. Consequently, these models often fail to generalize to real-world scenarios, such as amateur or semi-professional footage captured with handheld devices or single-angle setups under suboptimal conditions.

Regarding evaluation protocols, the survey notes that current standards frequently rely on mean Average Precision (mAP) computed with low confidence thresholds (e.g., 0.1), which permits multiple class predictions per frame. This approach is viewed as problematic for sports video event-spotting, where a single frame almost never contains more than one event.

**Required facts:** 1/2
**Missing:** permissive/multi-label

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

**Answer:**

The paper implements an audio model based on Log-Mel spectrograms. This model is used to test different audio sample windows to assess event detection and classification performance. While the audio model alone demonstrates poor performance compared to visual models, combining the two modalities generally improves performance, particularly for goals. For other events like cards and substitutions, the impact of the audio information is variable and can occasionally be detrimental depending on the tolerances applied.

**Required facts:** 1/2
**Missing:** ResNet

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

**Answer:**

No, adding audio does not always improve soccer event detection. While utilizing multiple modalities generally improves performance—and is particularly effective for events such as goals—the impact of audio is inconsistent for other event types. For events like cards and substitutions, the performance gain depends on the configured temporal tolerances, and in certain instances, adding audio information can actually be detrimental to detection performance.

**Required facts:** 2/2

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

**Answer:**

Based on the provided documents, several models are trained on single-GPU configurations:

*   **E2E-Spot:** Both the 200MF and 800MF variants are trained on a single NVIDIA A5000 GPU.
*   **MFS:** Experiments for this model are conducted on a single NVIDIA L40S GPU.

While the provided text mentions that the classification models (3D-CNN, 2D-ResNet, and CALF) were implemented and trained on an Nvidia DGX-2 system containing 16 Tesla V100 GPUs, it does not explicitly state whether those specific models can be trained on a single card or if they strictly require multi-GPU setups.

**Required facts:** 1/2
**Missing:** single GPU/one GPU

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

**Answer:**

Existing event-spotting methods struggle with sample imbalance, as certain events—such as "Ball out of play"—occur far more frequently than rare events like "Red card" or "Yellow→Red card." This imbalance results in poor performance for these underrepresented classes. 

While the provided text identifies this challenge as a significant issue for current methods, it does not explicitly describe a specific technical mechanism within the discussed systems for mitigating this class imbalance, other than noting it as a motivation for further research. Regarding data management, researchers address the background class by sampling between events to create a dataset that includes both event instances and background segments.

**Required facts:** 1/2
**Missing:** SoftIC/contrastive

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

**Answer:**

When temporal misalignment occurs in ground-truth labels—where the annotated timestamp does not perfectly align with the actual action—it can negatively affect model performance. To address this, a dynamic label assignment strategy has been proposed to mitigate the effects of such errors.

This strategy draws on the concept of minimum-cost matching from spatial object detection and applies it to the temporal domain. During training, it allows model predictions to have temporal offsets from the ground-truth action times. By calculating matching costs based on predicted action class scores and their temporal offsets, the method dynamically assigns labels to the most likely predictions, even when those predictions deviate from the ground-truth timestamps. This approach ensures consistent event spotting and is particularly effective in scenarios where events are visually distinct but suffer from common temporal label misalignment.

**Required facts:** 2/2

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

**Answer:**

Yes, several works in the provided context utilize both audio and visual modalities for event detection and classification:

*   **Multimodal Fusion via Softmax:** One approach involves using visual models (specifically 2D-ResNet for audio transformed into Log-Mel spectrograms) and fusing them using softmax predictions. Research shows that adding causally related audio information generally improves performance for state-of-the-art visual models, particularly for events like goals where the audio signal is predictable (e.g., crowd cheering or excited commentators). However, for other events like cards or substitutions, the benefit varies, and in some cases, adding audio can be detrimental.
*   **Systematic Studies of Fusion:** Etsen et al. conducted an early systematic study on the SoccerNet dataset, testing early, late, and hybrid fusion strategies. Their findings indicate that late fusion—processing audio and visual streams independently before combining them—is most effective, as it prevents the noise that can occur with tighter integration at the feature extraction stage.
*   **Transformer-based Architectures:** Xarles et al. proposed ASTRA, which uses a Transformer-based encoder–decoder architecture. Instead of simple concatenation, ASTRA uses learnable cross-modal queries within a multi-head attention framework to allow the model to adaptively focus on relevant cues from both audio and visual embeddings.

**Required facts:** 2/2

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

**Answer:**

If training data is limited, you can employ the following strategies:

*   **Active Learning:** You can use an active learning framework to minimize annotation requirements. This involves training a baseline model on a small labeled subset and using uncertainty heuristics (such as entropy and confidence) to identify and select the most informative samples for manual annotation. This iterative process can achieve competitive performance with significantly less labeled data compared to fully supervised approaches.
*   **Leverage Priors and Domain Adaptation:** When end-to-end learning is insufficient due to limited data or pixel-domain shifts, you may need to incorporate reliable priors, such as human pose.
*   **Weak/Few-Shot Learning:** Future research is focused on developing methods to spot events accurately with few or weak labels, which facilitates the creation of new datasets for downstream tasks.
*   **Dynamic Label Assignment:** Implementing dynamic label assignment strategies can improve training consistency and performance, particularly when dealing with temporal misalignment in existing ground-truth labels.

Note that transformer-based models may perform poorly when training data is extremely limited, as they require a substantial number of samples to generalize effectively.

**Required facts:** 0/2
**Missing:** semi-supervised/label-efficient/distillation, 10%/unlabeled/unlabelled

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

**Answer:**

Researchers evaluate spotting performance based on the specific requirements of the task:

*   **Precise Spotting:** This approach measures accuracy by checking if a detection falls within a very tight tolerance of the true event, often evaluating performance at a tolerance of 0 frames (the exact frame of human annotation).
*   **Coarse Action Spotting:** A detection is considered correct if it occurs within a loose time-window around the true event, typically ranging from 1–5 seconds or 5–60 seconds (equating to 10–100s of frames).
*   **Temporal Action Detection (TAD):** This is evaluated using interval-based metrics such as mean Average Precision (mAP) at a specific temporal Intersection-over-Union (IoU). 
*   **Temporal Action Segmentation (TAS):** This optimizes for interval-based metrics like F1 score at temporal overlap. While TAS also uses frame-level metrics, these reward accuracy on densely labeled intra-segment frames, whereas event frames in precise spotting datasets are sparse.

Unlike these methods, the E2E-Spot approach specifically advocates for the direct regression of timestamp values rather than classifying video chunks followed by Non-Maximum Suppression (NMS).

**Required facts:** 2/2

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

**Answer:**

Inspired by Faster R-CNN, Xu et al. created an end-to-end model for temporal action detection that generates temporal proposals, effectively applying the architecture originally designed for object detection to the temporal domain.

**Required facts:** 2/3
**Missing:** matching

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

**Answer:**

Yes, there are several benchmarks for table tennis, which is played indoors on a table:

*   **OpenTTGames:** Contains 12 high-definition table tennis matches recorded at 120 FPS, featuring 4,271 events annotated with frame-level precision (ball bounces, net hits, and empty events). It is specifically designed for high-speed gameplay conditions.
*   **P2A:** A large-scale dataset of 2,721 broadcast videos covering 14 fine-grained stroke classes across 8 action categories, with professional-validated frame-level annotations.
*   **TTA (Table Tennis Australia):** A benchmark consisting of 39 full-game videos (30 FPS) with 4,878 events. It includes eight classes (serve, bounce, forehand, and backhand) labeled by table side (near/far) and is notable for including non-ideal camera angles and occlusions. It was initially focused on "pass" and "drive" actions and was expanded in 2024 to include 12 ball-related classes.

**Required facts:** 2/2

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

**Answer:**

The provided context does not contain sufficient information to directly compare T-DEED’s approach to temporal resolution against E2E-Spot’s. 

While the text notes that T-DEED outperforms state-of-the-art methods on the FineDiving dataset and mentions that E2E-Spot utilizes dilation, it does not provide a technical breakdown of how T-DEED manages temporal resolution or explicitly contrast its architectural strategy with that of E2E-Spot.

**Required facts:** 2/3
**Missing:** encoder-decoder
**Refusal check:** REFUSED although the answer exists (over-refusal)

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

**Answer:**

The provided text does not contain information regarding a "Transformer Gate Shift" model or its comparison to the Multi-scale Feature Shift (MFS) module. Therefore, I cannot answer this question.

**Required facts:** 1/3
**Missing:** Vision Transformer/ViT, CNN
**Refusal check:** REFUSED although the answer exists (over-refusal)

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

**Answer:**

T-DEED and the methodology introduced in the Sony paper employ different strategies to enhance frame-level discriminability:

*   **T-DEED** focuses on enhancing token discriminability through architectural design. It utilizes a temporally-discriminant encoder-decoder neck featuring Scalable-Granularity Perception (SGP) layers and SGP-Mixer layers to process multiple temporal scales. Additionally, it incorporates Gate-Shift-Fuse (GSF) modules within its 2D RegNetY60 backbone to produce spatio-temporal per-frame representations.

*   **The Sony paper** addresses discriminability by introducing Temporal Feature Distillation (TFD). Unlike methods that rely on projection-level representations (such as DINO, which shows high inter-frame similarity that suppresses temporal discriminability), TFD aligns temporally structured backbone features directly from a TGS-enhanced ViT. This allows the model to preserve boundary-sensitive representations, as evidenced by sharp drops in inter-frame cosine similarity at event boundaries.

The provided context does not offer further details regarding the specific mechanisms of the "Sony paper" beyond the introduction of Temporal Feature Distillation.

**Required facts:** 1/2
**Missing:** SoftIC/contrastive/ASTRM

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

**Answer:**

MFS provides SOTA-level accuracy with significant gains in computational efficiency compared to heavier baselines like ASTRM. Specifically, MFS achieves performance parity with ASTRM while requiring 45.06 GFLOPs compared to ASTRM’s 67.24 GFLOPs. This represents a 33% reduction in FLOPs, or a 1.5× efficiency gain. These results demonstrate that MFS can achieve top-tier performance without the computational overhead associated with complex temporal mechanisms.

**Required facts:** 2/2

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

**Answer:**

Yes, both E2E-Spot and T-DEED utilize the RegNet-Y 200MF and 800MF backbones.

**Required facts:** 3/3

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

**Answer:**

The provided text discusses a dynamic label assignment method that addresses issues with training labels. Rather than relying on static assignment, this approach calculates matching costs between ground-truth labels and predictions to assign labels based on predicted class scores and times. This technique is designed to mitigate the impact of temporal misalignment in labels by allowing for offsets when predicted scores align with the ground-truth action class.

Additionally, the text mentions an evaluation using noisy labels, though it does not explicitly name specific papers or authors associated with these methods.

**Required facts:** 2/2

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

**Answer:**

The provided text does not contain a separate "survey definition" to compare against the SoccerNet definition. It only provides the formal definition of action spotting as used within the context of the SoccerNet research, which defines the task as the precise temporal localization of actions—identified by a single timestamp—within long, untrimmed video streams.

**Required facts:** 2/2
**Refusal check:** REFUSED although the answer exists (over-refusal)

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

**Answer:**

Based on the provided documents, the following datasets or benchmarks are introduced or described:

*   **OpenTTGames**: A table tennis dataset consisting of 12 high-definition matches recorded at 120 FPS, containing 4,271 labeled events, including ball bounces, net hits, and empty events.
*   **P2A**: A large-scale table tennis dataset comprised of 2,721 broadcast videos (272 hours) featuring 14 fine-grained stroke classes grouped into 8 higher-level action categories.
*   **TTA**: A table tennis benchmark consisting of 39 para-professional matches captured under real-world conditions, including non-ideal camera angles, occlusions, and less controlled environments.

The text also mentions using Tennis, FineDiving, Figure Skating (FS), and FineGym (FG) as benchmark datasets, but it does not specify which papers originally introduced them.

**Required facts:** 2/3
**Missing:** SoccerNet

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

**Answer:**

The provided text describes several approaches for capturing long-range temporal context in event-spotting tasks, which generally fall into two categories:

*   **Transformer-based methods:** These architectures are capable of capturing long-range temporal dependencies but are noted for their high computational overhead, often exceeding 600 GFLOPs.
*   **Lightweight temporal modules:** These approaches are designed for computational efficiency (typically 40–45 GFLOPs) and include:
    *   **Shift-based modules:** Methods such as TSM, GSM, and GSF use frame shifts to model temporal information. Because standard GSM is limited to a narrow temporal receptive field (t±1), the MFS (Multi-scale Feature Shift) approach was introduced to provide direct access to frames at varying distances (e.g., δ ∈ {1, 2, 3}) by using learnable multi-scale temporal shifts.
    *   **Recurrent and gated networks:** Research comparing different temporal network architectures for long-range dependencies indicates that Bi-GRU (Bidirectional Gated Recurrent Unit) performs favorably compared to other options.
    *   **SGP (Spatial-Temporal Gated Pooling):** These layers are used to model temporal information while maintaining high token discriminability, which helps prevent precision loss in tasks where adjacent frames may be spatially similar.

**Required facts:** 3/3

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

**Answer:**

The provided text does not contain information about other papers that build on or compare against E2E-Spot. While the context mentions that E2E-Spot itself compares against and outperforms strong baselines on precise temporal spotting tasks and is evaluated against prior work on the SoccerNet Action Spotting challenge, it does not name the specific papers that utilize or compare against E2E-Spot in their own research.

**Required facts:** 2/3
**Missing:** T-DEED
**Refusal check:** REFUSED although the answer exists (over-refusal)

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

**Answer:**

The primary evaluation metric used depends on the specific task:

*   **Temporal Action Localization (TAL):** The standard metric is mean Average Precision computed with temporal Intersection over Union thresholds (mAP@T-IoU).
*   **Action Spotting (AS) and Procedural Event Spotting (PES):** The primary metric is mAP@δ, which is calculated based on a temporal tolerance window (δ) around ground-truth timestamps. 

Additionally, the Area Under the Curve (AUC) is used to measure average recall across varying proposal counts (in proposals-based tasks) or to aggregate performance across different tolerance windows (δ ranging from 5–60 s).

**Required facts:** 1/1

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

**Answer:**

Based on the provided documents, the sports covered in the datasets and research include:

*   **Soccer:** Represented by datasets such as SoccerNet, SoccerNet-v2, SoccerNet Ball AS, SSET, SoccerDB, and TACDEC.
*   **Tennis:** Represented by the Tenniset and the Tennis dataset.
*   **Table Tennis:** Represented by OpenTTGames, P2A, and TTA.
*   **Basketball:** Represented by the NCAA dataset.

Additionally, the DeepSportradar-v2 dataset is noted as a multi-sport dataset.

**Required facts:** 2/3
**Missing:** figure skating/figureskating/diving/finediving

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

**Answer:**

The Gate Shift Module (GSM) is a mechanism used for video action recognition that extracts per-frame features containing short-term temporal information. It functions by applying temporal shifts to feature maps to facilitate information exchange across frames.

The following papers in the collection utilize or extend this concept:

*   **Sudhakaran et al.** introduced Gate-shift networks for video action recognition.
*   **Sudhakaran et al.** later published an extension titled "Gate-shift-fuse for video action recognition."
*   **The provided context** also describes a "Multi-scale Gated Temporal Shift" (MFS) approach that applies gated temporal shifts at multiple distances, as well as a "Temporal Gating Shift" (TGS) module that performs element-wise gating and shifting across past, future, and static feature groups to provide finer control over temporal information flow.

**Required facts:** 2/3
**Missing:** E2E-Spot

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

**Answer:**

Based on the provided context, the main challenges in Precise Event Spotting (PES) include:

*   **Temporal Precision:** Unlike Temporal Action Localization (TAL) which specifies intervals, PES requires identifying the exact frame where an event occurs. This necessitates a very small temporal tolerance ($\delta$), requiring models to pinpoint occurrences within just a few frames.
*   **Feature and Context Requirements:** Models must balance two conflicting requirements. They need task-specific, local spatial-temporal features to capture subtle visual and motion differences across neighboring frames, while simultaneously employing long-term temporal reasoning. This long-term window is necessary to disambiguate short, rare events that are difficult to identify in isolation but clearer when viewed within a broader temporal context (e.g., the acceleration and landing phases surrounding a jump).
*   **Data Sparsity:** Precise events are often extremely rare within video datasets (e.g., constituting only 0.23% to 2.2% of total frames in the provided examples).
*   **Annotation Effort:** Achieving frame-accurate labels for these events is demanding, as demonstrated by the need to manually re-annotate existing action datasets to reach the necessary temporal precision.
*   **Optimization Trade-offs:** Techniques like Non-Maximum Suppression (NMS) can improve precision by suppressing nearby false detections but may negatively impact recall.

**Required facts:** 2/3
**Missing:** imbalance

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

**Answer:**

The provided context does not contain information regarding E2E-Spot's performance on cricket boundary detection.

**Refusal check:** REFUSED (correct)

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

**Answer:**

T-DEED is the winning method of the SoccerNet Ball Action Spotting (2024) challenge. It utilizes a 2D RegNetY backbone augmented with Gate-Shift-Fuse modules to improve frame representation discriminability, achieve high temporal resolution, and capture information across multiple temporal scales to handle diverse action dynamics.

**Required facts:** 3/3

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

**Answer:**

The provided context does not contain information regarding basketball free-throw spotting or the accuracy of methods on such a task.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided context does not contain information regarding the GPU-hours required to train MFS. While it provides training cycle durations for the E2E-Spot model on an A5000 GPU and notes that MFS experiments were conducted on an NVIDIA L40S GPU, it does not specify the training duration for MFS.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided context does not contain information regarding the license under which the Table Tennis Australia (TTA) dataset is released.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided text does not contain information regarding the number of annotators who labeled SoccerNet-v2 or their inter-annotator agreement.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided context does not contain information regarding the inference latency of the ASTRM model in milliseconds. It only provides performance metrics in terms of mAP (%) and computational cost in GFLOPs.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided text does not contain information regarding the performance of T-DEED when tested on a sport it was not trained on. The documents mention T-DEED as a state-of-the-art baseline method for action-spotting but do not discuss its generalization or cross-sport transfer capabilities.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided text does not contain information regarding which of the mentioned methods are capable of running on a mobile phone.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided context does not contain information regarding the carbon footprint of training the event spotting models.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided context does not contain information regarding whether any commercial broadcasters have deployed E2E-Spot in production.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided text does not explicitly confirm whether any of the described methods support live streaming input. While the documents discuss real-time deployment, real-time sports analytics, and the need for low-latency buffering in real-time scenarios, they do not state whether these systems accept live video feeds or if they are designed to process recorded footage in a real-time manner.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided context does not contain information regarding the financial cost of annotating a SoccerNet game in US dollars.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided context does not contain any information regarding privacy or ethical concerns related to player tracking.

**Refusal check:** REFUSED (correct)

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

**Answer:**

The provided context does not contain information regarding how the multimodal soccer paper separates crowd noise between home and away stadiums.

**Refusal check:** REFUSED (correct)

---
