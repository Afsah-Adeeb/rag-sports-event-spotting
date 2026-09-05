# Closed-Book Control: what is retrieval actually worth?

Every question asked twice with the same model (`gemini-3.1-flash-lite`): once through the full RAG pipeline, and once with **no documents at all**. The gap between the two is what retrieval contributes.

This matters here because the corpus is nine well-known arXiv papers that were almost certainly in the model's training data. Without this control, a good score cannot be attributed to retrieval rather than recall.

*The closed-book arm is given its own prompt rather than the RAG prompt with an empty context block. The RAG prompt orders the model to answer only from provided context, so with no context it would refuse everything and score zero -- measuring the prompt, not the model.*

## Answer quality

| Metric | Full RAG | Closed book | Retrieval buys |
|---|---|---|---|
| Required facts present | 0.71 [0.62-0.79] | 0.58 [0.47-0.69] | **+0.13** |
| All facts present | 0.42 [0.29-0.58] | 0.35 [0.22-0.50] | +0.08 |

**Retrieval contributes +0.13** -- from 0.58 on memory alone to 0.71 with documents. That gap, not the 0.71, is the honest measure of what this pipeline adds.

### Question by question

| Outcome | Count | Questions |
|---|---|---|
| RAG better | 12 | `S03`, `S04`, `S12`, `S18`, `P01`, `P03`, `C04`, `C05`, `C06`, `M01`, `M03`, `M06` |
| Tied | 22 | `S01`, `S02`, `S05`, `S06`, `S07`, `S08`, `S09`, `S11`, `S13`, `S14`, `S16`, `S17`, `P02`, `P04`, `P07`, `P08`, `C01`, `C02`, `C03`, `C07`, `M02`, `M04` |
| Closed book better | 6 | `S10`, `S15`, `P05`, `P06`, `M05`, `M07` |

Averages hide direction. If retrieval wins on some questions and loses on others, the mean can look flat while both effects are real.

**Already known to the model (27 questions):** `S05`, `S06`, `S07`, `S08`, `S09`, `S10`, `S11`, `S14`, `S15`, `S16`, `S17`, `P02`, `P03`, `P04`, `P05`, `P06`, `P07`, `P08`, `C01`, `C03`, `C06`, `C07`, `M02`, `M03`, `M04`, `M05`, `M07`. These scored at least half their required facts with no documents supplied. The system's score on them cannot be credited to retrieval.

## Honesty: does retrieval make it more or less careful?

| Behaviour | Full RAG | Closed book |
|---|---|---|
| Refused the 15 unanswerable questions | 0.87 [0.62-0.96] | 0.00 [0.00-0.20] |
| Refused answerable questions (over-refusal) | 0.12 [0.05-0.26] | 0.00 [0.00-0.09] |

Retrieval makes the model **more** willing to decline unanswerable questions (+0.87). Seeing chunks that do not contain the answer gives it grounds to say so.

## By question type

| Type | n | Full RAG | Closed book | Retrieval buys |
|---|---|---|---|---|
| simple | 18 | 0.61 [0.47-0.77] | 0.52 [0.36-0.69] | +0.09 |
| paraphrase | 8 | 0.77 [0.60-0.94] | 0.71 [0.46-0.94] | +0.06 |
| comparison | 7 | 0.79 [0.57-0.95] | 0.43 [0.19-0.69] | +0.36 |
| multi_paper | 7 | 0.81 [0.71-0.95] | 0.76 [0.52-0.95] | +0.05 |

## Side by side

### S01 (simple) -- What are the main components of the E2E-Spot architecture?
**Required facts:** RAG 0/3 | closed book 0/3

**With retrieval:**

E2E-Spot is built upon three primary design principles: task-specialized spatial-temporal features, reasoning over sufficient temporal context, and efficient end-to-end learning. Architecturally, it utilizes a compact design that enables training on large temporal windows while remaining memory-efficient enough to be trained on a single GPU. It integrates well-established architectural components and, in specific configurations like the FigureSkating dataset, utilizes dilation to improve performance.

**From memory only:**

The **E2E-Spot** (End-to-End Spotting) architecture is designed for direct temporal event localization without a separate proposal generation stage. Its main components are:

1.  **Feature Extractor:** Typically a pre-trained backbone (e.g., I3D or SlowFast) that processes raw video segments into high-level spatial-temporal feature representations.
2.  **Encoder (Temporal Aggregation):** A temporal modeling module (often using dilated convolutions, Transformers, or Bi-LSTMs) that captures long-range dependencies and context across the video sequence.
3.  **Spotting Head (Detection Decoder):** A unified head that performs simultaneous classification and temporal boundary regression. It maps encoded features directly to event class probabilities and start/end time offsets relative to the anchor positions.
4.  **Loss Function:** A multi-task loss combining:
    *   **Classification Loss:** Usually Focal Loss or Cross-Entropy to handle class imbalance (background vs. action).
    *   **Regression Loss:** Smooth L1 or IoU-based loss to refine the precise temporal boundaries of the spotted events.

---

### S02 (simple) -- Which datasets did the E2E-Spot authors add frame-accurate annotations to?
**Required facts:** RAG 0/2 | closed book 0/2

**With retrieval:**

The provided text does not contain information regarding which datasets the E2E-Spot authors added frame-accurate annotations to. It mentions that the authors use FineGym and FineDiving to show the generality of the precise spotting task, but does not specify if they performed the annotation work themselves.

**From memory only:**

The authors of E2E-Spot added frame-accurate annotations to the **SoccerNet-v2** dataset. Specifically, they refined the original temporal annotations—which were provided as approximate timestamps (seconds)—to frame-accurate labels to support their end-to-end event spotting framework.

---

### S03 (simple) -- What is the SGP-Mixer layer in T-DEED and what problem does it solve?
**Required facts:** RAG 3/3 | closed book 1/3

**With retrieval:**

The SGP-Mixer layer is a component designed to aggregate features from different temporal scales within the decoder blocks of the architecture. It specifically addresses challenges related to temporal information modeling and token discriminability.

By integrating the SGP-Mixer module, the layer adapts the core principles of the SGP module to handle two distinct inputs: features $z \in \mathbb{R}^{L/k^j \times d}$ from the preceding layer and features $x \in \mathbb{R}^{L/k^{j-1} \times d}$ from the skip connection. The process involves upsampling the features from the previous layer via linear interpolation to match temporal dimensions, followed by the use of instant-level and window-level branches to improve token discriminability. This allows the network to effectively merge multiscale temporal information while boosting the model's ability to distinguish between tokens, which is particularly critical for precise, frame-level event spotting.

**From memory only:**

The **SGP-Mixer (Sub-Graph Pooling Mixer) layer** is a specialized module introduced in **T-DEED** (Temporal Action Detection with Event-aware Decoupling) designed to capture cross-scale temporal context while explicitly modeling the hierarchical dependencies between event segments and their surrounding action context.

### Technical Composition
The SGP-Mixer operates by constructing a directed graph where nodes represent temporal segments. It employs:
1.  **Sub-Graph Pooling:** Aggregates features from neighboring temporal segments into a localized sub-graph representation, effectively capturing context at multiple granularities.
2.  **Mixing Mechanism:** A gating or cross-attention-based mechanism that "mixes" these pooled sub-graph features with the original temporal feature sequence to refine segment-level representations.

### Problem Solved
The SGP-Mixer addresses the **contextual ambiguity** and **boundary imprecision** inherent in fine-grained sports event spotting:

*   **Contextual Over-smoothing:** Standard Transformers or TCNs often fail to distinguish between the immediate event and the background action (e.g., distinguishing a "shot" from the preceding "dribble"). The SGP-Mixer enforces a structured, hierarchical aggregation that preserves the distinct signatures of the event versus the surrounding context.
*   **Information Bottleneck:** By explicitly pooling sub-graphs, the layer mitigates the information loss that typically occurs in deep temporal stacks when high-frequency signal details (crucial for sports events) are flattened by standard global temporal pooling.
*   **Decoupling:** It supports the T-DEED architecture’s goal of decoupling event-center localization from the classification of action boundaries, ensuring that the local temporal features are robust to the scale variations common in high-speed sports footage.

---

### S04 (simple) -- On which datasets is T-DEED evaluated, and by how much does it improve over E2E-Spot?
**Required facts:** RAG 1/3 | closed book 0/3

**With retrieval:**

The provided text does not contain specific information regarding which datasets T-DEED is evaluated on, other than mentioning it is a recent state-of-the-art method. While the text notes that T-DEED outperforms state-of-the-art results on the FineDiving dataset by +4.87 on the tight metric and +3.58 on the loose metric, it does not provide a direct comparison or improvement value between T-DEED and E2E-Spot.

**From memory only:**

T-DEED (Temporal Deep Event Detection) is evaluated on the **SoccerNet** dataset.

In terms of performance, T-DEED reports an improvement in Average-mAP (Average precision) of **1.6%** over the E2E-Spot baseline (specifically, T-DEED achieves 49.3% mAP compared to E2E-Spot's 47.7% on the SoccerNet action spotting task).

---

### S05 (simple) -- What is the SoftIC loss and what problem does it address?
**Required facts:** RAG 2/3 | closed book 2/3

**With retrieval:**

The SoftIC loss is a loss function designed to make features more compact and increase their separability. It is inspired by the Instance Contrastive (IC) Loss and is intended to be used in conjunction with mixup augmentation. While the provided text mentions the SoftIC loss as a component of the training strategy (combined with Binary Cross Entropy loss using a weighting constant), it does not explicitly detail the specific mathematical formulation of the SoftIC loss itself.

**From memory only:**

The **SoftIC (Soft In-Class) loss** is a loss function primarily used in temporal action localization (TAL) to address the **foreground-background ambiguity** problem, specifically within weakly supervised or semi-supervised frameworks.

### The Problem Addressed
In sports video event-spotting and TAL, models often struggle to distinguish between discriminative action snippets and the surrounding background snippets that may share similar visual features (e.g., a "crowd" background being confused with a "celebration" event). Standard classification losses (like cross-entropy) often suffer from:
1. **Low confidence on sparse action samples:** The model may struggle to assign high scores to entire action segments if only a few frames are highly discriminative.
2. **Foreground-Background Confusion:** The model fails to maintain a clear margin between action and non-action, leading to false positives or overly fragmented action detection.

### Technical Mechanism
SoftIC addresses this by encouraging **intra-class consistency** and **inter-class separation** in the feature space:

1. **Intra-class consistency:** It forces the feature embeddings of snippets predicted as belonging to the same action class to be closer to their class prototype (or centroid).
2. **Soft-assignment:** Unlike hard-label classification, SoftIC computes a weighted loss based on the confidence scores of the snippet being an action. Snippets with higher classification scores are weighted more heavily, forcing the model to refine its temporal boundaries.
3. **Margin-based optimization:** By integrating a margin component (similar to contrastive or triplet-based approaches), SoftIC ensures that snippets identified as "background" are pushed away from the "action" feature manifold, effectively sharpening the temporal class activation maps (CAMs).

In summary, SoftIC improves the quality of temporal action proposals by enforcing **feature compactness** for action classes and **feature separability** against background noise.

---

### S06 (simple) -- What does the ASTRM module do?
**Required facts:** RAG 3/3 | closed book 3/3

**With retrieval:**

The ASTRM module enhances backbone features by jointly integrating spatial and temporal cues using three dedicated components: local spatial, local temporal, and global temporal blocks. Within this module, performance findings indicate that the spatial component alone is insufficient for capturing temporal information, and that the global temporal component provides superior performance compared to the local temporal component. By combining these three aspects, the ASTRM module contributes to achieving state-of-the-art results in temporal action localization tasks.

**From memory only:**

The **ASTRM (Attention-based Spatio-Temporal Relation Module)** is designed to capture complex inter-dependencies between detected action segments by modeling both spatial and temporal relationships. 

Specifically, it typically functions as follows:

*   **Spatio-Temporal Encoding:** It processes feature sequences (often from a backbone like I3D or SlowFast) to compute attention weights that capture long-range dependencies across the temporal dimension and spatial interactions within the frame.
*   **Contextual Refinement:** It utilizes self-attention mechanisms to refine segment features by incorporating global context. This addresses the limitation of local-only temporal window operations, allowing the model to better distinguish between action classes that share similar visual appearances but differ in global structural context.
*   **Relation Modeling:** By explicitly calculating relations between segments, it helps the network learn the hierarchical structure of sports events (e.g., the sequence of a wind-up, impact, and follow-through in a tennis serve), improving the temporal boundary localization accuracy.

In recent literature, it is often employed to mitigate the "boundary ambiguity" problem by forcing the model to consider the relationship between candidate segments and their surrounding temporal context.

---

### S07 (simple) -- What are the two limitations of the Gate Shift Module that MFS is designed to fix?
**Required facts:** RAG 1/2 | closed book 1/2

**With retrieval:**

MFS is designed to enhance the Gate Shift Module (GSM) by addressing its limitations regarding temporal context and spatial selectivity. Specifically, MFS expands the temporal context and adds spatial selectivity through the use of multi-scale shifts and grouped focusing.

**From memory only:**

The Gate Shift Module (GSM) suffers from two primary limitations that Multi-Fiber Shift (MFS) addresses:

1.  **Fixed temporal receptive field:** GSM utilizes a static shift operation with a predefined shifting ratio, which restricts the model’s ability to capture temporal dependencies at varying scales or speeds within a sports video.
2.  **Inflexible channel-wise modeling:** GSM applies a uniform gating mechanism across all channels, failing to account for the heterogeneous importance of different feature maps for complex event-spotting, leading to suboptimal feature representation.

---

### S08 (simple) -- What is the Table Tennis Australia dataset and how large is it?
**Required facts:** RAG 1/2 | closed book 1/2

**With retrieval:**

The Table Tennis Australia (TTA) dataset is a professional event-spotting (PES) benchmark for table tennis. It contains 39 full-game videos recorded at 30 FPS with a resolution of 1280x720, featuring 4,878 precisely annotated events across eight classes (serve, bounce, forehand, and backhand, with each labeled by near and far table sides). The dataset is divided into train, validation, and test sets containing 3,759, 598, and 521 events, respectively. Unlike broadcast-only datasets, it reflects real-world recording conditions, including varied camera angles and occlusions.

**From memory only:**

The Table Tennis Australia (TTA) dataset is a specialized collection of broadcast footage designed for fine-grained temporal action localization in table tennis. It is primarily used for spotting specific strokes (e.g., forehand drive, backhand push, serve) and rally segments.

**Technical Specifications:**
*   **Content:** Contains approximately 30 minutes of high-resolution professional table tennis match footage.
*   **Annotations:** It includes dense temporal annotations for individual strokes and rally events.
*   **Size/Scale:** The dataset consists of 4,000 action instances distributed across 20 distinct action classes.

It is frequently cited in the context of sports action recognition research as a benchmark for evaluating models on fast-paced, high-motion sports where temporal precision is critical.

---

### S09 (simple) -- Why does applying DINO directly to precise event spotting not work?
**Required facts:** RAG 3/3 | closed book 3/3

**With retrieval:**

Directly applying DINO to precise event spotting is ineffective because its projection-head alignment mechanism leads to the suppression of subtle, crucial motion cues, treating them as noise. This results in representations that lack sensitivity to precise event boundaries. 

Specifically, DINO features exhibit a near-constant cosine similarity (≥0.999) across ground-truth event boundaries, indicating that projection-level invariance masks the temporal discriminability required to distinguish between visually similar yet semantically distinct frames in precise event spotting.

**From memory only:**

Applying DINO (or other self-supervised Vision Transformers) directly to precise event spotting faces three primary challenges:

1.  **Temporal Granularity and Tokenization:** DINO is typically trained on image-level objectives (e.g., self-distillation, centering, and sharpening). When applied to video via frame-level tokenization, the temporal resolution is often too coarse. Standard patch-based embeddings lack the high-frequency temporal sensitivity required to pinpoint instantaneous events (e.g., the exact frame of a ball-to-racket contact) because the global attention mechanisms prioritize semantic spatial consistency over precise temporal boundaries.
2.  **Lack of Motion Dynamics:** DINO's pre-training objective is spatial-feature-focused. It learns robust object representations but remains largely agnostic to the specific motion patterns (temporal dynamics) that define a sports event. Without explicit temporal modeling—such as optical flow integration, temporal shift modules, or 3D convolutions—the latent representations fail to capture the "action transition" signatures necessary for localization.
3.  **Boundary Ambiguity (Soft vs. Hard):** Event spotting requires identifying a point in time within a continuous action. Because DINO representations are optimized for global image classification or retrieval, they lack a supervised "boundary awareness" signal. The resulting features are often smooth across time, making them incapable of producing the sharp, high-confidence peak activations necessary for non-maximum suppression (NMS) to distinguish between the event and the surrounding background context. 

In short, DINO provides excellent *semantic* features, but lacks the *temporal localization* precision and *motion-specific* discriminative power required for high-fidelity event spotting.

---

### S10 (simple) -- What is Transformer Gate Shift?
**Required facts:** RAG 1/3 | closed book 2/3

**With retrieval:**

The provided text does not contain information regarding a method called "Transformer Gate Shift." The documents describe "Gate-Shift Networks," "Gate-Shift-Fuse" (MFS), and "TGS" (which performs element-wise gating for temporal information flow), but they do not explicitly define or discuss a specific mechanism named "Transformer Gate Shift."

**From memory only:**

**Transformer Gate Shift (TGS)** is a temporal modeling mechanism designed to enhance the efficiency and performance of Vision Transformers (ViTs) in video action recognition and localization tasks. 

It addresses the high computational cost of full 3D attention by replacing standard temporal attention layers with a lightweight gating operation. The mechanism typically functions as follows:

1.  **Shift Operation:** Features are shifted along the temporal dimension, enabling the model to aggregate local temporal context without explicit attention computations (similar to the concept in TSM).
2.  **Gating Mechanism:** A learned gating module (often implemented via a 1D convolution or a small MLP) modulates the importance of the shifted features. This allows the model to selectively propagate or suppress temporal information based on the visual content.
3.  **Efficiency:** By substituting dense temporal attention with this combination of shift and gating, TGS reduces FLOPs and memory overhead while maintaining long-range dependency modeling.

It is particularly relevant in temporal action localization where balancing high-resolution temporal feature extraction with computational constraints is critical for real-time inference.

---

### S11 (simple) -- What is temporal misalignment in action spotting labels, and what causes it?
**Required facts:** RAG 2/2 | closed book 2/2

**With retrieval:**

Temporal misalignment refers to discrepancies between the ground-truth event labels provided in a dataset and the actual occurrence of the action within the video frames. 

This issue is caused by two primary factors:
*   **Human Annotation Errors:** Inaccuracies introduced during the manual labeling process.
*   **Inherent Ambiguity:** Difficulty in precisely identifying exact event boundaries across neighboring frames, leading to tagged frames that do not perfectly coincide with the visual appearance of the event. 

Furthermore, the process of precisely annotating event frames is described as cumbersome, as it requires annotators to repeatedly navigate between frames to verify boundaries.

**From memory only:**

Temporal misalignment in action spotting refers to the discrepancy between the ground-truth temporal boundaries (or timestamps) and the predicted action intervals, often manifesting as a systematic temporal shift or jitter. 

**Causes include:**

*   **Subjective Annotation Variance:** Human annotators define action starts and ends inconsistently, especially for fluid or complex sports actions (e.g., "the start of a pass" vs. "the moment of contact"). This leads to ambiguous "soft" boundaries.
*   **Sensor/Frame-Rate Latency:** Discrepancies between the temporal resolution of the ground-truth logs (e.g., manually annotated metadata) and the video frame rate, causing a misalignment of the exact frame index.
*   **Definition Inconsistency:** Lack of rigid taxonomy for action initialization. For instance, in soccer, an action might be labeled based on the *initiation of movement*, the *point of contact*, or the *release of the ball*.
*   **Temporal Smoothing/Pooling:** Architectures utilizing aggressive temporal downsampling or receptive field aggregation (e.g., dilated convolutions, transformers with large window sizes) often blur precise action boundaries, resulting in a localization offset relative to the point-annotation or segment-level ground truth.
*   **Causal Delay:** In real-time spotting models, causal convolutions or unidirectional attention mechanisms induce a systematic lag as the model requires sufficient context to confirm the action occurrence.

---

### S12 (simple) -- How does the dynamic label assignment method work?
**Required facts:** RAG 2/3 | closed book 1/3

**With retrieval:**

Dynamic label assignment replaces static, time-based ground-truth mapping with a process that considers both predicted class scores and temporal offsets. The method calculates matching costs between ground-truth labels and predictions, identifying pairs that minimize the total cost. 

By incorporating action class scores into the matching process, the model can assign ground-truth labels to predictions even when their predicted times exhibit offsets from the tagged ground-truth frames. This approach effectively mitigates temporal misalignment in labels, reducing the need for post-processing and improving training consistency compared to static assignment methods.

**From memory only:**

Dynamic label assignment (DLA) replaces static, IoU-based thresholding (e.g., in anchor-based detectors like RetinaNet or anchor-free ones like FCOS) with an optimization-based approach to determine positive and negative training samples. 

In the context of temporal action localization (TAL), the process generally follows these steps:

1.  **Cost Matrix Construction:** For each frame or proposal, a cost is computed that combines classification confidence (e.g., focal loss) and localization quality (e.g., IoU or GIoU between the predicted action segment and ground truth).
2.  **Matching Strategy:** Instead of fixed criteria, the network dynamically assigns labels based on the cost. Common approaches include:
    *   **Optimal Transport (OT):** Formulated as an optimal transport problem (e.g., OTA), where assigning anchors to ground truths is treated as minimizing the global transport cost.
    *   **SimOTA:** An efficient approximation of OT that simplifies the assignment by calculating a cost matrix and selecting the top-$k$ samples with the lowest cost for each ground truth.
    *   **Task-Aligned Assignment:** Evaluates the alignment between classification and regression heads (e.g., TLA in TOOD), where the labels are assigned based on a task-aligned score that balances localization precision and classification probability.
3.  **Adaptive Thresholding:** By making the assignment dependent on the model's current output (the cost matrix), the training process focuses on high-quality predictions rather than enforcing static geometric constraints, which is particularly effective in TAL for handling varying action durations and complex boundaries.

This paradigm effectively bridges the gap between anchor-based and anchor-free methods by allowing the assignment strategy to evolve alongside the model's weights during training.

---

### S13 (simple) -- How many games does SoccerNet Action Spotting contain, and how many annotated actions are in SoccerNet-v2?
**Required facts:** RAG 1/3 | closed book 1/3

**With retrieval:**

SoccerNet Action Spotting (v1) contains 500 full broadcast football matches. SoccerNet Action Spotting (v2) contains 110,458 annotated actions.

**From memory only:**

The **SoccerNet** dataset, as originally released, contains **500 full matches** (1,000 halves). 

**SoccerNet-v2**, which expanded the action spotting task to cover 17 classes, contains **664,547 annotated action instances** across those 500 matches.

---

### S14 (simple) -- What is the difference between tight and loose average-mAP in SoccerNet?
**Required facts:** RAG 3/3 | closed book 3/3

**With retrieval:**

In the context of the SoccerNet challenge, the average-mAP is calculated using different temporal tolerances to define a successful event detection. "Tight" settings refer to tolerances ranging from 1 to 5 seconds, while "loose" settings refer to a broader range of 5 to 60 seconds. 

These metrics are used to evaluate performance in action spotting, with performance often varying significantly between these two regimes depending on the model architecture. A key limitation noted in the literature is that for tolerances ($\delta$) greater than the temporal window utilized by a model (e.g., an 8-second window), the resulting average-mAP values can become misleading. Furthermore, the overall interpretation of these scores is complicated by inconsistent evaluation toolkit handling of classes without ground-truth events, where spurious predictions may or may not be penalized depending on the implementation.

**From memory only:**

In the context of the SoccerNet benchmarks for action spotting, the difference between "tight" and "loose" average-mAP (Average Precision) relates to the temporal tolerance threshold used to evaluate whether a predicted event is a true positive.

*   **Tight Average-mAP:** Uses a stricter temporal tolerance window—typically $\pm 5$ seconds around the ground truth timestamp. A prediction must fall within this narrow interval to be considered a correct detection. This evaluates the model's ability to precisely localize the exact moment an event occurs.
*   **Loose Average-mAP:** Uses a more permissive temporal tolerance window—typically $\pm 30$ seconds. This accounts for the inherent ambiguity in labeling sports events (e.g., when a "shot" officially begins vs. ends) and rewards models that identify the general temporal vicinity of the action.

The **Average-mAP** metric itself is calculated by integrating the Precision-Recall curve across a range of thresholds (usually $[5, 10, 15, 20, 25, 30]$ seconds), but the distinction is often highlighted to compare performance sensitivity to temporal jitter. Tight evaluation is significantly more challenging and sensitive to the latency between the action onset and the annotation.

---

### S15 (simple) -- What is the difference between Temporal Action Localization, Action Spotting, and Precise Event Spotting?
**Required facts:** RAG 2/3 | closed book 3/3

**With retrieval:**

The primary differences between Temporal Action Localization (TAL), Action Spotting (AS), and Precise Event Spotting (PES) involve their output representations, temporal tolerances, and intended use cases:

*   **Temporal Action Localization (TAL):** Represents actions as temporal intervals defined by start and end times. It is best suited for long-duration or continuous actions and typically utilizes a tolerance window of approximately 1–5 seconds. It is associated with high annotation costs.
*   **Action Spotting (AS):** Represents actions using single keyframes rather than intervals. It is designed for fast-paced or ambiguous actions and uses a tolerance window of 5–60 frames. This method offers lower annotation costs compared to TAL and is frequently used for sports highlight detection.
*   **Precise Event Spotting (PES):** Also represents actions as single keyframes but requires much higher temporal precision. It distinguishes events from general actions and uses a very tight tolerance window of 0–2 frames. It is intended for fine-grained, critical event detection.

**From memory only:**

In the context of sports video analysis, these terms are often used interchangeably, but they occupy distinct positions regarding temporal resolution and objective:

**1. Temporal Action Localization (TAL)**
*   **Objective:** Detecting both the semantic category and the temporal boundaries (start and end timestamps) of action segments.
*   **Granularity:** Focuses on entire duration intervals $[t_{start}, t_{end}]$. It is typically treated as a regression task on segment proposals or a frame-level classification task.
*   **Context:** Requires determining the presence and extent of an action, such as a "long jump" or "dribbling sequence."

**2. Action Spotting**
*   **Objective:** Detecting the occurrence of specific, often instantaneous or short-duration, actions within a video. 
*   **Granularity:** Usually treats the event as a single point in time or a very short temporal window. Unlike TAL, the exact "start" and "end" are often less critical than the occurrence itself.
*   **Context:** Frequently used in sports for rare events like "goal," "red card," or "substitution." Models are often optimized for high recall and temporal precision near the event, often using architectures like NetVLAD or Transformers for long-range dependency modeling.

**3. Precise Event Spotting**
*   **Objective:** A high-precision variant of action spotting that emphasizes exact temporal alignment (often frame-perfect accuracy).
*   **Granularity:** Requires identifying the specific timestamp where an event transitions from state A to state B (e.g., the exact frame of contact in a ball-strike or the release frame in a jump shot).
*   **Context:** Essential for applications requiring biomechanical analysis or automated refereeing. The error metric is typically defined by strict tolerances (e.g., within $\pm 0.1s$ or $\pm 3$ frames), making it significantly more sensitive to temporal jitter and sampling rate than standard TAL.

**Summary of Differences:**
*   **TAL:** Focuses on **duration** (intervals).
*   **Action Spotting:** Focuses on **occurrence** (detection of rare events).
*   **Precise Event Spotting:** Focuses on **temporal synchronization** (frame-accurate timestamps).

---

### S16 (simple) -- What criticisms does the survey make of existing benchmark datasets and evaluation protocols?
**Required facts:** RAG 1/2 | closed book 1/2

**With retrieval:**

The survey identifies a significant gap between academic benchmarks and real-world deployment. Existing datasets are primarily curated from professional broadcasts featuring high-quality cameras, stable lighting, and fixed angles. Consequently, these datasets fail to represent the challenges of amateur or semi-professional environments, where footage is often captured via handheld or single-angle setups under suboptimal conditions, hindering model generalization.

Regarding evaluation protocols, the survey notes that current benchmarks often utilize low confidence thresholds (e.g., 0.1), which permit multiple class predictions per frame. This is criticized as being inconsistent with the realities of sports video event-spotting, where a single frame rarely contains more than one event.

**From memory only:**

In the context of recent surveys on sports video event-spotting and temporal action localization (TAL), the criticisms of current benchmarks and protocols typically center on the following technical deficiencies:

### 1. Lack of Temporal Precision and Boundary Ambiguity
Existing datasets frequently suffer from poorly defined action boundaries. Many sports events are continuous or highly nuanced (e.g., the exact moment of a "kick" or "contact"). The survey highlights that annotator subjectivity leads to inconsistent ground truth, making standard metrics like Intersection over Union (IoU) unreliable for evaluating precise temporal localization.

### 2. Extreme Class Imbalance
Sports datasets are inherently long-tailed. Significant action segments (e.g., a "goal") are extremely sparse compared to the background (e.g., "play" or "idle"). Current evaluation protocols often fail to account for this sparsity, where models achieve high accuracy by simply predicting the majority class (background), rendering standard mean Average Precision (mAP) misleading without class-agnostic normalization.

### 3. Domain Gap and Lack of Generalization
Benchmarks are often restricted to specific sports or controlled environments (broadcast footage vs. egocentric/fixed-camera). The survey notes that evaluation protocols rarely test cross-domain robustness or "in-the-wild" performance. Models overfit to the specific camera angles, production styles, or officiating norms of the training dataset, failing to generalize to novel sports or heterogeneous camera configurations.

### 4. Over-reliance on "Shortcut" Cues
Evaluation protocols often do not penalize models that rely on "shortcut" features rather than temporal kinematics. For example, models may identify a sports event based on a static scoreboard overlay or the presence of a specific referee uniform rather than the action itself. Current protocols lack diagnostic procedures (e.g., causal intervention or feature masking) to ensure the model is learning the intended temporal semantics.

### 5. Inadequate Metric Sensitivity
The survey criticizes the reliance on mAP as a monolithic metric. It argues that mAP conflates classification performance with localization accuracy. Researchers suggest that current protocols should decouple these aspects, utilizing metrics that penalize temporal jittering or delay (e.g., using stricter IoU thresholds or dedicated temporal alignment metrics like Soft-DTW) to better reflect the utility of the system for automated officiating or highlight generation.

### 6. Limited Scope of Multi-Modal Integration
Benchmarks generally lack synchronized multi-modal inputs (e.g., telemetry, audio, or broadcast commentary). The survey argues that current evaluation protocols do not incentivize the fusion of these modalities, even though they are essential for state-of-the-art event spotting, thus limiting the community's progress toward truly robust sports intelligence systems.

---

### S17 (simple) -- How is audio processed in the multimodal soccer event detection paper?
**Required facts:** RAG 1/2 | closed book 1/2

**With retrieval:**

In the multimodal system described, audio is processed using a Log-Mel spectrogram-based model. Researchers tested this model using different audio sample windows to evaluate its performance in event detection and classification. 

The findings indicate that utilizing the audio model in isolation results in poor performance compared to visual models. However, when integrated with visual models, the multimodal approach generally improves performance, particularly for events like goals. For other event types such as cards and substitutions, the impact of the audio stream is variable; while causal audio analysis can enhance performance, adding audio information can sometimes be detrimental depending on the specific tolerances configured.

**From memory only:**

In multimodal soccer event detection papers (such as those utilizing the SoccerNet dataset or similar architectures), audio is typically processed as follows:

1.  **Feature Extraction:** Raw audio waveforms are converted into **Log-Mel Spectrograms** or processed via pre-trained audio models. Common choices include **VGGish** or **PANNs** (Pre-trained Audio Neural Networks) to extract high-level semantic embeddings.
2.  **Temporal Resolution:** To align with video frames (typically 1–2 fps), the audio signal is divided into segments corresponding to the temporal windows of the video. 
3.  **Cross-Modal Integration:** The audio features are projected into a joint latent space with visual features. This is often achieved through:
    *   **Feature Concatenation:** Feeding the fused vectors into a temporal model (e.g., Bi-LSTM or Transformer).
    *   **Cross-Attention Mechanisms:** Using visual queries to attend to audio features, or vice versa, to capture the synchronization between visual actions (e.g., a ball entering a net) and acoustic cues (e.g., crowd roar, whistle).
4.  **Specific Architectural Handling:** In models like **CALF** (Cross-modal Action Localization Framework), audio and video branches are processed in parallel via separate CNN/Transformer encoders before being fused via late-fusion or cross-modal attention blocks to refine the detection of sparse events like goals or cards.

---

### S18 (simple) -- Does adding audio always improve soccer event detection?
**Required facts:** RAG 2/2 | closed book 0/2

**With retrieval:**

No, adding audio does not always improve soccer event detection. While combining audio and visual models generally improves performance, particularly for goal detection, the impact of audio information is inconsistent for other event types. For events such as cards and substitutions, the performance gain depends on the defined tolerances, and in some cases, adding audio information can actually be detrimental to detection accuracy.

**From memory only:**

No, it does not. While audio provides strong discriminative cues for events like goals (crowd roar) or whistle blows (referee signals), its utility is conditional and often task-dependent:

*   **Signal-to-Noise Ratio:** In high-noise environments or games with low crowd engagement, audio features can introduce noise rather than signal, potentially degrading the performance of purely visual models.
*   **Temporal Misalignment:** Audio-visual temporal misalignment occurs because audio events (e.g., impact sounds, whistles) may precede or succeed the peak visual event (e.g., ball-foot contact, player tackle). If the fusion mechanism does not explicitly account for this temporal jitter, performance can plateau or decline.
*   **Redundancy and Modality Dominance:** In scenarios where visual motion is highly informative (e.g., clear tactical changes), audio provides redundant information that fails to offer additional gain. If a model is not trained to properly gate the modality contribution (e.g., via late fusion or attention mechanisms), the audio branch may dominate the decision-making process at the expense of precise visual localization.
*   **Domain Sensitivity:** Performance gains are heavily reliant on the quality of the audio-visual synchronization in the dataset. In uncurated, long-form sports broadcast data, background music or commentary can act as a distractor, often leading to false positives if the model is not robustly trained against non-event audio.

In conclusion, while audio-visual fusion is generally superior to uni-modal visual models for event spotting, adding audio only improves performance if the architecture effectively handles modality weighting and cross-modal temporal alignment.

---

### P01 (paraphrase) -- Which model here is small enough to train on a single graphics card?
**Required facts:** RAG 2/2 | closed book 0/2

**With retrieval:**

Based on the provided documents, several models can be trained on a single graphics card:

*   **E2E-Spot:** Both the 200MF and 800MF variants are trained on a single A5000 GPU.
*   **MFS:** Experiments for this model are conducted on a single NVIDIA L40S GPU.

While other models are discussed, the provided text does not explicitly state that they are restricted to or specifically trained on a single GPU (for example, the classification models mentioned are trained on an Nvidia DGX-2 containing 16 GPUs).

**From memory only:**

Most modern state-of-the-art temporal action localization (TAL) models—such as **ActionFormer**, **RTD-Net**, or **TriDet**—can be trained on a single high-end consumer or prosumer GPU (e.g., NVIDIA RTX 3090/4090 or A6000 with 24GB+ VRAM), provided you adjust the following parameters:

1.  **Batch Size:** Reducing the batch size (often to 1 or 2) is the primary method to fit these models on single-card setups. You may need to use gradient accumulation to maintain effective batch sizes for stable optimization.
2.  **Input Resolution/Frame Sampling:** Many TAL models ingest fixed-length feature sequences (e.g., extracted via I3D or SlowFast). If you are performing end-to-end training including the backbone, you must downsample the temporal resolution or input frame size to stay within memory constraints.
3.  **Backbone Choice:** Using a lightweight backbone (e.g., a frozen MobileNet or a truncated ResNet) rather than a full-scale ViT or VideoMAE backbone significantly lowers memory overhead.

**Models particularly well-suited for single-GPU environments:**
*   **Boundary-Matching Networks (BMN):** Lightweight and highly efficient; standard in many benchmarks for low-compute scenarios.
*   **G-TAD:** Graph-based temporal action detection; has a relatively small parameter footprint.
*   **SSAD (Single-Stream Temporal Action Detection):** Designed as a single-shot detector, making it memory-efficient compared to two-stage proposal-based architectures.

If you are training on datasets like THUMOS14 or ActivityNet, ensure you are utilizing **feature-level training** (loading pre-extracted features) rather than pixel-level end-to-end training, as the latter is rarely feasible on a single card for modern video transformers.

---

### P02 (paraphrase) -- How do these systems handle the fact that some events happen far more often than others?
**Required facts:** RAG 1/2 | closed book 1/2

**With retrieval:**

The provided text identifies the class imbalance in sports event datasets—where frequent events like "Ball out of play" significantly outnumber rare ones like "Red card"—as a primary cause for the poor performance of existing event-spotting methods on underrepresented classes. 

While the text highlights this challenge, it does not explicitly detail a specific mechanism used to mitigate this imbalance, other than noting that researchers have attempted to address the issue. For the purpose of data preparation, the provided context mentions creating a "background class" by sampling segments between events that are at least 180 seconds apart, though this appears to be a strategy for handling the temporal distribution of data rather than a direct solution to class frequency imbalance.

**From memory only:**

In sports video event-spotting and temporal action localization, the inherent class imbalance (e.g., "goal" vs. "background" or "pass") is primarily addressed through the following technical strategies:

1.  **Loss Function Modification**: 
    *   **Class-Balanced Cross-Entropy**: Weighting the loss function inversely proportional to class frequency to penalize the model more for misclassifying underrepresented minority classes.
    *   **Focal Loss**: Dynamically scales the cross-entropy loss by a factor $(1 - p_t)^\gamma$, down-weighting the loss contribution from "easy" background examples and forcing the model to focus on "hard" positive samples.

2.  **Sampling Strategies**:
    *   **Hard Negative Mining**: During training, explicitly selecting and prioritizing difficult background segments (false positives) to refine the decision boundary.
    *   **Oversampling/Augmentation**: Increasing the effective frequency of minority classes through temporal data augmentation (e.g., jittering, speed perturbation) or synthetic data generation.

3.  **Architectural Prior/Constraints**:
    *   **Non-Maximum Suppression (NMS) Thresholding**: Tuning temporal IoU thresholds to suppress redundant detections in dense, high-frequency action scenarios.
    *   **Two-Stage Hierarchical Approaches**: Using a coarse segment proposal network to filter background frames, followed by a fine-grained classification head designed to handle the remaining balanced distribution.

4.  **Metric Selection**: Researchers typically move beyond raw accuracy, utilizing **mean Average Precision (mAP)** at various temporal IoU thresholds or **Area Under the ROC Curve (AUC)** to ensure performance is evaluated across the entire precision-recall trade-off rather than being biased by the majority class.

---

### P03 (paraphrase) -- What happens when the people labelling the videos put the timestamp slightly in the wrong place?
**Required facts:** RAG 2/2 | closed book 1/2

**With retrieval:**

Temporal misalignment in ground-truth labels—where the annotated timestamp does not perfectly align with the actual event—is a common challenge in action spotting. To address this, current research employs a dynamic label assignment strategy. 

This approach, inspired by minimum-cost matching in spatial object detection, calculates matching costs based on both predicted action class scores and temporal offsets. By allowing predictions to maintain temporal offsets from the ground-truth times during training, the model can dynamically assign labels to the most likely predictions even when they deviate from the annotated times. This effectively mitigates the negative impact of inconsistent or misaligned labels, particularly in scenarios where events are visually distinct but subject to annotation error.

**From memory only:**

In temporal action localization (TAL) and event-spotting, temporal annotation jitter—the misalignment between the ground truth boundaries/timestamps and the actual onset/offset of an action—acts as a source of label noise that degrades model performance in several ways:

1.  **Objective Function Mismatch:** Most TAL models rely on temporal Intersection-over-Union (tIoU) or binary classification losses (e.g., cross-entropy). If the labels are jittered, the model is penalized for predicting the "correct" visual segment, effectively forcing it to learn a biased representation that shifts the action boundaries to match the noisy annotations.
2.  **Boundary Ambiguity:** In regression-based frameworks (like BMN or ActionFormer), jittered labels introduce variance in the regression head. This leads to unstable boundary predictions, especially for short-duration actions where a small temporal offset represents a large percentage of the total action duration.
3.  **Positive Sample Corruption:** In frame-level classification, jittered labels cause "background" frames to be mislabeled as "foreground" (and vice versa). This introduces label noise into the optimization process, often leading to over-smoothing of activation scores and poor localization precision (mAP at high IoU thresholds).
4.  **Inconsistent Evaluation:** Since standard evaluation metrics (mAP at IoU thresholds of 0.5, 0.75, etc.) are highly sensitive to boundary accuracy, even minor annotation jitter can lead to significantly deflated performance metrics, failing to reflect the model's actual ability to identify the action.

**Mitigation strategies include:**
*   **Soft-labeling/Gaussian smoothing:** Replacing hard temporal boundaries with Gaussian distributions to allow for marginal temporal tolerance.
*   **Boundary refinement modules:** Using iterative refinement or offset regression to "learn" the correction for noisy annotations during training.
*   **Robust loss functions:** Employing losses like PolyLoss or Focal Loss to reduce the influence of ambiguous or mislabeled transition frames.

---

### P04 (paraphrase) -- Is there any work here that uses sound as well as pictures?
**Required facts:** RAG 2/2 | closed book 2/2

**With retrieval:**

Yes, several works in the provided context incorporate both audio and visual modalities for event detection in sports video.

*   **Multimodal Fusion Studies:** Research by Etsen et al. investigated early, late, and hybrid fusion strategies for combining audio spectrogram features with visual embeddings, finding that late fusion generally performs best on the SoccerNet dataset.
*   **Transformer-based Architectures:** Xarles et al. introduced ASTRA, a Transformer-based encoder–decoder that utilizes learnable cross-modal queries and multi-head attention to jointly process audio and visual embeddings.
*   **Log-Mel Spectrogram Analysis:** Other experiments augment state-of-the-art visual models (such as those by Cioppa et al. and Rongved et al.) with a Log-Mel spectrogram-based audio model analyzed via a 2D-ResNet.

These studies consistently indicate that while audio-only models typically underperform compared to visual models, combining the two modalities generally improves detection accuracy, particularly for events like goals where the audio signal is highly predictable. However, the effectiveness of this multimodal approach is context-dependent; for certain events like cards or substitutions, the gain is more variable and can sometimes be detrimental to performance.

**From memory only:**

Yes. While the dominant paradigm in temporal action localization (TAL) and event-spotting remains unimodal (RGB-based), recent research has integrated audio to resolve ambiguities where visual cues are sparse (e.g., crowd reactions or whistle detection).

Key approaches include:

*   **Multimodal Fusion architectures:** Works like **AV-TAD** (Audio-Visual Temporal Action Detection) leverage architectures like SlowFast or I3D for video and VGGish/YAMNet for audio. Fusion typically occurs via late-stage concatenation or cross-modal attention mechanisms (e.g., Transformers with cross-attention layers).
*   **Audio-Visual Event Localization:** Papers such as *“Listen to Look: Action Recognition by Previewing Audio”* demonstrate that audio often provides earlier temporal cues for events (e.g., the sound of a ball strike before the visual frame change).
*   **Sports-Specific Datasets:** Some recent benchmarks for sports event spotting have begun including audio streams to assist in identifying discrete events like "referee whistles" or "bat-ball contact," which are often clearer in the audio signal than in the low-frame-rate visual stream.
*   **Contrastive Learning:** Research using **Audio-Visual Contrastive Learning** (e.g., applying principles from CLIP or AV-based self-supervised learning) aims to learn joint representations where audio and visual features are synchronized, effectively allowing the model to perform "audio-supervised" event spotting.

If you are looking for a specific paper or architecture implementation (e.g., late fusion vs. token-level fusion), please specify the sport or event type.

---

### P05 (paraphrase) -- What can you do if you do not have enough labelled training footage?
**Required facts:** RAG 1/2 | closed book 2/2

**With retrieval:**

If you lack sufficient labeled training footage, you can utilize the following strategies:

*   **Active Learning:** This approach reduces annotation requirements by using an iterative pipeline. A model is trained on a small labeled subset and then used to predict on unlabeled videos. Samples are then ranked by uncertainty (using entropy or confidence heuristics), and the most informative clips are selected for manual annotation and added back to the training pool. This method can achieve competitive results with significantly less labeled data than fully supervised baselines.
*   **Learning with Weak or Few Labels:** Research indicates that learning to spot events using few or weak labels can help mitigate data scarcity. This is considered a potential area for future work to accelerate the curation of new datasets.
*   **Dynamic Label Assignment:** Improving training consistency through dynamic label assignment strategies can help optimize the use of available data. By adjusting temporal weightings (lambda-time), models can more effectively select visually consistent event frames during training, which can improve performance when dealing with temporal misalignment in ground-truth labels.

Be aware that if your approach relies on transformer-based architectures, performance may be limited when datasets are small (e.g., fewer than ten videos), as these models require a substantial number of samples to train effectively. Additionally, if the data exhibits significant domain-shift or limited availability, end-to-end learning alone may be insufficient.

**From memory only:**

If you lack sufficient annotated training footage for temporal action localization (TAL) or event-spotting, you can employ the following strategies:

1.  **Weakly Supervised Learning (WSL):** Use video-level labels (e.g., presence/absence of an event in a trim) rather than frame-level temporal boundaries. Techniques like Multiple Instance Learning (MIL) are typically used to rank video segments by their probability of containing the event, often constrained by sparsity and smoothness priors to delineate boundaries.
2.  **Self-Supervised/Unsupervised Pre-training:** Leverage large-scale unlabeled video datasets to learn robust spatio-temporal representations. Models pre-trained on tasks like masked autoencoding (MAE-ST), temporal order verification, or contrastive learning (e.g., SimCLR/MoCo applied to video) provide strong feature extractors that reduce the need for extensive task-specific labels.
3.  **Cross-Modal/Multi-Modal Distillation:** Utilize audio-visual correspondence or text-video retrieval models (e.g., CLIP-based variants like VideoCLIP or ActionCLIP). These models are pre-trained on massive web-scraped data and can act as zero-shot or few-shot detectors by aligning semantic text prompts with specific temporal windows.
4.  **Data Augmentation and Synthetic Data:**
    *   **Temporal Augmentation:** Apply jittering, speed perturbation, or temporal cropping to increase variance.
    *   **Mixup/CutMix:** Interpolate between feature vectors of different samples to encourage better decision boundary robustness.
    *   **Synthetic Data:** Use game engines (e.g., Unreal/Unity) to render high-fidelity sports scenarios with automated, pixel-perfect ground-truth generation.
5.  **Transfer Learning & Domain Adaptation:** Fine-tune models pre-trained on large-scale datasets like Kinetics or ActivityNet. If the source and target sports domains differ, utilize Unsupervised Domain Adaptation (UDA) methods (e.g., adversarial domain alignment) to bridge the feature distribution gap.
6.  **Active Learning:** Implement an iterative pipeline where the model identifies the most informative, ambiguous, or high-uncertainty samples for human annotators, maximizing the impact of every manual annotation performed.
7.  **Few-Shot/Meta-Learning:** Utilize episodic training frameworks (e.g., Prototypical Networks or MAML) designed to generalize to new, unseen action classes given only a handful of labeled examples (support set).

---

### P06 (paraphrase) -- How do researchers score whether a spotting system got the moment right?
**Required facts:** RAG 1/2 | closed book 2/2

**With retrieval:**

Researchers distinguish between two primary scoring criteria for spotting systems based on the required precision:

*   **Precise Spotting:** A detection is considered correct if it falls within a very narrow tolerance, such as within 1 frame of the human-annotated ground truth.
*   **Coarse Action Spotting:** A detection is deemed correct if it occurs within a looser time window around the true event, typically ranging from 1–5 seconds or 5–60 seconds (equating to 10–100s of frames). 

In contrast, other related tasks like Temporal Action Detection (TAD) and Temporal Action Segmentation (TAS) use interval-based metrics, such as temporal Intersection-over-Union (IoU) or F1-scores based on temporal overlap, which do not enforce frame-level accuracy on action boundaries.

**From memory only:**

In sports video event-spotting and temporal action localization, researchers primarily evaluate system performance using **Temporal Intersection over Union (tIoU)** and **Mean Average Precision (mAP)**.

The standard procedure is as follows:

1.  **Temporal IoU Criteria:** A predicted interval $[s_p, e_p]$ is considered a True Positive (TP) if it overlaps with a ground truth interval $[s_g, e_g]$ such that:
    $$\text{tIoU} = \frac{\text{intersection}(p, g)}{\text{union}(p, g)} \ge \alpha$$
    where $\alpha$ is typically a threshold value (common thresholds range from 0.1 to 0.5 for coarse localization, up to 0.7 or 0.9 for fine-grained accuracy).

2.  **Point-based Spotting (Tolerance windows):** For datasets where events are defined as instantaneous points rather than intervals, researchers use a **tolerance window ($\pm \delta$ seconds)**. A predicted timestamp $t_p$ is a TP if $|t_p - t_g| \le \delta$.

3.  **Evaluation Metrics:**
    *   **mAP (Mean Average Precision):** The primary metric, calculated by integrating the Precision-Recall curve across different tIoU thresholds. 
    *   **Recall vs. False Positives per Video (FPPV):** Often used in spotting tasks where the goal is to identify specific event occurrences (e.g., goals, fouls) rather than segmenting the entire duration. This is visualized via **Free-response Receiver Operating Characteristic (FROC)** curves.

4.  **Special Considerations:**
    *   **Action Completion:** Some tasks require the system to distinguish between the initiation, peak, and completion of an action, necessitating evaluation based on temporal order or specific frame-level alignment.
    *   **Duplicate Detections:** To prevent over-counting, non-maximum suppression (NMS) is applied; if multiple predictions overlap the same ground truth, only the highest confidence prediction is counted as a TP, while others are counted as False Positives.

---

### P07 (paraphrase) -- Which method borrows an idea from object detection and applies it to time instead of space?
**Required facts:** RAG 2/3 | closed book 2/3

**With retrieval:**

The method inspired by Faster R-CNN, as described in the provided text, applies the object detection architecture to the temporal domain to create an end-to-end model for temporal action detection that generates temporal proposals.

**From memory only:**

The most prominent method that translates object detection principles to the temporal domain is **R-C3D (Region Convolutional 3D Network)**. 

It adapts the **Faster R-CNN** architecture by replacing 2D spatial region proposals with 1D temporal proposals. It utilizes a 3D ConvNet as a feature extractor, followed by a Temporal Proposal Network (TPN) to generate temporal segments, which are then refined and classified using a temporal RoI (Region of Interest) pooling layer—directly mirroring the spatial RoI pooling mechanism used in Faster R-CNN.

---

### P08 (paraphrase) -- Is there a benchmark for a racket sport played indoors on a small table?
**Required facts:** RAG 2/2 | closed book 2/2

**With retrieval:**

Yes, there are several datasets specifically for table tennis, which is an indoor racket sport played on a table:

*   **OpenTTGames:** Contains 12 high-definition table tennis matches recorded at 120 FPS with 4,271 labeled events, including ball bounces, net hits, and empty events annotated at the frame level. It is designed for training models under high-speed gameplay conditions.
*   **P2A:** A large-scale dataset featuring 272 hours of broadcast table tennis videos. It includes 14 fine-grained stroke classes organized into 8 higher-level categories, with professional-validated, frame-level annotations.
*   **TTA (Table Tennis Australia):** A benchmark consisting of 39 full-game, 30 FPS videos. It includes 4,878 precisely annotated events across eight classes (serve, bounce, forehand, and backhand), with specific labels for near and far table sides. It is notable for capturing non-ideal camera angles, frequent occlusions, and real-world recording conditions. TTA has also been expanded to include 12 ball-related classes to support modeling of gameplay flow.

**From memory only:**

Yes. The benchmark is **Table Tennis** (often referred to as TTNet). The primary dataset associated with this domain is the **TTNet-2022** dataset, which is designed for multi-task learning in table tennis, specifically focusing on event spotting (e.g., ball bounces, net hits) and temporal action localization (e.g., stroke types).

---

### C01 (comparison) -- How does T-DEED's approach to temporal resolution differ from E2E-Spot's?
**Required facts:** RAG 2/3 | closed book 2/3

**With retrieval:**

The provided text does not contain information comparing the specific approaches to temporal resolution used by T-DEED and E2E-Spot. While the text notes that T-DEED outperforms state-of-the-art methods on the FineDiving dataset, it does not explicitly describe or contrast their respective methodologies regarding temporal resolution.

**From memory only:**

T-DEED (Temporal Discrete-Continuous Event Detection) and E2E-Spot utilize fundamentally different approaches to temporal resolution:

*   **T-DEED** employs a **multi-scale, hierarchy-based refinement** strategy. It utilizes a discrete-to-continuous framework where an initial coarse proposal generation (often via a 1D CNN backbone) is refined by a dedicated regressor that explicitly predicts the temporal offsets of event boundaries. It maps discrete video segments into continuous time coordinates, allowing for sub-frame or precise boundary localization by treating the refinement as a regression task on the temporal displacement.
*   **E2E-Spot** (End-to-End Spotting) approaches resolution through **anchor-free, direct frame-wise classification** or a **query-based mechanism** (often Transformer-based). Unlike T-DEED’s focus on refining predefined temporal segments, E2E-Spot typically computes action scores directly at the feature map level or through learnable queries that attend to global temporal context. It relies on the inherent resolution of the feature extractor’s stride and the alignment between the input feature sequence and the ground truth labels, often bypassing the explicit "proposal-to-refinement" pipeline found in T-DEED.

In summary, T-DEED explicitly optimizes boundary regression for fine-grained temporal localization, whereas E2E-Spot favors a holistic, direct-mapping approach to identification and localization via spatial-temporal feature transformation.

---

### C02 (comparison) -- Both MFS and Transformer Gate Shift extend the Gate Shift Module. How do they differ?
**Required facts:** RAG 1/3 | closed book 1/3

**With retrieval:**

The provided context does not contain information regarding a "Transformer Gate Shift" model or its comparison to the Multi-scale Feature Shift (MFS) module. Therefore, I cannot describe how they differ.

**From memory only:**

The primary difference lies in the integration mechanism and the scope of information aggregation within the temporal window.

**MFS (Multi-scale Feature Shift)** focuses on hierarchical temporal modeling by applying varying shift scales across different feature channels. It aims to capture multi-granularity motion patterns by partitioning channels into groups, where each group is shifted by a different temporal stride. This mimics a multi-scale receptive field in the temporal dimension without increasing computational overhead.

**Transformer Gate Shift (TGS)** integrates self-attention mechanisms into the shifting process to provide data-dependent, rather than static, temporal aggregation. While standard GSM (Gate Shift Module) performs fixed shifts, TGS utilizes a lightweight Transformer block—typically cross-attention or temporal self-attention—to adaptively reweight or aggregate features before or after the shifting operation. This allows for content-aware temporal dependency modeling that goes beyond the rigid cyclic shifts of the base GSM.

In short: **MFS** extends GSM via **static multi-scale temporal sampling** (parallel shifts), whereas **TGS** extends GSM via **dynamic, attention-based temporal feature modulation**.

---

### C03 (comparison) -- Compare how T-DEED and the Sony paper each try to make frame-level features more distinguishable.
**Required facts:** RAG 1/2 | closed book 1/2

**With retrieval:**

T-DEED and the Sony paper employ different strategies to enhance frame-level feature discriminability:

*   **T-DEED:** Focuses on improving token discriminability through architectural modifications, specifically by utilizing Gate-Shift-Fuse (GSF) modules in its RegNetY60 backbone to produce spatio-temporal representations. It further employs a temporally-discriminant neck architecture featuring Scalable-Granularity Perception (SGP) layers and SGP-Mixer layers, which are designed to increase token discriminability within the same temporal sequence and accommodate distinct temporal scales.
*   **The Sony paper (in the context of Temporal Feature Distillation):** Addresses the issue where standard backbone features (such as DINO) exhibit high inter-frame cosine similarity, leading to "projection-level invariance" that obscures event boundaries. It introduces Temporal Feature Distillation (TFD) to align temporally structured backbone features directly—rather than using projected representations—thereby forcing the features to maintain sharp similarity drops at ground-truth event boundaries while preserving overall feature quality.

**From memory only:**

Both T-DEED and the Sony paper (likely referring to the CVPR 2020 paper by Ullah et al., "Attention-based CNN for Video Action Recognition" or similar event-spotting architectures) address the over-smoothing of frame-level features through temporal modeling and attention mechanisms, though their approaches differ in focus:

**T-DEED (Temporal-DEcomposed Event Detection):**
*   **Decomposition Strategy:** T-DEED employs a feature decomposition strategy that explicitly separates "event-specific" features from "background/contextual" features. 
*   **Mechanism:** It utilizes a dual-stream architecture (often utilizing a temporal attention module) to disentangle the representation. By explicitly modeling the boundary versus the interior of an action, it prevents the temporal diffusion of features, ensuring that frames surrounding a sport event (e.g., a penalty kick) are not incorrectly smoothed into the event feature representation. This enforces discriminability through explicit class-agnostic attention masks.

**Sony (typically referring to the framework using Actionness/Salience or the 2021/2022 temporal refinement works):**
*   **Refinement Strategy:** The Sony-affiliated research often focuses on **Temporal Context Aggregation** via learnable kernels or multi-scale dilation to improve feature discriminability.
*   **Mechanism:** Instead of explicit decomposition, they emphasize the use of **Temporal Context Transformers** or **Temporal-Aware Convolutional layers** that weight features based on their relationship with long-range temporal dependencies. By using a self-attention mechanism, they force the network to attend to the most "action-salient" frames within a window, effectively sharpening the feature responses at the peak of an event and suppressing redundant background features.

**Summary of Difference:**
*   **T-DEED** relies on **structural decomposition** (explicitly separating event vs. background to prevent feature contamination).
*   **Sony** relies on **temporal salience and attention** (dynamically re-weighting feature importance based on global temporal context to highlight event-critical features).

---

### C04 (comparison) -- How computationally efficient is MFS compared with the heavier architectures it is measured against?
**Required facts:** RAG 2/2 | closed book 0/2

**With retrieval:**

MFS demonstrates significant computational efficiency compared to heavier state-of-the-art baselines like ASTRM. Specifically, MFS achieves SOTA-level accuracy—matching ASTRM’s performance on the Tennis dataset (δ = 0)—while requiring only 45.06 GFLOPs compared to ASTRM’s 67.24 GFLOPs. This represents a 33% reduction in FLOPs, or a 1.5× efficiency gain. These results suggest that efficient multi-scale shifting allows for top-tier performance without the need for the complex temporal mechanisms found in heavier architectures.

**From memory only:**

MFS (Multi-scale Feature Fusion) is designed for high computational efficiency by utilizing a lightweight backbone and a temporal feature pyramid network (FPN) architecture that avoids the heavy overhead of recurrent neural networks or dense 3D-CNN temporal sliding windows.

Compared to heavier architectures, such as those employing **I3D, SlowFast, or multi-stream TSN backbones** combined with complex temporal aggregation modules (like **GTAD** or **BSN**), MFS demonstrates significantly lower FLOPs and parameter counts. 

Specifically:
*   **Inference Latency:** MFS typically operates at a fraction of the inference time per video segment because it replaces expensive heavy feature extraction (e.g., dense optical flow streams) with more efficient 2D-based spatio-temporal features or lightweight 3D kernels.
*   **Memory Footprint:** By employing a more streamlined feature fusion mechanism rather than multi-stage, multi-model ensemble architectures, MFS drastically reduces GPU memory consumption during inference.
*   **Throughput:** In benchmarks, MFS often achieves high real-time throughput (often exceeding 100+ FPS on standard hardware), whereas architectures relying on heavy 3D-CNNs for feature extraction or recurrent refinement modules struggle to maintain real-time performance on high-resolution, long-form sports videos.

In summary, MFS prioritizes a "backbone-light" approach, prioritizing competitive temporal localization performance while minimizing the computational bottlenecks typically found in earlier action localization frameworks.

---

### C05 (comparison) -- Do E2E-Spot and T-DEED use the same backbone sizes?
**Required facts:** RAG 3/3 | closed book 0/3

**With retrieval:**

Yes, both E2E-Spot and T-DEED utilize the same RegNet-Y 200MF and 800MF backbone sizes.

**From memory only:**

No. 

**E2E-Spot** typically utilizes a **ResNet-50** backbone (often pre-trained on ImageNet/Kinetics) to extract frame-level features. 

**T-DEED** (Temporal Deep Event Detection) typically employs a **ResNet-18** or **ResNet-50** backbone depending on the specific experiment, but it is frequently paired with a significantly different architectural design—specifically a dual-stream or feature-pyramid network structure—to handle the temporal dependencies of event spotting. 

They are not constrained to the same backbone size; researchers frequently vary these based on the computational budget and the specific feature representation requirements of the temporal localization heads (e.g., whether the backbone is frozen or fine-tuned).

---

### C06 (comparison) -- Which papers here tackle problems with the training labels rather than the model architecture?
**Required facts:** RAG 2/2 | closed book 1/2

**With retrieval:**

The provided text describes a dynamic label assignment method that addresses issues with training labels rather than model architecture. This approach replaces static ground-truth assignment with a method that assigns labels based on predicted class scores and times, which helps mitigate the impact of temporal misalignment in labels. 

Additionally, the context notes that some performance limitations in datasets like FS and FG are attributed to ambiguity in the definitions of event frames, and it mentions an evaluation specifically regarding noisy labels. Aside from these discussions of label assignment strategies, noise, and frame ambiguity, the provided context does not identify specific papers by title or authorship that focus solely on training labels.

**From memory only:**

In the context of sports video event-spotting and temporal action localization, the following papers primarily address issues related to noisy, incomplete, or weak supervision in training labels:

*   **"Weakly Supervised Action Localization with Expectation-Maximization" (Nguyen et al., 2019):** Addresses the lack of precise temporal annotations by treating them as latent variables and using an EM-based approach to iteratively refine action localization.
*   **"CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise" (Lee et al., 2018):** While general-purpose, it is frequently cited in sports-domain research for mitigating label noise in large-scale datasets by learning a similarity metric to prune or reweight noisy labels.
*   **"TSA-Net: Tube-and-Segment-Aware Network for Action Detection with Weakly Labeled Data" (Zhang et al., 2021):** Focuses on the "sparse supervision" problem in sports, specifically how to generate high-quality pseudo-labels from sparse event-spotting timestamps to bridge the gap between action classification and localization.
*   **"Learning from Noisy Labels for Video Action Recognition" (Thulasidasan et al., 2019/similar):** Addresses the common issue of inconsistent timestamping in sports datasets (e.g., human annotator bias) through noise-robust loss functions.
*   **"Action Segmentation with Mixed-Up Labels" (Various works building on Mixup/Manifold Mixup):** Papers exploring data augmentation techniques specifically designed to handle ambiguous boundaries between consecutive sports actions (e.g., transition frames where the label is uncertain).

If you are looking for research on **"label quality"** specifically in major sports datasets (like SoccerNet), works like **"SoccerNet-v2: Localized Spotting and Recognition" (Giancola et al., 2021)** address label scarcity and the long-tail distribution problem through the introduction of specific data splits and metrics that penalize label-related uncertainty.

---

### C07 (comparison) -- How does the survey's definition of Action Spotting compare with how the SoccerNet paper uses the term?
**Required facts:** RAG 2/2 | closed book 2/2

**With retrieval:**

The provided text defines action spotting as the task of identifying and precisely localizing actions in long, untrimmed video streams with a single timestamp. The SoccerNet paper aligns with this definition, formally characterizing an action spotting method as a function that maps a video to a set of predicted actions, where each action is represented by a triplet containing a predicted class, a timestamp, and an optional confidence score. Both the general survey definition and the SoccerNet methodology treat the task as the temporal localization of events within a continuous video stream using discrete timestamps.

**From memory only:**

In the context of the literature (e.g., Cioppa et al., 2021; Giancola et al., 2018), the definitions of **Action Spotting** are largely consistent but differ in their operational granularity and evaluation constraints:

*   **SoccerNet Definition:** The *SoccerNet* benchmarks define Action Spotting as a **point-level detection task**. An action is treated as a discrete timestamp (a single frame) within a long-form broadcast, typically associated with a specific event class (e.g., goal, card, substitution). Evaluation relies on the **Average-mAP (Average mean Average Precision)** metric, which uses a temporal tolerance window ($\pm \delta$ seconds) around the ground-truth timestamp to account for annotation variance.
*   **Survey Definition:** Surveys on Temporal Action Localization (TAL) typically categorize Action Spotting as a specific sub-problem of **Temporal Action Detection (TAD)**. While the survey definition acknowledges the point-level nature used by *SoccerNet*, it often generalizes the term to include **interval-based spotting**. Unlike the rigid "spot-the-timestamp" approach of *SoccerNet*, survey taxonomies frequently distinguish between:
    *   **Atomic Action Spotting:** Identifying the single instant an action occurs (matching *SoccerNet*).
    *   **Segmental Spotting:** Identifying the start and end boundaries of an action (closer to standard TAL/TAD).

**Key Technical Difference:** The *SoccerNet* paper prioritizes the **ranking of event proposals** in continuous, multi-hour untrimmed video, often treating it as a retrieval/detection problem where the "action" is a point rather than a temporal segment. General surveys emphasize that *SoccerNet*-style spotting is a constrained version of the broader TAL pipeline, where the temporal duration of the action is ignored in favor of minimizing the temporal distance between the predicted point and the ground-truth anchor.

---

### M01 (multi_paper) -- Which papers in this collection introduce a new dataset or benchmark, and what are they?
**Required facts:** RAG 2/3 | closed book 1/3

**With retrieval:**

Based on the provided documents, the following datasets or benchmarks are introduced or described:

*   **OpenTTGames**: A table tennis dataset consisting of 12 high-definition matches recorded at 120 FPS with 4,271 labeled events, including ball bounces, net hits, and empty events.
*   **P2A**: A large-scale table tennis dataset containing 2,721 broadcast videos (272 hours) from major tournaments, featuring 14 fine-grained stroke classes grouped into 8 action categories.
*   **TTA**: A table tennis benchmark consisting of 39 para-professional matches, characterized by non-ideal camera angles, frequent occlusions, and less controlled recording conditions. 

The text also references the use of several other datasets for experimentation, including Tennis, FineDiving, Figure Skating (FS), and FineGym (FG), though it does not explicitly state that the provided papers introduced these specific datasets.

**From memory only:**

In the domain of sports video event-spotting and temporal action localization, the following seminal papers introduced influential datasets or benchmarks:

*   **THUMOS Challenge (THUMOS'14):** Introduced the THUMOS-14 dataset, which established the first large-scale benchmark for temporal action localization (TAL) across 20 action categories, significantly impacting the evaluation of action detection models.
*   **ActivityNet (Heilbron et al., 2015):** Introduced the ActivityNet dataset, a large-scale benchmark for human activity understanding, including temporal action localization in untrimmed videos. The associated challenges remain a standard for the field.
*   **FineGym (Shao et al., 2020):** Introduced a hierarchical dataset for fine-grained action understanding, focusing on gymnastics. It is highly technical, requiring the detection of specific movement primitives rather than high-level action classes.
*   **SoccerNet (Giancola et al., 2018):** Introduced a large-scale dataset for action spotting in soccer (e.g., goals, fouls, cards). It shifted the focus toward long-form, broadcast-style video analysis.
*   **Hockey-Action (Cigla et al., 2019):** Introduced a dataset specifically for fine-grained action recognition and temporal localization within ice hockey, designed to handle high-speed, complex interactions.
*   **EPIC-Kitchens (Damen et al., 2018):** While egocentric, it is frequently cited in the broader temporal localization literature for its large-scale benchmark on action segmentation and detection.
*   **SportsMOT (Cui et al., 2022):** Introduced a large-scale multi-object tracking dataset for sports, focusing on temporal association and localization within the context of team sports.

If you are looking for a specific dataset within a sub-discipline (e.g., specific court sports or combat sports) not listed above, please specify, as many niche datasets exist for specialized applications.

---

### M02 (multi_paper) -- What are the different ways these papers capture long-range temporal context?
**Required facts:** RAG 3/3 | closed book 3/3

**With retrieval:**

The provided papers identify two primary paradigms for capturing long-range temporal context in sports event-spotting:

*   **Transformer-based architectures:** These methods are capable of capturing long-range temporal dependencies but are often computationally intensive, frequently exceeding 600 GFLOPs. Additionally, research indicates that Transformers may suffer from a "rank-loss" problem, where token similarity increases across layers, leading to a loss of discriminability.
*   **Lightweight shift-based modules:** These modules (e.g., TSM, GSM, GSF) allow 2D vision backbones to model temporal information efficiently with significantly lower computational costs (40–45 GFLOPs). While standard shift-based modules are often limited to a near-instantaneous receptive field (t±1), newer approaches like MFS extend this by introducing learnable multi-scale temporal shifts, providing access to frames at varying distances (e.g., δ ∈{1, 2, 3}).

Other temporal modeling networks discussed include Recurrent Neural Networks, specifically the Bi-GRU, which is noted for performing favorably against other temporal networks, and SGP layers, which demonstrate high discriminability for output tokens.

**From memory only:**

To capture long-range temporal context in sports video event-spotting and temporal action localization, current research generally employs the following architectural strategies:

*   **Self-Attention and Transformers:** Replacing or augmenting sliding-window convolutions with Transformer encoders. Global self-attention allows the model to compute dependencies between any two time steps regardless of their distance, effectively modeling long-range dependencies across the entire sequence.
*   **Temporal Hierarchical Modeling:** Utilizing multi-scale architectures (e.g., Dilated Convolutions, Feature Pyramids, or Temporal Pyramid Networks) to increase the receptive field. By stacking layers with increasing temporal strides or dilation rates, the network captures coarse-to-fine temporal dynamics.
*   **Recurrent Neural Networks (RNNs/LSTMs/GRUs):** Historically used to model temporal evolution by maintaining a hidden state. While less common in state-of-the-art models due to training inefficiencies and vanishing gradients, bidirectional RNNs (Bi-LSTMs) remain a mechanism for aggregating past and future context.
*   **Segment-based Pooling/Aggregation:** Approaches that divide a long video into segments (e.g., SNANet, TSN) and aggregate local features via temporal pooling (max/average) or learned attention weights, providing a compact representation of the entire temporal span.
*   **Memory-Augmented Networks:** Incorporating external memory banks (e.g., Gated Recurrent Units or learned memory slots) that store key temporal features from earlier frames, allowing the model to "query" relevant past information when processing the current segment.
*   **State-Space Models (SSMs):** Emerging architectures (e.g., Mamba) that offer linear scaling with sequence length. These utilize selective state-space mechanisms to compress long-range historical information into a latent state, circumventing the quadratic complexity of standard self-attention while maintaining long-range dependency capture.
*   **Graph Convolutional Networks (GCNs):** Modeling temporal relationships as a graph where nodes are segments/frames and edges represent temporal transitions or semantic similarity, allowing for non-local information propagation across the temporal graph.

---

### M03 (multi_paper) -- Which papers build on or compare against E2E-Spot?
**Required facts:** RAG 3/3 | closed book 2/3

**With retrieval:**

Based on the provided context, there is no mention of other papers that build on or explicitly compare against E2E-Spot. The text mentions that E2E-Spot is compared against "strong baselines" and prior work, and notes that T-DEED outperforms the state-of-the-art on the FineDiving dataset, but it does not specify which papers constitute those baselines or prior work.

**From memory only:**

**E2E-Spot** (Battash et al., CVPR 2022) focuses on end-to-end temporal action localization using a sparse, anchor-free approach with transformer-based architectures. Papers that build upon or compare against it generally address the challenges of boundary refinement, long-sequence modeling, or cross-domain adaptation in sports.

### Papers that build on or cite/compare against E2E-Spot:

*   **ASFormer (Temporal Action Localization via Adaptive Segmental Transformers):** Often used as a baseline for comparison. While ASFormer focuses on multi-scale feature hierarchies, subsequent research evaluating E2E-Spot often benchmarks against it to contrast anchor-free sparse detection versus dense segmental regression.
*   **ActionFormer (Zhang et al., ECCV 2022):** A primary contemporary architecture in the field. Many recent works (e.g., **VSLNet** variants or **TriDet**) compare against both E2E-Spot and ActionFormer to measure the performance trade-offs between end-to-end training and feature-based two-stage pipelines.
*   **SoccerNet Challenge-related works:** Because E2E-Spot was evaluated on the SoccerNet dataset, papers participating in the annual **SoccerNet Challenges** (2022–2024) frequently use it as a standard baseline. Specifically:
    *   **CALF (Context-Aware Localization Framework):** Often compared against E2E-Spot in the context of action spotting where temporal context is crucial.
    *   **Various Transformer-based backbones (e.g., TimeSformer/VideoMAE adapters):** Research that replaces E2E-Spot’s feature extraction backbone (typically I3D or SlowFast) with self-supervised video transformers often benchmarks against the original E2E-Spot paper to justify the feature representation gain.

### Methodological Context
Papers comparing against E2E-Spot typically categorize it as an **"End-to-End Dense/Sparse Detector."** They often test against E2E-Spot to highlight:
1.  **Computational efficiency:** E2E-Spot’s advantage in removing multi-stage training pipelines.
2.  **Boundary precision:** Critiquing its performance on high-precision temporal boundaries versus newer regression-based models like **TriDet** or **RTD-Net**.

**Note:** If you are looking for specific recent citations (2023–2024), please note that most state-of-the-art papers in this domain now prioritize **VideoMAE V2** or **InternVideo** backbones, using E2E-Spot primarily as a legacy "anchor-free" architecture baseline. I do not have a exhaustive list of every paper that has ever performed a comparison, as this is a highly active area of publication.

---

### M04 (multi_paper) -- What evaluation metric is used across most of these papers?
**Required facts:** RAG 1/1 | closed book 1/1

**With retrieval:**

The evaluation metrics vary based on the specific task:

*   **Temporal Action Localization (TAL):** The standard metric is mean Average Precision computed with temporal Intersection over Union thresholds (mAP@T-IoU). In this context, predictions are classified as true positives if their temporal IoU with the ground truth exceeds a specified threshold.
*   **Action Spotting (AS) and Play Event Spotting (PES):** The primary evaluation metric is mAP@δ. This is computed using a temporal tolerance window (δ) around the ground-truth timestamp—typically 5–60 seconds for AS and 0–2 frames for PES. 

Additionally, for proposal-based tasks, the Area Under the Curve (AUC) is used to measure average recall across varying proposal counts.

**From memory only:**

The standard evaluation metric across temporal action localization (TAL) and event-spotting literature is **mean Average Precision (mAP)**, typically calculated at multiple **Intersection over Union (IoU) thresholds** (e.g., $\{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7\}$).

For event-spotting specifically, where actions are often defined as temporal points rather than intervals, the **Average Precision (AP)** is frequently supplemented by **Recall at top-k detections** or the **Average-mAP** computed over a range of IoU thresholds (common in benchmarks like HACS or ActivityNet). In fine-grained sports datasets (e.g., SoccerNet), **Average Precision (AP)** is standard, often reported alongside a **relaxed temporal tolerance** (e.g., $\pm 5$ seconds) to account for human labeling ambiguity.

---

### M05 (multi_paper) -- Which sports are covered across this collection of papers?
**Required facts:** RAG 2/3 | closed book 3/3

**With retrieval:**

Based on the provided documents, the sports covered in the datasets and research include:

*   **Soccer:** Represented by multiple datasets such as SoccerNet, SoccerNet-v2, SoccerNet Ball AS, SSET, SoccerDB, and TACDEC.
*   **Tennis:** Represented by datasets like Tenniset and the Tennis dataset (focused on court-specific ball contacts).
*   **Table Tennis:** Represented by datasets including OpenTTGames, P2A, and TTA.
*   **Basketball:** Represented by the NCAA dataset.
*   **General/Multi-sport:** The DeepSportradar-v2 dataset is explicitly identified as a multi-sport dataset, and the collection also references invasion games generally.

**From memory only:**

Across the literature on sports video event-spotting and temporal action localization, the coverage is heavily skewed toward high-action, broadcast-style team sports. The most frequently featured sports include:

*   **Soccer (Football):** Dominates most datasets (e.g., SoccerNet, various localized variants). Tasks typically involve goal detection, penalty kicks, cards, and tactical event spotting.
*   **Basketball:** Common in older action recognition datasets and specialized temporal localization benchmarks focusing on shots, dribbling, and foul detection.
*   **Volleyball:** Frequently used for multi-person action recognition and fine-grained temporal labeling (e.g., Volleyball dataset).
*   **Tennis:** Often studied for stroke classification and point-level temporal segmentation.
*   **Figure Skating:** Used for segmenting specific sub-actions (jumps, spins) within a continuous performance.
*   **Fencing:** Primarily used for fine-grained action recognition and temporal alignment in competitive settings.
*   **American Football:** Represented in large-scale broadcast analysis research, focusing on play segmentation.
*   **Olympic/Multi-sport:** Some benchmarks, such as THUMOS, incorporate a heterogeneous mix including diving, swimming, gymnastics, and weightlifting.

The primary datasets driving this research are **SoccerNet** (soccer), **THUMOS'14** (multi-sport), **ActivityNet** (general/sports), **FineGym** (gymnastics), and **Sports-1M** (large-scale classification).

---

### M06 (multi_paper) -- What is the Gate Shift Module, and which papers here use or extend it?
**Required facts:** RAG 2/3 | closed book 1/3

**With retrieval:**

The Gate Shift Module (GSM) is a component designed for video action recognition that extracts per-frame features by incorporating short-term temporal information. It functions as a mechanism for feature-wise control, enabling the exchange of temporal information across frames.

The papers provided that reference or utilize this module are:

*   **Gate-shift networks for video action recognition** (Sudhakaran et al., 2020), which introduces the module.
*   **Gate-shift-fuse for video action recognition** (Sudhakaran et al., 2023), which extends the original work.
*   The provided text also notes that the GSM is used as a backbone component in other models for extracting features to be subsequently processed by long-term temporal layers, such as Gated Recurrent Units (GRU). 

Additionally, the context describes TGS (Temporal Gating Shift) as a refined approach that improves upon coarse gating by providing full element-wise gating for finer control over temporal information flow. MFS (Multi-scale Feature Shift) is another related mechanism mentioned that applies gated temporal shifts at multiple distances using 3D convolutional gating layers.

**From memory only:**

The **Gate Shift Module (GSM)** is a lightweight architectural component designed to facilitate temporal modeling in deep neural networks (specifically 2D CNNs) by shifting a portion of feature channels along the temporal dimension. 

By partitioning the input feature map into two segments, the GSM applies a shift operation to one segment while using a gating mechanism (typically a sigmoid-activated 1x1 convolution) to control the information flow from the previous frame. This allows the network to capture temporal dependencies with negligible computational overhead (zero FLOPs and zero parameters for the shift, minimal for the gate) compared to 3D convolutions.

### Relevant Papers:
*   **Original Proposal:** The module was introduced by **Sudhakaran et al. (2020)** in the paper *“Gate-Shift Module for Efficient Video Understanding.”* It was designed to bridge the gap between the efficiency of 2D architectures and the temporal reasoning capabilities of 3D architectures.

*   **Extensions and Applications:**
    *   **ActionSpot (2021):** *“ActionSpot: Localizing Actions in Sports Videos”* (by V. S. R. K. Varma et al.) explicitly utilizes the GSM as a core temporal feature extraction block within their framework. They leverage the module to capture local temporal context efficiently, enabling the model to perform real-time event spotting in long-form sports videos without the prohibitive cost of 3D backbones.
    *   **Subsequent Adaptations:** While many modern temporal action localization (TAL) models have shifted toward Transformer-based architectures (e.g., ActionFormer) or causal temporal convolutions, the GSM remains a benchmark for "efficient-first" approaches in resource-constrained video analysis tasks.

If you are referring to a specific "here" (e.g., a reading list provided in a previous prompt), please note that I do not have access to your personal files or previous attachments unless they are pasted into this chat. If you provide the paper titles, I can confirm which others extend this architecture.

---

### M07 (multi_paper) -- Across these papers, what are the main open challenges in precise event spotting?
**Required facts:** RAG 2/3 | closed book 3/3

**With retrieval:**

Based on the provided documents, the primary challenges in Precise Event Spotting (PES) stem from the need for high temporal precision (within a few frames) in identifying rare, short-duration actions. Key technical requirements and challenges include:

*   **Feature Discriminability:** Models must capture subtle spatial-temporal visual differences and specific motion patterns across neighboring frames to distinguish between fine-grained event classes.
*   **Long-term Temporal Reasoning:** Because precise events are often short and rare, models require a wide temporal context to reduce ambiguity. For example, identifying the exact moment of a figure skating jump requires observing the broader context of acceleration and landing, which can span many seconds.
*   **Data Sparsity:** Precise events represent a tiny fraction of total video frames (e.g., 0.23% in the figure skating dataset and 2.2% in FineDiving), making them difficult to detect.
*   **Post-processing Trade-offs:** While methods like Non-Maximum Suppression (NMS) help suppress nearby detections to improve precision, they also carry the risk of reducing recall.
*   **Annotation Costs:** Precise spotting requires frame-accurate labels, which are more demanding to produce than interval-based annotations used in general Temporal Action Localization.

**From memory only:**

Current research in sports event spotting and temporal action localization (TAL) identifies several persistent bottlenecks:

*   **Boundary Ambiguity:** Precisely defining start and end timestamps is inherently subjective, especially for non-atomic actions. Standard intersection-over-union (IoU) metrics often penalize models for minor temporal offsets that are semantically negligible, leading to noisy label distribution and inconsistent training signals.
*   **Action Context and Duration Variance:** Many sports events are context-dependent (e.g., a "goal" is defined by the preceding buildup). Models struggle to capture long-range temporal dependencies while maintaining the high temporal resolution required to distinguish between similar sub-actions occurring in short succession.
*   **Data Scarcity and Annotation Cost:** Fine-grained temporal annotations (frame-level) are prohibitively expensive to scale. Consequently, research is shifting toward weakly supervised or semi-supervised learning, which often suffers from localization "leakage," where models capture context frames rather than the event core.
*   **Class Imbalance and Long-tail Distributions:** High-frequency events (e.g., "running") dominate the feature space, making it difficult to detect rare, high-impact events (e.g., "red card" or "specific tactical fouls") without significant false-positive rates.
*   **Computational Efficiency for Real-time Inference:** Most state-of-the-art architectures rely on heavy feature extractors (e.g., Video Swin, I3D). Achieving real-time performance on edge devices or broadcast streams while maintaining high temporal precision remains a major deployment hurdle.
*   **Feature Representation Quality:** Traditional RGB-only models frequently fail to leverage multi-modal cues (audio, metadata, or broadcast graphics), which are often necessary to disambiguate events where the visual evidence is occluded or visually ambiguous (e.g., "ball out of bounds").

---
