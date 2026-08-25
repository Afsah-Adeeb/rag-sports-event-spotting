# Evaluation Report

## Aggregate retrieval metrics

| k | Hit Rate@k | Precision@k |
|---|---|---|
| 1 | 0.67 | 0.67 |
| 3 | 1.00 | 0.67 |
| 5 | 1.00 | 0.73 |

- **Hit Rate@k**: fraction of questions where at least one chunk from a correct source paper appeared in the top-k retrieved chunks (did we find the right paper?).
- **Precision@k**: average fraction of the top-k retrieved chunks that came from a correct source paper (how much noise is mixed in?).

## Per-question detail

Manually review each **Generated answer** against its **Retrieved chunks** and mark whether every claim in the answer is actually supported by the chunks.

### Q1: What is the E2E-Spot architecture?
**Correct paper(s):** Spotting Temporally Precise, Fine-Grained Events in Video.pdf

**Retrieved chunks:**
1. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.14-14, score=0.518)
2. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.11-11, score=0.495)
3. [ ] Deep learning for action spotting in association football videos.pdf (p.14-14, score=0.460)
4. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.3-3, score=0.451)
5. [x] Spotting Temporally Precise, Fine-Grained Events in Video.pdf (p.15-15, score=0.418)

**Generated answer:**

Based on the provided context, the exact layer-by-layer details of the E2E-Spot architecture are not fully detailed, but its core design characteristics include:

* **End-to-End Sequence Design:** E2E-Spot is a compact, sequence-based end-to-end architecture that combines well-established architectural components to extract task-specialized spatial-temporal features [`Spotting Temporally Precise, Fine-Grained Events in Video.pdf`, p.3-3, p.14-14].
* **Global and Local Reasoning:** It is designed to make stateful predictions by reasoning both globally and locally over large temporal windows (e.g., 100s of frames) [`Spotting Temporally Precise, Fine-Grained Events in Video.pdf`, p.11-11; `Deep learning for action spotting in association football videos.pdf`, p.14-14].
* **Efficiency and Variants:** Its compact footprint minimizes memory utilization, allowing it to be trained on a single GPU [`Spotting Temporally Precise, Fine-Grained Events in Video.pdf`, p.3-3; `Deep learning for action spotting in association football videos.pdf`, p.14-14]. It comes in model variants such as 200MF and 800MF [`Spotting Temporally Precise, Fine-Grained Events in Video.pdf`, p.11-11, p.15-15].

*Note: The context does not contain further specifics on the exact network backbones, heads, or sub-modules.*

**Faithfulness check (fill in manually):** [ ] Faithful  [ ] Contains unsupported claims

---

### Q2: How does T-DEED handle temporal discriminability?
**Correct paper(s):** T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf

**Retrieved chunks:**
1. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.2-2, score=0.477)
2. [ ] Multi-Focus Temporal Shifting for Precise Event.pdf (p.6-6, score=0.445)
3. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.1-1, score=0.438)
4. [ ] Deep learning for action spotting in association football videos.pdf (p.16-16, score=0.429)
5. [x] T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf (p.3-3, score=0.418)

**Generated answer:**

Based on the provided context, T-DEED handles temporal discriminability through the following mechanisms:

* **SGP Layer and Temporal Modules:** T-DEED incorporates a Scalable Granular Pooling (SGP) layer specifically designed to increase token discriminability within the same temporal sequence `[T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf, p.2-2]`, alongside dedicated temporal modules aimed at boosting token discriminability `[T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf, p.1-1]`.
* **Temporally Discriminant Encoder-Decoder:** It utilizes a multi-scale encoder-decoder architecture that captures both local and global temporal information to enhance token and frame-level discriminability while achieving a high output temporal resolution for precise event spotting `[T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf, p.1-1, p.3-3; Deep learning for action spotting in association football videos.pdf, p.16-16]`.

**Faithfulness check (fill in manually):** [ ] Faithful  [ ] Contains unsupported claims

---

### Q3: What datasets does SoccerNet provide?
**Correct paper(s):** Deep learning for action spotting in association football videos.pdf, Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf

**Retrieved chunks:**
1. [ ] Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf (p.2-2, score=0.740)
2. [x] Deep learning for action spotting in association football videos.pdf (p.3-3, score=0.689)
3. [x] Deep learning for action spotting in association football videos.pdf (p.5-5, score=0.657)
4. [x] Deep learning for action spotting in association football videos.pdf (p.3-3, score=0.643)
5. [x] Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf (p.16-16, score=0.628)

**Generated answer:**

Based on the provided context, the datasets provided under the SoccerNet umbrella include:

* **SoccerNet Action Spotting (v1)** (2018): Consists of 500 full broadcast football matches (764 hours) from major European leagues, annotated across 3 main action classes (Goals, Cards, Substitutions) with event timestamps [Deep learning for action spotting in association football videos.pdf, p.3-3, p.5-5].
* **SoccerNet Action Spotting (v2)**: Expands the original dataset to 500+50 games and 110,458 annotations across 17 action classes (adding events like "Foul," "Throw-in," and "Shot on target"). Annotations use single timestamps (rather than 1-second intervals) accompanied by visibility tags (explicitly visible vs. inferred) [Deep learning for action spotting in association football videos.pdf, p.5-5; Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf, p.16-16].
* **SoccerNet Ball Action Spotting (2023)**: Contains 11,041 annotations across 7+2 classes [Deep learning for action spotting in association football videos.pdf, p.5-5].
* **SoccerNet Ball Action Spotting (2024)**: Contains 11,041 annotations across 7+2 classes [Deep learning for action spotting in association football videos.pdf, p.5-5].

**Faithfulness check (fill in manually):** [ ] Faithful  [ ] Contains unsupported claims

---
