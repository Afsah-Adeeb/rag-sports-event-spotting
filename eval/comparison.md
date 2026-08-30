# Semantic RAG vs Agentic RAG

Corpus: 9 papers. Questions: 3. Model (both arms): `gemini-3.1-flash-lite`.

**Semantic** pre-embeds every chunk and returns the top-5 nearest to the question, in one step. **Agentic** has no index and searches the papers with grep/read tools in a loop, deciding what to look at next.

## Results

| Metric | Semantic | Agentic | Ratio |
|---|---|---|---|
| Hit Rate (correct paper retrieved) | **1.00** | **1.00** | — |
| Median latency | 6.1s | 12.1s | 2.0x |
| Mean input tokens | 1,357 | 5,296 | 3.9x |
| Mean output tokens | 157 | 306 | 1.9x |
| Total tokens (all questions) | 4,542 | 16,805 | 3.7x |
| Mean API round trips | 1.0 | 2.3 | 2.3x |

**Hit Rate** = fraction of questions where at least one correct source paper appeared in what was retrieved (top-k chunk sources for semantic; papers actually grepped or read for agentic).

Precision@k is not reported: semantic returns exactly k chunks so precision is well-defined, while agentic returns however many papers the model chose to open. The same formula would mean different things on each side.

> **Sample size warning: only 3 question(s).** Both arms scored identically, which at this sample size means the questions did not separate them — not that the approaches are equivalent. Roughly 15-20 labelled questions are needed before Hit Rate is worth quoting. The token, latency, and round-trip ratios are far more trustworthy at this n, because they average a per-question cost that varies little rather than a pass/fail proportion.

## Per-question detail

Both answers are shown for manual faithfulness review — as in `evaluate.py`, answer quality is deliberately not auto-scored.

### Q1: What is the E2E-Spot architecture?

**Correct paper(s):** Spotting Temporally Precise, Fine-Grained Events in Video.pdf

| | Semantic | Agentic |
|---|---|---|
| Found correct paper | yes | yes |
| Latency | 7.2s | 8.3s |
| Tokens (in/out) | 1,368/169 | 3,385/353 |
| API round trips | 1 | 2 |

<details><summary>What each retrieved</summary>

*Semantic — top-k chunks:*

- Spotting Temporally Precise, Fine-Grained Events in Video.pdf p.14-14 (0.518)
- Spotting Temporally Precise, Fine-Grained Events in Video.pdf p.11-11 (0.495)
- Deep learning for action spotting in association football videos.pdf p.14-14 (0.460)
- Spotting Temporally Precise, Fine-Grained Events in Video.pdf p.3-3 (0.451)
- Spotting Temporally Precise, Fine-Grained Events in Video.pdf p.15-15 (0.418)

*Agentic — tool calls made:*

- grep_papers(pattern='E2E-Spot') -> 7373 chars

</details>

**Semantic answer:**

The E2E-Spot architecture is a compact, end-to-end model designed for precise, fine-grained temporal action spotting. It is built from the ground up using well-established architectural components and focuses on three key design principles: the use of task-specialized spatial-temporal features, reasoning over sufficient temporal context, and efficient end-to-end learning. 

The architecture is designed to be computationally efficient, allowing it to be trained quickly on a single GPU (such as an A5000) and to process large temporal windows despite limited memory. As a sequence model, it is capable of making stateful predictions over long temporal contexts (e.g., hundreds of frames). Performance can be further enhanced using larger CNNs or optical flow, although these are not strictly required for the model's functionality.

**Agentic answer:**

E2E-Spot is a frame-aware, end-to-end trainable architecture specifically designed for the task of Precise Event Spotting (PES) in sports videos. It is widely recognized as a foundational, efficient baseline in the domain.

Its key architectural components and characteristics include:

*   **Fully Trainable Backbone:** Unlike prior methods that utilized pre-extracted or frozen features from models trained on unrelated datasets, E2E-Spot employs a fully trainable CNN backbone—typically based on **RegNetY**—allowing it to be fine-tuned directly for the specific action-spotting task.
*   **Temporal Modeling:** The model incorporates a **Gated Shift Module (GSM)** to perform temporal reasoning. This module helps the network capture temporal dynamics, and the architecture is complemented by a **GRU-based module** for processing sequence information and producing per-frame predictions.
*   **Efficiency:** The model features a compact design, which enables training on large temporal windows even with limited GPU memory. It is highly computationally efficient at inference, often completing the entire end-to-end task faster than the feature extraction phase alone in many earlier methods.
*   **Head:** The output head is designed for dense frame prediction, facilitating precise, frame-accurate event localization.

By integrating these components, E2E-Spot established a standard framework for PES that has since served as a foundation for various state-of-the-art extensions, such as T-DEED, UGLF, and ASTRM, which often use E2E-Spot as a baseline or a framework for incorporating more advanced temporal modules.

**Faithfulness check (fill in manually):** semantic [ ] faithful [ ] unsupported · agentic [ ] faithful [ ] unsupported

---

### Q2: How does T-DEED handle temporal discriminability?

**Correct paper(s):** T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf

| | Semantic | Agentic |
|---|---|---|
| Found correct paper | yes | yes |
| Latency | 6.1s | 13.5s |
| Tokens (in/out) | 1,427/82 | 9,307/296 |
| API round trips | 1 | 3 |

<details><summary>What each retrieved</summary>

*Semantic — top-k chunks:*

- T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf p.2-2 (0.477)
- Multi-Focus Temporal Shifting for Precise Event.pdf p.6-6 (0.445)
- T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf p.1-1 (0.438)
- Deep learning for action spotting in association football videos.pdf p.16-16 (0.429)
- T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.pdf p.3-3 (0.418)

*Agentic — tool calls made:*

- grep_papers(pattern='T-DEED', context=3) -> 11889 chars
- read_paper(offset=5, limit=50, name='T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder for Precise.txt') -> 7572 chars

</details>

**Semantic answer:**

T-DEED handles temporal discriminability by incorporating a specifically designed temporal module—the (SGP) layer—into its architecture. This layer is designed to increase token discriminability within the same temporal sequence. Furthermore, the model's overall encoder-decoder architecture is engineered to leverage multiple temporal scales, which, combined with the SGP layer, enhances frame-level discriminability to support precise action spotting.

**Agentic answer:**

T-DEED (Temporal-Discriminability Enhancer Encoder-Decoder) improves temporal discriminability primarily through its use of **SGP-Mixer (Scalable-Granularity Perception-Mixer) layers**.

These layers are designed to address the challenge of differentiating between adjacent frames that may have high spatial similarity. The T-DEED architecture achieves this in the following ways:

*   **Token Discriminability:** The SGP-Mixer module adapts the core principles of Scalable-Granularity Perception (SGP) to aggregate information across different temporal scales, which specifically enhances the discriminability of tokens (frames) within the same temporal sequence.
*   **Encoder-Decoder Integration:** By incorporating residual connections, the SGP-Mixer layers are integrated into an encoder-decoder architecture. This allows the model to leverage features from multiple temporal scales, helping it capture actions with diverse temporal contexts.
*   **Temporal Resolution:** The encoder-decoder structure is designed to restore the original temporal resolution, which is critical for maintaining high precision in spotting fast-paced sports events where even a minor temporal deviation could result in a missed event.

**Faithfulness check (fill in manually):** semantic [ ] faithful [ ] unsupported · agentic [ ] faithful [ ] unsupported

---

### Q3: What datasets does SoccerNet provide?

**Correct paper(s):** Deep learning for action spotting in association football videos.pdf, Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf

| | Semantic | Agentic |
|---|---|---|
| Found correct paper | yes | yes |
| Latency | 3.5s | 12.1s |
| Tokens (in/out) | 1,275/221 | 3,196/268 |
| API round trips | 1 | 2 |

<details><summary>What each retrieved</summary>

*Semantic — top-k chunks:*

- Automated Event Detection and Classification in Soccer - The Potential of Using Multiple Modalities.pdf p.2-2 (0.740)
- Deep learning for action spotting in association football videos.pdf p.3-3 (0.689)
- Deep learning for action spotting in association football videos.pdf p.5-5 (0.657)
- Deep learning for action spotting in association football videos.pdf p.3-3 (0.643)
- Deep Learning for Sports Video Event Detection - Tasks, Datasets, Methods, and Challenges.pdf p.16-16 (0.628)

*Agentic — tool calls made:*

- grep_papers(pattern='SoccerNet') -> 8194 chars

</details>

**Semantic answer:**

The SoccerNet umbrella provides several datasets for action spotting, including:

*   **SoccerNet Action Spotting (v1):** The original 2018 release, containing 500 full broadcast football matches (764 hours) covering three action classes: goals, cards, and substitutions.
*   **SoccerNet Action Spotting (v2):** An extension that expands the action classes from 3 to 17, including events such as fouls, throw-ins, and shots on target. Unlike v1, events are annotated with a single timestamp and include a visibility tag.
*   **SoccerNet Ball Action Spotting (2023 and 2024 versions):** Datasets focused on ball-related actions, featuring 7 plus 2 classes and 11,041 annotations. 

Additionally, the collection includes datasets that incorporate SoccerNet data alongside other sources, such as a dataset comprising 270 matches from SoccerNet combined with 76 matches from the Chinese Super League and FIFA World Cup editions.

**Agentic answer:**

The SoccerNet initiative provides a large-scale collection of datasets designed for sports video understanding, primarily focused on association football (soccer). The main datasets and variants mentioned include:

*   **SoccerNet (SoccerNet-v1):** The initial dataset, introduced to establish the task of action spotting in broadcast soccer videos. It consists of over 550 complete broadcast games annotated with various common soccer events, such as goals, substitutions, and cards.
*   **SoccerNet-v2:** An evolved version of the original dataset that includes more comprehensive benchmarks for the holistic understanding of broadcast soccer videos. It supports tasks like action spotting with relaxed (loose) and tight evaluation criteria, accommodating different temporal tolerances (from seconds to nearly a minute).
*   **SoccerNet Action Spotting:** A specific subset of the SoccerNet collection dedicated to the task of action spotting, where models are trained to identify and categorize events within the broadcast stream.
*   **SoccerNet Ball Action Spotting:** A specialized variant or challenge developed for more granular event identification involving the ball.

The SoccerNet initiative also organizes annual challenges, using these datasets to evaluate state-of-the-art methods in tasks such as action spotting and precise event spotting.

**Faithfulness check (fill in manually):** semantic [ ] faithful [ ] unsupported · agentic [ ] faithful [ ] unsupported

---
