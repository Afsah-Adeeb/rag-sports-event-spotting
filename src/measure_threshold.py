"""
Measure where to set config.LOW_CONFIDENCE_THRESHOLD.

The monitoring layer (telemetry.py) flags an answer as "hallucination risk"
when retrieval scored badly but the model answered confidently anyway. That
needs a number for "scored badly" -- and picking one by vibe is exactly the
kind of unexplainable magic constant worth avoiding.

So we measure it. Run a batch of questions the corpus *should* answer and a
batch it clearly should not, then look at the top-1 similarity each produces.
If the two groups separate cleanly, the threshold belongs in the gap between
them; the midpoint maximizes the margin on both sides.

This is corpus- and model-specific: change the papers or the embedding model
and the distributions shift, so re-run this and update config.py.

    cd src
    ../.venv/Scripts/python.exe measure_threshold.py
"""

import statistics

import config
from retrieve import retrieve

# Questions the corpus genuinely covers.
ON_TOPIC = [
    "What is the E2E-Spot architecture?",
    "How does T-DEED improve temporal discriminability?",
    "What is the SoccerNet action spotting dataset?",
    "How is average-mAP computed for action spotting?",
    "What is the CALF temporal context aware loss?",
    "How do these methods handle class imbalance?",
    "What backbone networks are used for feature extraction?",
    "What is temporal action localization?",
    "How is label misalignment in annotations handled?",
    "What role does optical flow play in event spotting?",
]

# Questions the corpus clearly does not cover. Deliberately spread across
# unrelated domains so a single odd match can't skew the maximum.
OFF_TOPIC = [
    "What houses exist at Hogwarts?",
    "How do I bake sourdough bread?",
    "What is the capital of Brazil?",
    "Explain quantum entanglement to a child.",
    "Who won the 2018 FIFA World Cup final?",
    "What are the side effects of ibuprofen?",
    "How do I file my income tax return?",
    "Write a poem about the ocean.",
]


def probe(label, questions):
    """Retrieve for each question and report the top-1 similarity spread."""
    print(f"\n=== {label} ===")
    top_scores = []
    for question in questions:
        chunks = retrieve(question, top_k=config.DEFAULT_TOP_K)
        top = chunks[0]["score"] if chunks else 0.0
        top_scores.append(top)
        print(f"  top1={top:.3f}  {question}")
    print(f"  --> min={min(top_scores):.3f}  "
          f"median={statistics.median(top_scores):.3f}  max={max(top_scores):.3f}")
    return top_scores


def main():
    on = probe("ON-TOPIC (corpus should answer these)", ON_TOPIC)
    off = probe("OFF-TOPIC (corpus should not answer these)", OFF_TOPIC)

    lowest_on, highest_off = min(on), max(off)
    print("\n=== SEPARATION ===")
    print(f"  lowest on-topic top-1  : {lowest_on:.3f}")
    print(f"  highest off-topic top-1: {highest_off:.3f}")

    if lowest_on <= highest_off:
        print("\n  The two groups OVERLAP -- no threshold separates them cleanly.")
        print("  A single similarity cutoff will misclassify some questions here.")
        print("  Consider more/better chunks, a stronger embedding model, or")
        print("  treating the flag as advisory only.")
        return

    midpoint = (lowest_on + highest_off) / 2
    print(f"  gap                    : {lowest_on - highest_off:+.3f} (clean separation)")
    print(f"\n  Suggested LOW_CONFIDENCE_THRESHOLD = {midpoint:.2f}")
    print(f"  (config.py currently has {config.LOW_CONFIDENCE_THRESHOLD})")


if __name__ == "__main__":
    main()
