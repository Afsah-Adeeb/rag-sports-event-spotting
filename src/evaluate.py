"""
Step 5: Evaluate retrieval quality (automated) and answer faithfulness (manual).

WHAT WE MEASURE, AND WHY
---------------------------
Given a set of hand-labeled test questions (each tagged with which paper(s)
should count as a correct source), we compute two retrieval metrics at
several values of k:

  - Hit Rate@k: for what fraction of questions did AT LEAST ONE of the
    top-k retrieved chunks come from a correct source paper? This answers
    "did retrieval find the right paper at all?" -- it's the metric most
    people mean when they informally say "precision@k" for RAG retrieval.

  - Precision@k: of the top-k retrieved chunks, what fraction came from a
    correct source paper, on average? This is precision@k in the strict
    IR sense -- it penalizes retrieving a lot of irrelevant chunks even if
    one correct one sneaks in near the top. (Interview-relevant distinction:
    these two metrics are commonly conflated but measure different things --
    hit rate is about whether you found it, precision is about how much
    noise surrounds it.)

We do NOT automate answer faithfulness (whether the generated answer only
makes claims the retrieved chunks actually support). Automating that
well requires an LLM-as-judge setup, which is more moving parts than this
project needs right now. Instead, this script generates an answer for
every test question and writes a report with the answer next to its
retrieved chunks, with a checklist for you to review by eye -- exactly
what was asked for: "manually check if generated answers stay faithful."

INPUT FORMAT (eval/test_questions.json)
------------------------------------------
A JSON list of objects: {"question": str, "correct_papers": [filename, ...]}.
`correct_papers` lists every source-paper filename that would count as a
correct retrieval for that question (usually one, sometimes more if
multiple papers cover the same topic).

RELATIONSHIP TO telemetry.py
------------------------------
This script and telemetry.py answer different questions and are deliberately
kept separate:

  evaluate.py  -- offline, on questions I hand-labeled, with a ground truth
                  to score against. "Is retrieval correct?"
  telemetry.py -- online, on whatever real users actually asked, with no
                  ground truth. "What is it doing in production, how fast,
                  and where does it look unsafe?"

Note that evaluate.py deliberately does NOT write telemetry events. A batch
eval run would otherwise dump 20 synthetic questions into the production
metrics and skew every latency and confidence number in the dashboard.
"""

import json
from statistics import mean

import config
from generate import generate_answer_from_chunks
from retrieve import retrieve


def load_test_questions():
    with open(config.TEST_QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def hit_at_k(retrieved_chunks, correct_papers, k):
    top_k = retrieved_chunks[:k]
    return 1 if any(c["source_paper"] in correct_papers for c in top_k) else 0


def precision_at_k(retrieved_chunks, correct_papers, k):
    top_k = retrieved_chunks[:k]
    relevant = sum(1 for c in top_k if c["source_paper"] in correct_papers)
    return relevant / k


def evaluate():
    questions = load_test_questions()
    max_k = max(config.EVAL_K_VALUES)

    per_question_results = []
    hit_scores = {k: [] for k in config.EVAL_K_VALUES}
    precision_scores = {k: [] for k in config.EVAL_K_VALUES}

    for q in questions:
        question_text = q["question"]
        correct_papers = q["correct_papers"]

        # Retrieve once at the largest k we need; smaller k's are just prefixes of this list.
        retrieved = retrieve(question_text, top_k=max_k)
        answer = generate_answer_from_chunks(question_text, retrieved[: config.DEFAULT_TOP_K])

        for k in config.EVAL_K_VALUES:
            hit_scores[k].append(hit_at_k(retrieved, correct_papers, k))
            precision_scores[k].append(precision_at_k(retrieved, correct_papers, k))

        per_question_results.append({
            "question": question_text,
            "correct_papers": correct_papers,
            "retrieved": retrieved,
            "answer": answer,
        })

    summary = {
        k: {"hit_rate": mean(hit_scores[k]), "precision": mean(precision_scores[k])}
        for k in config.EVAL_K_VALUES
    }

    write_report(per_question_results, summary)
    print_summary(summary)


def write_report(results, summary):
    lines = ["# Evaluation Report", ""]

    lines.append("## Aggregate retrieval metrics")
    lines.append("")
    lines.append("| k | Hit Rate@k | Precision@k |")
    lines.append("|---|---|---|")
    for k, scores in summary.items():
        lines.append(f"| {k} | {scores['hit_rate']:.2f} | {scores['precision']:.2f} |")
    lines.append("")
    lines.append(
        "- **Hit Rate@k**: fraction of questions where at least one chunk from a "
        "correct source paper appeared in the top-k retrieved chunks (did we find the right paper?).\n"
        "- **Precision@k**: average fraction of the top-k retrieved chunks that came "
        "from a correct source paper (how much noise is mixed in?)."
    )
    lines.append("")

    lines.append("## Per-question detail")
    lines.append("")
    lines.append("Manually review each **Generated answer** against its **Retrieved chunks** "
                  "and mark whether every claim in the answer is actually supported by the chunks.")
    lines.append("")

    for i, r in enumerate(results, start=1):
        lines.append(f"### Q{i}: {r['question']}")
        lines.append(f"**Correct paper(s):** {', '.join(r['correct_papers'])}")
        lines.append("")
        lines.append("**Retrieved chunks:**")
        for j, c in enumerate(r["retrieved"], start=1):
            mark = "[x]" if c["source_paper"] in r["correct_papers"] else "[ ]"
            lines.append(f"{j}. {mark} {c['source_paper']} (p.{c['page_start']}-{c['page_end']}, score={c['score']:.3f})")
        lines.append("")
        lines.append(f"**Generated answer:**\n\n{r['answer']}")
        lines.append("")
        lines.append("**Faithfulness check (fill in manually):** [ ] Faithful  [ ] Contains unsupported claims")
        lines.append("")
        lines.append("---")
        lines.append("")

    config.EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary(summary):
    print("Aggregate retrieval metrics:")
    for k, scores in summary.items():
        print(f"  k={k}: Hit Rate@{k}={scores['hit_rate']:.2f}  Precision@{k}={scores['precision']:.2f}")
    print(f"\nFull report (with generated answers for manual faithfulness review) written to {config.EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()
