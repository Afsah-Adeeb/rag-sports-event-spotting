"""
The control experiment: how much does retrieval actually contribute?

THE PROBLEM THIS SOLVES
-------------------------
Every other number in this project measures the system *with* retrieval.
None of them establish that retrieval is what produced the answer.

That matters here more than in most RAG projects, because this corpus is
nine well-known, publicly available research papers. E2E-Spot, T-DEED and
SoccerNet are on arXiv, have been cited hundreds of times, and were almost
certainly in the model's training data. So a strong score is ambiguous:
it could mean retrieval is working, or it could mean the model already
knew the answer and the retrieved chunks were decorative.

The standard way to break that ambiguity is a CLOSED-BOOK baseline: ask
the same questions with no documents at all, and see what the model can
produce from memory alone. The difference between the two arms is what
retrieval is worth. If closed-book scores 0.70 and the full system scores
0.80, the RAG pipeline is buying ten points, not eighty.

FAIRNESS: THE CLOSED-BOOK ARM GETS ITS OWN PROMPT
---------------------------------------------------
The obvious implementation -- reuse build_prompt() with an empty context
block -- is wrong, and quietly so. That prompt instructs the model to
answer ONLY from the provided context and to say so when the context is
insufficient. Handed no context, it will refuse every single question and
score zero, and the experiment would have measured the prompt rather than
the model's knowledge.

So the closed-book arm gets a prompt that asks the question straight,
with explicit permission to say "I don't know". It is the same model, the
same questions, and the same scoring; the only difference is whether
retrieved text is present. That is the one variable under test.

WHAT THE UNANSWERABLE QUESTIONS SHOW HERE
-------------------------------------------
Running the 15 unanswerable questions through both arms answers a second
question that matters just as much: does retrieval make the model MORE
honest or LESS? Handing it five chunks that are topically relevant but do
not contain the answer could plausibly do either -- give it grounds to say
"this isn't in the sources", or give it enough related material to
confabulate around. Measured rather than assumed.
"""

import argparse
import sys
import time

# This script lives in a subfolder but imports the pipeline modules that sit in
# src/ (config, retrieve, generate, telemetry). Running a script puts its OWN
# folder on the import path, not its parent, so the parent is added explicitly.
# Siblings inside this folder import normally.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import config
import eval_core as ec
from evaluate import run_generation, run_retrieval
from generate import _get_client
from telemetry import looks_like_refusal


def closed_book_prompt(question):
    """Ask the question with no documents, fairly.

    Deliberately NOT generate.build_prompt() with an empty context -- see the
    module docstring. This prompt gives the model every reasonable chance to
    answer from memory, while still allowing an honest "I don't know", so the
    refusal comparison between the two arms stays meaningful.
    """
    return f"""You are a research assistant answering questions about academic \
papers on sports video event-spotting and temporal action localization.

Answer from your own knowledge. Follow these rules:
- If you do not know the answer, say so explicitly instead of guessing.
- Be concise and technical -- this is for a researcher, not a general audience.

Question: {question}

Answer:"""


def run_closed_book(records, model=None, pause=0.0):
    """Answer every question with no retrieved context, and score the result."""
    model = model or config.BENCHMARK_MODEL_NAME
    client = _get_client()
    total = len(records)

    for i, record in enumerate(records, start=1):
        print(f"  closed-book {i}/{total}  {record['id']} ...",
              end="\r", file=sys.stderr, flush=True)

        response = ec.call_with_retry(
            lambda: client.models.generate_content(
                model=model, contents=closed_book_prompt(record["question"])),
            label="closed-book")
        answer = response.text

        record["cb_answer"] = answer
        record["cb_refused"] = looks_like_refusal(answer)
        frac, found, total_facts = ec.fact_coverage(answer, record["must_mention"])
        record["cb_fact_frac"] = frac
        record["cb_facts_found"] = found
        record["cb_facts_total"] = total_facts
        record["cb_missing"] = ec.missing_facts(answer, record["must_mention"])

        if pause:
            time.sleep(pause)

    print(" " * 60, end="\r", file=sys.stderr)
    return records


def score(records):
    scored = [r for r in records if r["answerable"] and r["correct_papers"]]
    unans = [r for r in records if not r["answerable"]]

    rag_fracs = [r["fact_frac"] for r in scored if r.get("fact_frac") is not None]
    cb_fracs = [r["cb_fact_frac"] for r in scored if r.get("cb_fact_frac") is not None]

    summary = {
        "n_scored": len(scored),
        "n_unanswerable": len(unans),
        "rag_facts": ec.summarise_mean(rag_fracs),
        "cb_facts": ec.summarise_mean(cb_fracs),
        "rag_perfect": ec.summarise_proportion(
            [1 if r.get("fact_frac") == 1.0 else 0 for r in scored]),
        "cb_perfect": ec.summarise_proportion(
            [1 if r.get("cb_fact_frac") == 1.0 else 0 for r in scored]),
        "rag_refuse_unans": ec.summarise_proportion(
            [1 if r.get("refused") else 0 for r in unans]),
        "cb_refuse_unans": ec.summarise_proportion(
            [1 if r.get("cb_refused") else 0 for r in unans]),
        "rag_refuse_ans": ec.summarise_proportion(
            [1 if r.get("refused") else 0 for r in scored]),
        "cb_refuse_ans": ec.summarise_proportion(
            [1 if r.get("cb_refused") else 0 for r in scored]),
    }
    summary["contribution"] = summary["rag_facts"]["value"] - summary["cb_facts"]["value"]

    # Paired per-question comparison. The aggregate difference can hide the
    # fact that retrieval helps on some questions and actively hurts on others.
    summary["rag_better"] = [r["id"] for r in scored
                             if (r.get("fact_frac") or 0) > (r.get("cb_fact_frac") or 0)]
    summary["cb_better"] = [r["id"] for r in scored
                            if (r.get("cb_fact_frac") or 0) > (r.get("fact_frac") or 0)]
    summary["tied"] = [r["id"] for r in scored
                       if (r.get("fact_frac") or 0) == (r.get("cb_fact_frac") or 0)]

    # "Gold leak": the model produced required facts for a question WITHOUT
    # being shown any source. Names the questions this corpus cannot claim
    # credit for, because the model already knew them.
    summary["gold_leak"] = [r["id"] for r in scored if (r.get("cb_fact_frac") or 0) >= 0.5]
    return summary


def by_type(records):
    out = {}
    for qtype, group in ec.group_by_type(records).items():
        rag = [r["fact_frac"] for r in group if r.get("fact_frac") is not None]
        cb = [r["cb_fact_frac"] for r in group if r.get("cb_fact_frac") is not None]
        if not rag and not cb:
            continue
        out[qtype] = {
            "n": len(group),
            "rag": ec.summarise_mean(rag),
            "cb": ec.summarise_mean(cb),
        }
    return out


def write_report(records, summary, types):
    L = ["# Closed-Book Control: what is retrieval actually worth?", ""]
    L.append("Every question asked twice with the same model "
             f"(`{config.BENCHMARK_MODEL_NAME}`): once through the full RAG pipeline, "
             "and once with **no documents at all**. The gap between the two is what "
             "retrieval contributes.")
    L.append("")
    L.append("This matters here because the corpus is nine well-known arXiv papers that "
             "were almost certainly in the model's training data. Without this control, "
             "a good score cannot be attributed to retrieval rather than recall.")
    L.append("")
    L.append("*The closed-book arm is given its own prompt rather than the RAG prompt "
             "with an empty context block. The RAG prompt orders the model to answer "
             "only from provided context, so with no context it would refuse "
             "everything and score zero -- measuring the prompt, not the model.*")
    L.append("")

    caveat = ec.sample_size_caveat(summary["n_scored"])
    if caveat:
        L += ["> " + caveat, ""]

    L += ["## Answer quality", "",
          f"| Metric | Full RAG | Closed book | Retrieval buys |", "|---|---|---|---|"]
    L.append(f"| Required facts present | {ec.format_ci(summary['rag_facts'])} | "
             f"{ec.format_ci(summary['cb_facts'])} | "
             f"**{summary['contribution']:+.2f}** |")
    L.append(f"| All facts present | {ec.format_ci(summary['rag_perfect'])} | "
             f"{ec.format_ci(summary['cb_perfect'])} | "
             f"{summary['rag_perfect']['value'] - summary['cb_perfect']['value']:+.2f} |")
    L.append("")

    cb = summary["cb_facts"]["value"]
    rag = summary["rag_facts"]["value"]
    if cb <= 0.05:
        L.append(f"**Closed-book scores {cb:.2f}.** The model cannot answer these "
                 f"questions from memory, so essentially all of the {rag:.2f} the full "
                 f"system achieves is produced by retrieval. That is the cleanest "
                 f"possible version of this result.")
    elif summary["contribution"] <= 0.10:
        L.append(f"**Closed-book scores {cb:.2f} against the full system's {rag:.2f}.** "
                 f"Retrieval is buying only {summary['contribution']:+.2f}. The model "
                 f"already knows most of what these questions ask, which means the "
                 f"headline scores elsewhere in this project overstate what the "
                 f"pipeline contributes. Worth stating plainly rather than quoting the "
                 f"full number as if retrieval earned it.")
    else:
        L.append(f"**Retrieval contributes {summary['contribution']:+.2f}** -- from "
                 f"{cb:.2f} on memory alone to {rag:.2f} with documents. That gap, not "
                 f"the {rag:.2f}, is the honest measure of what this pipeline adds.")
    L.append("")

    L += ["### Question by question", "",
          "| Outcome | Count | Questions |", "|---|---|---|"]
    for label, key in [("RAG better", "rag_better"), ("Tied", "tied"),
                       ("Closed book better", "cb_better")]:
        ids = summary[key]
        L.append(f"| {label} | {len(ids)} | {', '.join(f'`{i}`' for i in ids) or '-'} |")
    L.append("")
    L.append("Averages hide direction. If retrieval wins on some questions and loses "
             "on others, the mean can look flat while both effects are real.")
    L.append("")

    if summary["gold_leak"]:
        L.append(f"**Already known to the model ({len(summary['gold_leak'])} questions):** "
                 f"{', '.join(f'`{i}`' for i in summary['gold_leak'])}. These scored at "
                 f"least half their required facts with no documents supplied. The "
                 f"system's score on them cannot be credited to retrieval.")
        L.append("")

    L += ["## Honesty: does retrieval make it more or less careful?", "",
          f"| Behaviour | Full RAG | Closed book |", "|---|---|---|"]
    L.append(f"| Refused the {summary['n_unanswerable']} unanswerable questions | "
             f"{ec.format_ci(summary['rag_refuse_unans'])} | "
             f"{ec.format_ci(summary['cb_refuse_unans'])} |")
    L.append(f"| Refused answerable questions (over-refusal) | "
             f"{ec.format_ci(summary['rag_refuse_ans'])} | "
             f"{ec.format_ci(summary['cb_refuse_ans'])} |")
    L.append("")
    delta = (summary["rag_refuse_unans"]["value"] - summary["cb_refuse_unans"]["value"])
    if delta > 0.05:
        L.append(f"Retrieval makes the model **more** willing to decline unanswerable "
                 f"questions ({delta:+.2f}). Seeing chunks that do not contain the "
                 f"answer gives it grounds to say so.")
    elif delta < -0.05:
        L.append(f"Retrieval makes the model **less** willing to decline ({delta:+.2f}). "
                 f"Handed five topically-related chunks, it confabulates around them "
                 f"rather than noticing the answer is absent. This is the failure mode "
                 f"the confidence threshold in `config.py` cannot catch either.")
    else:
        L.append("Retrieval barely changes refusal behaviour on unanswerable questions.")
    L.append("")

    L += ["## By question type", "",
          "| Type | n | Full RAG | Closed book | Retrieval buys |", "|---|---|---|---|---|"]
    for qtype in ("simple", "paraphrase", "comparison", "multi_paper"):
        if qtype not in types:
            continue
        e = types[qtype]
        L.append(f"| {qtype} | {e['n']} | {ec.format_ci(e['rag'])} | "
                 f"{ec.format_ci(e['cb'])} | "
                 f"{e['rag']['value'] - e['cb']['value']:+.2f} |")
    L.append("")

    L += ["## Side by side", ""]
    for r in records:
        if not (r["answerable"] and r["correct_papers"]):
            continue
        L.append(f"### {r['id']} ({r['type']}) -- {r['question']}")
        L.append(f"**Required facts:** RAG {r['facts_found']}/{r['facts_total']} "
                 f"| closed book {r['cb_facts_found']}/{r['cb_facts_total']}")
        L.append("")
        L.append(f"**With retrieval:**\n\n{r['answer']}")
        L.append("")
        L.append(f"**From memory only:**\n\n{r['cb_answer']}")
        L.append("")
        L.append("---")
        L.append("")

    path = config.CLOSED_BOOK_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def print_summary(summary):
    print()
    print(f"{'':<28}{'full RAG':>20}{'closed book':>20}")
    print(f"{'required facts present':<28}"
          f"{ec.format_ci(summary['rag_facts']):>20}"
          f"{ec.format_ci(summary['cb_facts']):>20}")
    print(f"{'all facts present':<28}"
          f"{ec.format_ci(summary['rag_perfect']):>20}"
          f"{ec.format_ci(summary['cb_perfect']):>20}")
    print(f"{'refused unanswerable':<28}"
          f"{ec.format_ci(summary['rag_refuse_unans']):>20}"
          f"{ec.format_ci(summary['cb_refuse_unans']):>20}")
    print()
    print(f"  retrieval contributes {summary['contribution']:+.2f} on fact coverage")
    print(f"  RAG better on {len(summary['rag_better'])}, "
          f"tied on {len(summary['tied'])}, "
          f"closed book better on {len(summary['cb_better'])}")
    if summary["gold_leak"]:
        print(f"  already known to the model: {len(summary['gold_leak'])} questions "
              f"({', '.join(summary['gold_leak'])})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Measure what retrieval contributes, by asking every question "
                    "with and without documents.")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds between calls, to stay under the per-minute limit")
    args = parser.parse_args()

    questions = ec.load_questions()
    print(f"Retrieving for {len(questions)} questions ...")
    records = run_retrieval(questions)

    print(f"Arm 1/2: generating WITH retrieval ({config.BENCHMARK_MODEL_NAME}) ...")
    run_generation(records, pause=args.pause)

    print(f"Arm 2/2: generating WITHOUT documents ...")
    run_closed_book(records, pause=args.pause)

    summary = score(records)
    types = by_type(records)
    ec.save_records(records, config.ANSWER_CACHE_PATH)
    ec.update_summary("closed_book", {
        "n_answerable": summary["n_scored"],
        "rag_facts": summary["rag_facts"],
        "cb_facts": summary["cb_facts"],
        "contribution": summary["contribution"],
        "rag_refuse_unans": summary["rag_refuse_unans"],
        "cb_refuse_unans": summary["cb_refuse_unans"],
        "n_gold_leak": len(summary["gold_leak"]),
        "gold_leak": summary["gold_leak"],
    })
    path = write_report(records, summary, types)
    print_summary(summary)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
