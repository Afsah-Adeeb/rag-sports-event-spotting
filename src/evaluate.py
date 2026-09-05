"""
Step 5: Offline evaluation against the labelled test set.

WHAT THIS ANSWERS
-------------------
"On questions where I know the right answer, how good is the system?"
Retrieval is scored automatically against hand-labelled correct papers;
answer quality is scored against hand-written required facts.

WHAT CHANGED, AND WHY
-----------------------
This script used to report a bare Hit Rate@k and Precision@k over three
example questions, and left answer quality to a checkbox in the report
that nobody ever ticked. Three problems, all now fixed:

  1. NO ERROR BARS. A hit rate of 1.00 over 3 questions and a hit rate of
     1.00 over 40 are not the same claim, but they printed identically.
     Every number here now carries a 95% interval (see eval_core.py for
     why proportions get a Wilson interval and means get a bootstrap).
  2. NO BREAKDOWN. One average across all questions hides the failure
     mode: 85% overall can be 100% on simple lookups and 0% on questions
     needing several papers. Everything is reported per question type too.
  3. NO RANK INFORMATION. Hit Rate@5 scores "found at position 1" and
     "found at position 5" the same, so a system quietly depending on all
     five slots looked identical to one that nails it first. MRR fixes that.

RETRIEVAL RUNS FREE, GENERATION IS OPT-IN
-------------------------------------------
Scoring retrieval touches no API at all -- it is local embedding plus a
FAISS search, so the default run costs nothing and takes seconds. That
matters because the settings sweep (sweep.py) re-runs retrieval dozens of
times. Answer generation costs one API call per question, so it is behind
`--generate` rather than happening on every run by accident.

Generation uses config.BENCHMARK_MODEL_NAME, not the model the live app
uses. Free-tier quota is per-model-per-day; a 40-question run on the app's
model would exhaust the deployed demo's daily budget.

THE UNANSWERABLE QUESTIONS
----------------------------
15 of the 55 questions have no answer in the corpus. They are not scored
for retrieval accuracy -- there is no correct paper to find. What they
give instead is a control group: retrieval confidence on questions that
DO have an answer should sit clearly above confidence on questions that
do not. If those two distributions overlap, the low-confidence threshold
in config.py cannot separate "found it" from "found nothing", and the
hallucination flag in the monitoring dashboard is guesswork.

RELATIONSHIP TO telemetry.py
------------------------------
Kept deliberately separate, and this script still does NOT write telemetry
events:

  evaluate.py  -- offline, on questions I labelled, with a ground truth to
                  score against. "Is retrieval correct?"
  telemetry.py -- online, on whatever real users asked, with no ground
                  truth. "What is it doing in production, and where does
                  it look unsafe?"

A batch run dumping 55 synthetic questions into the production event log
would skew every latency and confidence number on the dashboard.
"""

import argparse
import sys
import time

import config
import eval_core as ec
from generate import generate_with_usage
from retrieve import retrieve
from telemetry import looks_like_refusal


# --- Running -----------------------------------------------------------------

def run_retrieval(questions, top_k=None):
    """Retrieve for every question and score the retrieval. No API calls."""
    top_k = top_k or max(config.EVAL_K_VALUES)
    records = []

    for q in questions:
        chunks = retrieve(q["question"], top_k=top_k)
        correct = q.get("correct_papers", [])

        record = {
            "id": q["id"],
            "question": q["question"],
            "type": q.get("type", "untyped"),
            "answerable": q.get("answerable", True),
            "correct_papers": correct,
            "must_mention": q.get("must_mention", []),
            "nice_to_mention": q.get("nice_to_mention", []),
            "note": q.get("note", ""),
            "retrieved": chunks,
            "top_score": ec.top_score(chunks),
        }

        # Only meaningful when there is a correct paper to find.
        if record["answerable"] and correct:
            for k in config.EVAL_K_VALUES:
                record[f"hit@{k}"] = ec.hit_at_k(chunks, correct, k)
                record[f"p@{k}"] = ec.precision_at_k(chunks, correct, k)
            record["rank"] = ec.first_correct_rank(chunks, correct)
            record["rr"] = ec.reciprocal_rank(chunks, correct)
            record["paper_coverage"] = ec.papers_covered(chunks, correct)

        records.append(record)

    return records


def run_generation(records, top_k=config.DEFAULT_TOP_K, model=None, pause=0.0):
    """Generate an answer per question and score facts + refusal. Costs quota."""
    model = model or config.BENCHMARK_MODEL_NAME
    total = len(records)

    for i, record in enumerate(records, start=1):
        print(f"  generating {i}/{total}  {record['id']} ...",
              end="\r", file=sys.stderr, flush=True)

        # Wrapped in the retry: the free tier allows 15 requests/minute, and a
        # 55-question run trips that within the first minute. Without this the
        # whole batch dies partway through and the quota already spent is
        # wasted, which is exactly what happened on the first attempt.
        result = ec.call_with_retry(
            lambda: generate_with_usage(
                record["question"], record["retrieved"][:top_k], model=model),
            label="evaluate.run_generation")
        answer = result["answer"]

        record["answer"] = answer
        record["generation_ms"] = result["generation_ms"]
        record["input_tokens"] = result.get("input_tokens")
        record["output_tokens"] = result.get("output_tokens")

        # Shared with the live monitoring layer on purpose: the eval and the
        # dashboard must agree on what counts as a refusal, or the two
        # disagree about the same answer.
        record["refused"] = looks_like_refusal(answer)

        frac, found, total_facts = ec.fact_coverage(answer, record["must_mention"])
        record["fact_frac"] = frac
        record["facts_found"] = found
        record["facts_total"] = total_facts
        record["missing"] = ec.missing_facts(answer, record["must_mention"])

        nice_frac, nice_found, nice_total = ec.fact_coverage(
            answer, record["nice_to_mention"])
        record["nice_frac"] = nice_frac
        record["nice_found"] = nice_found
        record["nice_total"] = nice_total

        if pause:
            time.sleep(pause)

    print(" " * 60, end="\r", file=sys.stderr)
    return records


# --- Scoring -----------------------------------------------------------------

def rescore_from_cache(records, cache):
    """Re-apply the answer-side scoring to previously generated answers. Free.

    Same principle as telemetry._derive_flags(): raw signals are stored, and
    judgement is recomputed on read. Improving the refusal heuristic or fixing
    a wrong label should retroactively correct the report, not require paying
    for 55 fresh generations to see the effect.

    Matched on question TEXT rather than id, deliberately. Ids change when a
    question is reclassified -- U02 became S19 after the evaluation revealed it
    was mislabelled -- and matching on id would silently drop exactly the
    answers that mattered most.
    """
    by_question = {c["question"]: c for c in cache if c.get("answer")}
    reused = 0
    for record in records:
        cached = by_question.get(record["question"])
        if not cached:
            continue
        reused += 1
        answer = cached["answer"]
        record["answer"] = answer
        record["generation_ms"] = cached.get("generation_ms")
        record["input_tokens"] = cached.get("input_tokens")
        record["output_tokens"] = cached.get("output_tokens")
        record["refused"] = looks_like_refusal(answer)

        frac, found, total = ec.fact_coverage(answer, record["must_mention"])
        record["fact_frac"] = frac
        record["facts_found"] = found
        record["facts_total"] = total
        record["missing"] = ec.missing_facts(answer, record["must_mention"])
        nf, nfound, ntotal = ec.fact_coverage(answer, record["nice_to_mention"])
        record["nice_frac"], record["nice_found"], record["nice_total"] = nf, nfound, ntotal
    return records, reused


def score(records):
    """Aggregate per-question records into headline numbers with intervals."""
    scored = [r for r in records if r["answerable"] and r["correct_papers"]]
    unans = [r for r in records if not r["answerable"]]

    summary = {"n_scored": len(scored), "n_unanswerable": len(unans)}

    for k in config.EVAL_K_VALUES:
        summary[f"hit@{k}"] = ec.summarise_proportion([r[f"hit@{k}"] for r in scored])
        summary[f"p@{k}"] = ec.summarise_mean([r[f"p@{k}"] for r in scored])

    summary["mrr"] = ec.summarise_mean([r["rr"] for r in scored])
    summary["paper_coverage"] = ec.summarise_mean([r["paper_coverage"] for r in scored])

    # The control-group comparison: confidence with an answer vs without one.
    summary["conf_answerable"] = ec.summarise_mean([r["top_score"] for r in scored])
    summary["conf_unanswerable"] = ec.summarise_mean([r["top_score"] for r in unans])
    if scored and unans:
        summary["conf_gap"] = (min(r["top_score"] for r in scored)
                               - max(r["top_score"] for r in unans))
        summary["conf_separated"] = summary["conf_gap"] > 0

    # Where the first correct chunk actually lands.
    ranks = [r["rank"] for r in scored if r["rank"] is not None]
    summary["rank_1"] = sum(1 for r in ranks if r == 1)
    summary["rank_dist"] = {i: sum(1 for r in ranks if r == i)
                            for i in range(1, max(config.EVAL_K_VALUES) + 1)}

    if any("answer" in r for r in records):
        summary.update(_score_generation(records, scored, unans))

    return summary


def _score_generation(records, scored, unans):
    """Answer-side aggregates. Only present when --generate was used."""
    out = {"generated": True}

    fact_fracs = [r["fact_frac"] for r in scored if r.get("fact_frac") is not None]
    out["fact_coverage"] = ec.summarise_mean(fact_fracs)
    out["fact_perfect"] = ec.summarise_proportion(
        [1 if r.get("fact_frac") == 1.0 else 0
         for r in scored if r.get("fact_frac") is not None])

    # The two directions of the refusal question, scored separately. A system
    # that refuses everything gets a perfect score on one and a terrible one
    # on the other, which is why neither is reported alone.
    out["correct_refusal"] = ec.summarise_proportion(
        [1 if r.get("refused") else 0 for r in unans])
    out["over_refusal"] = ec.summarise_proportion(
        [1 if r.get("refused") else 0 for r in scored])

    answered_unanswerable = [r for r in unans if not r.get("refused")]
    out["hallucination_candidates"] = [r["id"] for r in answered_unanswerable]

    tokens = [r.get("input_tokens") for r in records if r.get("input_tokens")]
    out["mean_input_tokens"] = sum(tokens) / len(tokens) if tokens else None
    gen_ms = [r.get("generation_ms") for r in records if r.get("generation_ms")]
    out["median_generation_ms"] = sorted(gen_ms)[len(gen_ms) // 2] if gen_ms else None

    return out


def score_by_type(records):
    """The same headline numbers, computed within each question type."""
    out = {}
    for qtype, group in ec.group_by_type(records).items():
        scored = [r for r in group if r["answerable"] and r["correct_papers"]]
        entry = {"n": len(group)}
        if scored:
            entry["hit@5"] = ec.summarise_proportion(
                [r[f"hit@{max(config.EVAL_K_VALUES)}"] for r in scored])
            entry["mrr"] = ec.summarise_mean([r["rr"] for r in scored])
            entry["paper_coverage"] = ec.summarise_mean(
                [r["paper_coverage"] for r in scored])
        entry["conf"] = ec.summarise_mean([r["top_score"] for r in group])
        if any("answer" in r for r in group):
            fracs = [r["fact_frac"] for r in group if r.get("fact_frac") is not None]
            if fracs:
                entry["fact_coverage"] = ec.summarise_mean(fracs)
            entry["refused"] = ec.summarise_proportion(
                [1 if r.get("refused") else 0 for r in group])
        out[qtype] = entry
    return out


# --- Reporting ---------------------------------------------------------------

TYPE_ORDER = ["simple", "paraphrase", "comparison", "multi_paper", "unanswerable"]


def _ordered_types(by_type):
    known = [t for t in TYPE_ORDER if t in by_type]
    return known + [t for t in by_type if t not in TYPE_ORDER]


def print_summary(summary, by_type):
    max_k = max(config.EVAL_K_VALUES)
    print()
    print(f"Retrieval  (n={summary['n_scored']} answerable questions, "
          f"95% intervals)")
    for k in config.EVAL_K_VALUES:
        print(f"  Hit Rate@{k}      {ec.format_ci(summary[f'hit@{k}'])}"
              f"     Precision@{k}  {ec.format_ci(summary[f'p@{k}'])}")
    print(f"  MRR              {ec.format_ci(summary['mrr'], pct=False)}"
          f"      (right paper first: {summary['rank_1']}/{summary['n_scored']})")
    print(f"  Paper coverage   {ec.format_ci(summary['paper_coverage'])}"
          f"     (fraction of ALL correct papers reached)")

    print()
    print(f"Confidence control  (n={summary['n_unanswerable']} unanswerable)")
    print(f"  answerable       {ec.format_ci(summary['conf_answerable'], pct=False)}")
    print(f"  unanswerable     {ec.format_ci(summary['conf_unanswerable'], pct=False)}")
    if "conf_gap" in summary:
        verdict = "separated" if summary["conf_separated"] else "OVERLAPPING"
        print(f"  gap              {summary['conf_gap']:+.3f}  ({verdict}; "
              f"threshold is {config.LOW_CONFIDENCE_THRESHOLD})")

    print()
    print("By question type")
    print(f"  {'type':<14}{'n':>4}  {'Hit@'+str(max_k):>18}  {'MRR':>18}")
    for qtype in _ordered_types(by_type):
        entry = by_type[qtype]
        hit = ec.format_ci(entry["hit@5"]) if "hit@5" in entry else "-"
        mrr = ec.format_ci(entry["mrr"], pct=False) if "mrr" in entry else "-"
        print(f"  {qtype:<14}{entry['n']:>4}  {hit:>18}  {mrr:>18}")

    if summary.get("generated"):
        print()
        print("Answers")
        print(f"  Fact coverage    {ec.format_ci(summary['fact_coverage'])}"
              f"     (all facts present: {ec.format_ci(summary['fact_perfect'])})")
        print(f"  Correct refusal  {ec.format_ci(summary['correct_refusal'])}"
              f"     (declined when it should)")
        print(f"  Over-refusal     {ec.format_ci(summary['over_refusal'])}"
              f"     (declined when it should not have)")
        if summary["hallucination_candidates"]:
            print(f"  ANSWERED ANYWAY: {', '.join(summary['hallucination_candidates'])}")

    caveat = ec.sample_size_caveat(summary["n_scored"])
    if caveat:
        print()
        print("  " + caveat.replace("**", ""))
    print()
    print(f"Full report written to {config.EVAL_RESULTS_PATH}")


def write_report(records, summary, by_type, top_k):
    max_k = max(config.EVAL_K_VALUES)
    L = ["# Evaluation Report", ""]
    L.append(f"Test set: {summary['n_scored']} answerable + "
             f"{summary['n_unanswerable']} unanswerable questions. "
             f"Retrieval at top-k={top_k}.")
    L.append("")
    L.append("All figures carry a 95% confidence interval in brackets. Proportions use "
             "a Wilson interval, means use a seeded bootstrap -- see `eval_core.py` for "
             "why the two differ.")
    L.append("")

    caveat = ec.sample_size_caveat(summary["n_scored"])
    if caveat:
        L += ["> " + caveat, ""]

    L += ["## Retrieval", "", "| Metric | Score [95% CI] |", "|---|---|"]
    for k in config.EVAL_K_VALUES:
        L.append(f"| Hit Rate@{k} | {ec.format_ci(summary[f'hit@{k}'])} |")
    for k in config.EVAL_K_VALUES:
        L.append(f"| Precision@{k} | {ec.format_ci(summary[f'p@{k}'])} |")
    L.append(f"| MRR | {ec.format_ci(summary['mrr'], pct=False)} |")
    L.append(f"| Paper coverage | {ec.format_ci(summary['paper_coverage'])} |")
    L.append("")
    L.append("- **Hit Rate@k** -- did at least one of the top-k chunks come from a "
             "correct paper? *Did we find it at all.*")
    L.append("- **Precision@k** -- what fraction of the top-k came from a correct "
             "paper? *How much noise came with it.*")
    L.append("- **MRR** -- 1/(position of the first correct chunk). *Where we found "
             "it.* Hit Rate scores position 1 and position 5 identically; this does not.")
    L.append("- **Paper coverage** -- fraction of ALL correct papers reached, which "
             "matters for the multi-paper questions that need three or four at once.")
    L.append("")

    L += ["### Where the first correct chunk lands", "",
          "| Rank | Questions |", "|---|---|"]
    for rank, count in sorted(summary["rank_dist"].items()):
        L.append(f"| {rank} | {count} |")
    missed = summary["n_scored"] - sum(summary["rank_dist"].values())
    L.append(f"| not in top-{top_k} | {missed} |")
    L.append("")
    L.append("If most of the mass sits at rank 4-5, the system depends on every slot "
             "and lowering `DEFAULT_TOP_K` would break it. If it sits at rank 1, a "
             "smaller k is free.")
    L.append("")

    L += ["## Confidence control: answerable vs unanswerable", ""]
    L.append("The 15 unanswerable questions have no correct paper to find, so they are "
             "not scored for accuracy. They serve as a control: retrieval confidence "
             "on questions the corpus *can* answer should sit clearly above confidence "
             "on questions it cannot.")
    L.append("")
    L += ["| Group | Mean top-1 similarity [95% CI] |", "|---|---|"]
    L.append(f"| Answerable | {ec.format_ci(summary['conf_answerable'], pct=False)} |")
    L.append(f"| Unanswerable | {ec.format_ci(summary['conf_unanswerable'], pct=False)} |")
    L.append("")
    if "conf_gap" in summary:
        if summary["conf_separated"]:
            L.append(f"The two groups **separate cleanly** (gap of "
                     f"{summary['conf_gap']:+.3f} between the worst answerable and the "
                     f"best unanswerable question). `LOW_CONFIDENCE_THRESHOLD` is "
                     f"currently {config.LOW_CONFIDENCE_THRESHOLD}.")
        else:
            L.append(f"The two groups **overlap** by {abs(summary['conf_gap']):.3f}. No "
                     f"single similarity threshold can separate 'found it' from 'found "
                     f"nothing' on this corpus, which means the hallucination flag in "
                     f"the monitoring dashboard cannot be fully trusted. Worth saying "
                     f"out loud rather than tuning the threshold until it looks clean.")
    L.append("")

    L += ["## By question type", "",
          f"| Type | n | Hit@{max_k} | MRR | Paper coverage | Mean confidence |",
          "|---|---|---|---|---|---|"]
    for qtype in _ordered_types(by_type):
        e = by_type[qtype]
        L.append(f"| {qtype} | {e['n']} | "
                 f"{ec.format_ci(e['hit@5']) if 'hit@5' in e else '-'} | "
                 f"{ec.format_ci(e['mrr'], pct=False) if 'mrr' in e else '-'} | "
                 f"{ec.format_ci(e['paper_coverage']) if 'paper_coverage' in e else '-'} | "
                 f"{ec.format_ci(e['conf'], pct=False)} |")
    L.append("")
    L.append("This table is where the findings are. A single average across all "
             "questions can be 100% on simple lookups and 0% on multi-paper ones and "
             "still print as a healthy number.")
    L.append("")

    if summary.get("generated"):
        L += ["## Answers", "", "| Metric | Score [95% CI] |", "|---|---|"]
        L.append(f"| Fact coverage | {ec.format_ci(summary['fact_coverage'])} |")
        L.append(f"| All required facts present | {ec.format_ci(summary['fact_perfect'])} |")
        L.append(f"| Correct refusal (unanswerable) | {ec.format_ci(summary['correct_refusal'])} |")
        L.append(f"| Over-refusal (answerable) | {ec.format_ci(summary['over_refusal'])} |")
        L.append("")
        L.append("Refusal is reported in **both** directions on purpose. A system that "
                 "declines everything scores 1.00 on correct refusal and 1.00 on "
                 "over-refusal; neither number means anything alone.")
        L.append("")
        if summary["hallucination_candidates"]:
            L.append(f"**Answered a question the corpus cannot answer:** "
                     f"{', '.join(summary['hallucination_candidates'])}. These are the "
                     f"hallucination cases -- read them.")
            L.append("")

    L += ["## Per-question detail", ""]
    for r in records:
        L.append(f"### {r['id']} ({r['type']}) -- {r['question']}")
        if r["note"]:
            L.append(f"*Why this question is here: {r['note']}*")
        L.append("")
        if r["answerable"]:
            L.append(f"**Correct paper(s):** {', '.join(r['correct_papers'])}")
            if "rank" in r:
                L.append(f"**First correct chunk at rank:** "
                         f"{r['rank'] if r['rank'] else f'not in top-{top_k}'}")
        else:
            L.append("**Expected behaviour:** refuse -- the corpus does not contain "
                     "this answer.")
        L.append("")
        L.append("**Retrieved:**")
        for j, c in enumerate(r["retrieved"], start=1):
            mark = "[x]" if c["source_paper"] in r["correct_papers"] else "[ ]"
            L.append(f"{j}. {mark} {c['source_paper']} "
                     f"(p.{c['page_start']}-{c['page_end']}, score={c['score']:.3f})")
        L.append("")
        if "answer" in r:
            L.append(f"**Answer:**\n\n{r['answer']}")
            L.append("")
            if r["facts_total"]:
                L.append(f"**Required facts:** {r['facts_found']}/{r['facts_total']}")
                if r["missing"]:
                    pretty = ", ".join("/".join(f) for f in r["missing"])
                    L.append(f"**Missing:** {pretty}")
            if not r["answerable"]:
                verdict = "REFUSED (correct)" if r["refused"] else "ANSWERED ANYWAY (hallucination)"
                L.append(f"**Refusal check:** {verdict}")
            elif r["refused"]:
                L.append("**Refusal check:** REFUSED although the answer exists "
                         "(over-refusal)")
            L.append("")
        L.append("---")
        L.append("")

    config.EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.EVAL_RESULTS_PATH.write_text("\n".join(L), encoding="utf-8")


# --- Entry point -------------------------------------------------------------

def evaluate(top_k=None, generate=False, pause=0.0, from_cache=False):
    top_k = top_k or max(config.EVAL_K_VALUES)
    questions = ec.load_questions()

    print(f"Retrieving for {len(questions)} questions at top-k={top_k} "
          f"(no API calls) ...")
    records = run_retrieval(questions, top_k=top_k)

    if from_cache:
        cache = ec.load_records(config.ANSWER_CACHE_PATH)
        if not cache:
            print(f"No cache at {config.ANSWER_CACHE_PATH.name}. "
                  f"Run with --generate first.")
        else:
            records, reused = rescore_from_cache(records, cache)
            print(f"  re-scored {reused}/{len(records)} cached answers "
                  f"(no API calls)")

    if generate:
        print(f"Generating {len(records)} answers with "
              f"{config.BENCHMARK_MODEL_NAME} ...")
        run_generation(records, top_k=config.DEFAULT_TOP_K, pause=pause)
        # Cached so judge.py grades exactly these answers rather than paying to
        # regenerate a different set -- otherwise a disagreement between the
        # two reports could be resampling rather than a real difference.
        ec.save_records(records, config.ANSWER_CACHE_PATH)
        print(f"  answers cached to {config.ANSWER_CACHE_PATH.name}")

    summary = score(records)
    by_type = score_by_type(records)

    biggest_k = max(config.EVAL_K_VALUES)
    ec.update_summary("retrieval", {
        "n_answerable": summary["n_scored"],
        "n_unanswerable": summary["n_unanswerable"],
        "top_k": top_k,
        "hit@1": summary["hit@1"],
        f"hit@{biggest_k}": summary[f"hit@{biggest_k}"],
        "mrr": summary["mrr"],
        "paper_coverage": summary["paper_coverage"],
        "conf_answerable": summary["conf_answerable"],
        "conf_unanswerable": summary["conf_unanswerable"],
        "conf_separated": summary.get("conf_separated"),
        "by_type": {t: {"n": e["n"], "hit": e.get("hit@5"), "mrr": e.get("mrr")}
                    for t, e in by_type.items()},
        "correct_refusal": summary.get("correct_refusal"),
        "over_refusal": summary.get("over_refusal"),
        "fact_coverage": summary.get("fact_coverage"),
        "hallucination_candidates": summary.get("hallucination_candidates", []),
    })

    write_report(records, summary, by_type, top_k)
    print_summary(summary, by_type)
    return records, summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval (free) and optionally answers (costs quota).")
    parser.add_argument("--generate", action="store_true",
                        help="also generate an answer per question and score facts "
                             "and refusal behaviour (one API call per question)")
    parser.add_argument("--top-k", type=int, default=None,
                        help=f"chunks to retrieve (default {max(config.EVAL_K_VALUES)})")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds to wait between generations, to stay under the "
                             "free tier's per-minute rate limit")
    parser.add_argument("--from-cache", action="store_true",
                        help="re-score previously generated answers instead of paying "
                             "to regenerate them -- use after improving the refusal "
                             "heuristic or fixing a label")
    args = parser.parse_args()
    evaluate(top_k=args.top_k, generate=args.generate, pause=args.pause,
             from_cache=args.from_cache)


if __name__ == "__main__":
    main()
