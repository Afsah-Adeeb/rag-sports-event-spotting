"""
An LLM judge, used as a cross-check rather than as the truth.

WHY A JUDGE AT ALL
--------------------
The fact-coverage metric in eval_core.py is deterministic, free, and
reproducible -- but it is literal. It checks whether specific strings
appear. It cannot tell that "a bidirectional GRU processes the per-frame
features" satisfies a required fact written as ["GRU"] plus context, and
it cannot notice that an answer is fluent, confident, and completely
unsupported by the retrieved text. An LLM judge can, in principle, do
both. It is also what the field actually uses, so being able to build one
and reason about it is worth more than avoiding it.

WHY IT IS NOT THE HEADLINE METRIC
-----------------------------------
The 2026 literature on LLM-as-judge reliability is not encouraging, and
the specific findings apply directly to this setup:

  - Raw agreement with human graders overstates real skill badly. Across
    21 judge models, exact-match agreement exceeded chance-corrected
    agreement by 34-41 percentage points; a judge reporting "85% agreement"
    was really at kappa ~0.48.
  - Automated RAG metric suites correlate with human judgement at around
    0.55 -- useful for spotting trends, not for grading.
  - Consistency is not correctness. Judges with test-retest reliability
    above 0.95 were simultaneously carrying position bias above 0.10:
    reliably wrong.
  - Position bias varies about 100-fold ACROSS MODELS IN THE SAME FAMILY.
    Gemini 2.5 Pro measured 0.002; Gemini 2.5 Flash measured 0.125, 62x
    worse. This project runs a Flash-class model for cost reasons, which
    puts it on the wrong side of that split.

So the judge here grades the same answers the deterministic metric graded,
and the report's real output is WHERE THE TWO DISAGREE. Agreement is
reassuring; disagreement is where a human should look. Neither is treated
as ground truth.

SELF-CONSISTENCY IS MEASURED, NOT ASSUMED
-------------------------------------------
Every answer is judged twice, in separate calls. If the two passes
disagree with each other, the judge cannot be trusted to adjudicate
anything, and that number is reported before any of its verdicts are.
This is the cheapest version of the "run >=3 iterations" advice in the
literature, and it is the check most projects skip.

The judge is deliberately NOT asked to score on a 1-10 scale. Fine-grained
numeric rubrics are where judge noise concentrates; a small set of
discrete, concretely-defined labels is more stable and easier to audit.
"""

import argparse
import json
import re
import sys
import time

import config
import eval_core as ec
from evaluate import rescore_from_cache, run_generation, run_retrieval
from generate import _get_client

VERDICTS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "REFUSED")


def judge_prompt(question, chunks, answer):
    """Grade one answer against the context it was given.

    Asks for grounding, not correctness-in-general: the question is whether
    the retrieved text supports the answer, which is the thing a RAG system
    is actually responsible for. Asking "is this true?" would let the judge
    grade from its own training data and quietly measure something else.
    """
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return f"""You are grading a retrieval-augmented answer. You will see a QUESTION, \
the CONTEXT passages the system retrieved, and the ANSWER it produced.

Judge ONLY whether the ANSWER is supported by the CONTEXT. Do not use outside \
knowledge, and do not reward an answer for being correct if the CONTEXT does not \
support it.

Reply with a JSON object and nothing else:
{{"verdict": "...", "unsupported_claims": ["..."], "reason": "one short sentence"}}

"verdict" must be exactly one of:
- "SUPPORTED"   -- every claim in the answer is backed by the context
- "PARTIAL"     -- the answer is mostly backed, but contains at least one claim the \
context does not support
- "UNSUPPORTED" -- the answer's main claims are not backed by the context
- "REFUSED"     -- the answer declines to answer, or says the context is insufficient

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


def _parse(text):
    """Pull the JSON verdict out of the reply, tolerating fences and stray prose."""
    if not text:
        return {"verdict": None, "unsupported_claims": [], "reason": "empty reply"}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"verdict": None, "unsupported_claims": [], "reason": "no JSON found"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": None, "unsupported_claims": [], "reason": "bad JSON"}
    verdict = str(data.get("verdict", "")).strip().upper()
    return {
        "verdict": verdict if verdict in VERDICTS else None,
        "unsupported_claims": data.get("unsupported_claims") or [],
        "reason": str(data.get("reason", ""))[:300],
    }


def judge_once(client, record, model, pass_name):
    prompt = judge_prompt(record["question"], record["retrieved"], record["answer"])
    response = ec.call_with_retry(
        lambda: client.models.generate_content(model=model, contents=prompt),
        label=f"judge {pass_name}")
    return _parse(response.text)


def run_judge(records, model=None, pause=0.0, passes=2):
    """Judge every answer `passes` times, in separate calls."""
    model = model or config.BENCHMARK_MODEL_NAME
    client = _get_client()
    total = len(records) * passes

    done = 0
    for p in range(1, passes + 1):
        for record in records:
            done += 1
            print(f"  judging {done}/{total}  pass {p}  {record['id']} ...",
                  end="\r", file=sys.stderr, flush=True)
            record[f"judge{p}"] = judge_once(client, record, model, f"pass {p}")
            if pause:
                time.sleep(pause)

    print(" " * 70, end="\r", file=sys.stderr)
    return records


# --- Scoring -----------------------------------------------------------------

def _judge_says_ok(verdict):
    """Collapse the judge's label to a binary, for comparison with fact coverage."""
    return verdict == "SUPPORTED"


def score(records):
    judged = [r for r in records if r.get("judge1", {}).get("verdict")]
    both = [r for r in judged if r.get("judge2", {}).get("verdict")]

    agree = [1 if r["judge1"]["verdict"] == r["judge2"]["verdict"] else 0 for r in both]
    summary = {
        "n": len(records),
        "n_judged": len(judged),
        "self_consistency": ec.summarise_proportion(agree),
        "unparseable": sum(1 for r in records
                           if not r.get("judge1", {}).get("verdict")),
    }

    counts = {v: 0 for v in VERDICTS}
    for r in judged:
        counts[r["judge1"]["verdict"]] += 1
    summary["verdicts"] = counts

    # The comparison this script exists for: judge vs deterministic metric,
    # on answerable questions where both produced a score.
    scored = [r for r in judged
              if r["answerable"] and r.get("fact_frac") is not None]
    rows = []
    for r in scored:
        facts_ok = r["fact_frac"] == 1.0
        judge_ok = _judge_says_ok(r["judge1"]["verdict"])
        rows.append({
            "id": r["id"], "type": r["type"],
            "question": r["question"],
            "fact_frac": r["fact_frac"],
            "facts_ok": facts_ok,
            "verdict": r["judge1"]["verdict"],
            "judge_ok": judge_ok,
            "agree": facts_ok == judge_ok,
            "reason": r["judge1"]["reason"],
            "missing": r.get("missing", []),
            "claims": r["judge1"]["unsupported_claims"],
        })
    summary["rows"] = rows
    summary["agreement"] = ec.summarise_proportion([r["agree"] for r in rows])
    summary["judge_lenient"] = [r for r in rows if r["judge_ok"] and not r["facts_ok"]]
    summary["judge_strict"] = [r for r in rows if r["facts_ok"] and not r["judge_ok"]]

    # Refusal: does the judge agree with the string-matching refusal heuristic
    # that telemetry.py and the Metrics dashboard both rely on?
    ref_rows = [r for r in judged if "refused" in r]
    summary["refusal_agreement"] = ec.summarise_proportion(
        [1 if (r["judge1"]["verdict"] == "REFUSED") == bool(r["refused"]) else 0
         for r in ref_rows])
    summary["refusal_disagree"] = [
        {"id": r["id"], "heuristic": bool(r["refused"]),
         "judge": r["judge1"]["verdict"], "answerable": r["answerable"]}
        for r in ref_rows
        if (r["judge1"]["verdict"] == "REFUSED") != bool(r["refused"])]

    # Unanswerable questions: the judge's read on whether the system held the line.
    unans = [r for r in judged if not r["answerable"]]
    summary["unans_verdicts"] = {v: sum(1 for r in unans
                                        if r["judge1"]["verdict"] == v)
                                 for v in VERDICTS}
    return summary


def write_report(summary):
    L = ["# LLM Judge: a cross-check, not a verdict", ""]
    L.append(f"Every answer graded twice by `{config.BENCHMARK_MODEL_NAME}` for whether "
             f"it is supported by the context it was given. Run against the exact "
             f"answers the deterministic fact-coverage metric scored, from a shared "
             f"cache -- so any difference between the two is a difference in "
             f"*judgement*, not in which answers were sampled.")
    L.append("")
    L.append("**This is deliberately not the headline metric.** Published 2026 work "
             "finds automated RAG grader suites correlate with human judgement at "
             "around 0.55; that raw judge-human agreement overstates chance-corrected "
             "agreement by 34-41 points; and that position bias varies roughly 100-fold "
             "between models, with Flash-class models measuring far worse than Pro-class "
             "ones. This project runs a Flash-class model for cost reasons, which puts "
             "it on the wrong side of that split. So the judge is used to find "
             "disagreements worth a human's attention, not to hand out scores.")
    L.append("")

    L += ["## First: can the judge agree with itself?", ""]
    sc = summary["self_consistency"]
    L.append(f"Each answer was judged twice in independent calls. The two passes agreed "
             f"on **{ec.format_ci(sc)}** of answers.")
    L.append("")
    if sc["value"] >= 0.9:
        L.append("High self-consistency. Note what this does and does not establish: it "
                 "means the judge is *repeatable*, not that it is *right*. Judges with "
                 "test-retest reliability above 0.95 have been measured carrying severe "
                 "systematic bias at the same time. Repeatability is a precondition for "
                 "trust, not evidence of it.")
    else:
        L.append(f"**The judge does not reliably agree with itself.** At "
                 f"{sc['value']:.0%} self-consistency, a single verdict from it is close "
                 f"to a coin flip on the disputed cases, and none of the numbers below "
                 f"should be read as measurements. This is the check that most projects "
                 f"skip, and it is the reason the deterministic metric stays the "
                 f"headline.")
    L.append("")
    if summary["unparseable"]:
        L.append(f"*{summary['unparseable']} reply/replies could not be parsed as a "
                 f"verdict and are excluded.*")
        L.append("")

    L += ["## Verdicts", "", "| Verdict | Count |", "|---|---|"]
    for v, c in summary["verdicts"].items():
        L.append(f"| {v} | {c} |")
    L.append("")

    L += ["## Judge vs deterministic fact checking", ""]
    L.append(f"On the {len(summary['rows'])} answerable questions, the two methods "
             f"reached the same conclusion **{ec.format_ci(summary['agreement'])}** of "
             f"the time.")
    L.append("")
    L.append("They are asking different questions, so perfect agreement was never the "
             "goal: fact coverage asks *did the answer contain the required facts*, the "
             "judge asks *is everything the answer said supported by the context*. An "
             "answer can pass one and fail the other for good reasons. The disagreements "
             "are the output.")
    L.append("")

    L += [f"### Judge passed it, required facts missing ({len(summary['judge_lenient'])})",
          ""]
    if summary["judge_lenient"]:
        L.append("The answer is well-grounded in what it was given, but does not contain "
                 "everything a complete answer needs. Usually a **retrieval** failure "
                 "rather than a generation one: the model faithfully reported chunks "
                 "that did not carry the missing fact.")
        L.append("")
        L += ["| Question | Facts | Missing |", "|---|---|---|"]
        for r in summary["judge_lenient"]:
            miss = ", ".join("/".join(f) for f in r["missing"][:3]) or "-"
            L.append(f"| `{r['id']}` {r['question'][:60]} | {r['fact_frac']:.2f} | {miss} |")
    else:
        L.append("None.")
    L.append("")

    L += [f"### Facts all present, judge flagged it ({len(summary['judge_strict'])})", ""]
    if summary["judge_strict"]:
        L.append("The answer contained every required fact but the judge found claims "
                 "the context does not support. These are the candidate hallucinations "
                 "that a string-matching metric cannot see, and the reason a judge is "
                 "worth running at all.")
        L.append("")
        for r in summary["judge_strict"]:
            L.append(f"**`{r['id']}`** ({r['type']}) -- verdict {r['verdict']}")
            L.append(f"- {r['question']}")
            L.append(f"- judge's reason: {r['reason']}")
            for c in r["claims"][:3]:
                L.append(f"- flagged claim: {c}")
            L.append("")
    else:
        L.append("None.")
    L.append("")

    L += ["## Does the judge agree with the refusal heuristic?", ""]
    L.append(f"`telemetry.looks_like_refusal()` decides by string matching in the first "
             f"~150 characters, and the Metrics dashboard's hallucination flag is built "
             f"on it. The judge agrees with it on "
             f"**{ec.format_ci(summary['refusal_agreement'])}** of answers.")
    L.append("")
    if summary["refusal_disagree"]:
        L += ["| Question | Heuristic says refused | Judge says | Answerable |",
              "|---|---|---|---|"]
        for d in summary["refusal_disagree"]:
            L.append(f"| `{d['id']}` | {d['heuristic']} | {d['judge']} | "
                     f"{d['answerable']} |")
        L.append("")
        L.append("Each row is worth reading by hand. A cheap heuristic disagreeing with "
                 "an unreliable judge does not tell you which one is wrong -- it tells "
                 "you where to look.")
    else:
        L.append("No disagreements.")
    L.append("")

    L += ["## The unanswerable questions, as the judge sees them", "",
          "| Verdict | Count |", "|---|---|"]
    for v, c in summary["unans_verdicts"].items():
        L.append(f"| {v} | {c} |")
    L.append("")
    L.append("`REFUSED` is the correct outcome for all of these. Anything scored "
             "`SUPPORTED` is the judge claiming the context backs an answer to a "
             "question the corpus cannot answer -- which would be the judge failing, "
             "the system failing, or both.")
    L.append("")

    path = config.JUDGE_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def print_summary(summary):
    print()
    print(f"  judge self-consistency  {ec.format_ci(summary['self_consistency'])}")
    print(f"  agrees with fact check  {ec.format_ci(summary['agreement'])}")
    print(f"  agrees with refusal fn  {ec.format_ci(summary['refusal_agreement'])}")
    print()
    print(f"  verdicts: " + "  ".join(f"{v}={c}" for v, c in summary["verdicts"].items()))
    print(f"  judge lenient (facts missing, judge passed): "
          f"{len(summary['judge_lenient'])}")
    print(f"  judge strict  (facts present, judge flagged): "
          f"{len(summary['judge_strict'])}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-check answers with an LLM judge and report disagreements.")
    parser.add_argument("--pause", type=float, default=config.API_MIN_PAUSE_SECONDS)
    parser.add_argument("--passes", type=int, default=2,
                        help="how many times to judge each answer (>=2 measures "
                             "self-consistency)")
    parser.add_argument("--regenerate", action="store_true",
                        help="ignore the cached answers and generate fresh ones")
    args = parser.parse_args()

    questions = ec.load_questions()
    cache = None if args.regenerate else ec.load_records(config.ANSWER_CACHE_PATH)

    if cache:
        # Re-derive labels and answer scores from the CURRENT test set and the
        # CURRENT refusal heuristic rather than trusting what the cache stored.
        # The cached fields are frozen at generation time, so a question that
        # has since been relabelled, or a refusal phrase added to the marker
        # list afterwards, would otherwise be silently graded against a stale
        # rule -- and the judge-vs-heuristic disagreement this script exists to
        # report would be measuring the staleness instead of the judgement.
        print(f"Using {len(cache)} cached answers from "
              f"{config.ANSWER_CACHE_PATH.name} (no generation calls).")
        records = run_retrieval(questions)
        records, reused = rescore_from_cache(records, cache)
        print(f"  re-derived labels and scores for {reused} answers "
              f"against the current test set")
    else:
        print("No cached answers found -- generating them first.")
        records = run_retrieval(questions)
        run_generation(records, pause=args.pause)
        ec.save_records(records, config.ANSWER_CACHE_PATH)

    records = [r for r in records if r.get("answer")]
    print(f"Judging {len(records)} answers x {args.passes} passes ...")
    run_judge(records, pause=args.pause, passes=args.passes)

    summary = score(records)
    path = write_report(summary)
    ec.update_summary("judge", {
        "self_consistency": summary["self_consistency"],
        "agreement_with_facts": summary["agreement"],
        "refusal_agreement": summary["refusal_agreement"],
        "verdicts": summary["verdicts"],
        "n_judge_lenient": len(summary["judge_lenient"]),
        "n_judge_strict": len(summary["judge_strict"]),
    })
    print_summary(summary)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
