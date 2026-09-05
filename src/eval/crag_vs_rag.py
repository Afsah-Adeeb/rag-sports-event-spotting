"""
Plain RAG vs Corrective RAG, on the same questions, scored by the same code.

WHAT IS BEING COMPARED, AND THE ONE DECISION THAT MAKES IT FAIR
-----------------------------------------------------------------
Both arms are scored on **the context the generator actually received**, not
on what retrieval happened to return.

That choice is the whole ballgame. Plain RAG retrieves five chunks and sends
all five. CRAG retrieves five, grades them, and may look at fifteen before
sending its best five. Scoring CRAG on all fifteen would hand it a higher hit
rate for free -- more chunks cannot lose a paper a smaller set already found --
and the comparison would measure how deep each arm searched rather than how
well it chose. Scoring both on what the model was given keeps the question
honest: given the same budget of five passages, which arm fills them better?

On a refusal CRAG sends nothing, so it scores zero retrieval on that question.
That is the correct penalty: refusing an answerable question means the model
got no useful context, and the arm should be charged for it.

WHAT THIS MEASURES
--------------------
1. Retrieval, on the sent context: hit rate, MRR, paper coverage.
2. Answers: required-fact coverage, and refusal in both directions.
3. Out-of-scope detection -- the headline. config.LOW_CONFIDENCE_THRESHOLD
   was measured to catch only 1 of 15 unanswerable questions, and that failure
   is documented in config.py. This reports what the grader catches instead,
   which is the specific thing CRAG was built to fix.
4. Grader quality, against the labels already in the test set.
5. Cost: API calls, tokens, latency. CRAG buys its correction with an extra
   call, and that has to be shown, not waved away.

WHAT IT CANNOT SHOW, STATED UP FRONT
--------------------------------------
Stage 1 CRAG can only remove chunks or search deeper down the same ranked
list. It cannot invent a better query. So on questions where the right paper
simply is not in the index's top results, no amount of grading will find it,
and hit rate should not be expected to move much. The honest place to look for
improvement is out-of-scope detection and the composition of what gets sent.
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
import grader
from crag import crag_answer
from generate import generate_with_usage
from retrieve import retrieve
from telemetry import looks_like_refusal


# --- Running the two arms ----------------------------------------------------

def run_plain(question, top_k):
    """Plain RAG: retrieve, then answer from everything retrieved."""
    started = time.perf_counter()
    chunks = retrieve(question, top_k=top_k)
    retrieval_ms = (time.perf_counter() - started) * 1000

    result = ec.call_with_retry(
        lambda: generate_with_usage(question, chunks,
                                    model=config.BENCHMARK_MODEL_NAME),
        label="plain RAG")
    return {
        "answer": result["answer"],
        "sent_chunks": chunks,          # plain RAG sends everything it retrieved
        "retrieved": len(chunks),
        "retrieval_ms": retrieval_ms,
        "generation_ms": result["generation_ms"],
        "grading_ms": 0.0,
        "input_tokens": result.get("input_tokens") or 0,
        "output_tokens": result.get("output_tokens") or 0,
        "total_input_tokens": result.get("input_tokens") or 0,
        "api_calls": 1,
        "decision": "answer",
        "deepened": False,
        "labels": [],
    }


def run_crag(question, top_k):
    """Corrective RAG: retrieve, grade, then answer / deepen / refuse."""
    result = ec.call_with_retry(
        lambda: crag_answer(question, top_k=top_k,
                            model=config.BENCHMARK_MODEL_NAME),
        label="CRAG")
    return {
        "answer": result["answer"],
        "sent_chunks": result["sent_chunks"],
        "retrieved": len(result["chunks"]),
        "retrieval_ms": result["retrieval_ms"],
        "generation_ms": result["generation_ms"],
        "grading_ms": result["grading_ms"],
        "input_tokens": result.get("input_tokens") or 0,
        "output_tokens": result.get("output_tokens") or 0,
        "total_input_tokens": result["total_input_tokens"],
        "api_calls": result["api_calls"],
        "decision": result["decision"],
        "deepened": result["deepened"],
        "labels": result["labels"],
        "chunks": result["chunks"],
        "grade_counts": result["grade_counts"],
        "grader_failed": result["grader_failed"],
    }


ARMS = {"plain": run_plain, "crag": run_crag}


def score_one(question_spec, arm_result, top_k):
    """Score one arm's output for one question. Identical code for both arms."""
    correct = question_spec.get("correct_papers", [])
    sent = arm_result["sent_chunks"]
    answer = arm_result["answer"]

    record = dict(arm_result)
    record["id"] = question_spec["id"]
    record["type"] = question_spec.get("type", "untyped")
    record["answerable"] = question_spec.get("answerable", True)
    record["question"] = question_spec["question"]
    record["correct_papers"] = correct

    if record["answerable"] and correct:
        record["hit"] = ec.hit_at_k(sent, correct, top_k)
        record["rr"] = ec.reciprocal_rank(sent, correct)
        record["coverage"] = ec.papers_covered(sent, correct)

    record["refused"] = (arm_result["decision"] == "refuse"
                         or looks_like_refusal(answer))

    frac, found, total = ec.fact_coverage(answer, question_spec.get("must_mention", []))
    record["fact_frac"] = frac
    record["facts_found"] = found
    record["facts_total"] = total
    return record


def run(questions, top_k, pause):
    """Run every question through both arms, alternating so drift hits both."""
    results = {"plain": [], "crag": []}
    total = len(questions) * len(ARMS)
    done = 0

    for spec in questions:
        for name, runner in ARMS.items():
            done += 1
            print(f"  {done}/{total}  {name:<6} {spec['id']} ...",
                  end="\r", file=sys.stderr, flush=True)
            results[name].append(score_one(spec, runner(spec["question"], top_k), top_k))
            if pause:
                time.sleep(pause)

    print(" " * 60, end="\r", file=sys.stderr)
    return results


# --- Scoring -----------------------------------------------------------------

def summarise_arm(records, top_k):
    scored = [r for r in records if r["answerable"] and r["correct_papers"]]
    unans = [r for r in records if not r["answerable"]]

    out = {
        "n_answerable": len(scored),
        "n_unanswerable": len(unans),
        f"hit@{top_k}": ec.summarise_proportion([r["hit"] for r in scored]),
        "mrr": ec.summarise_mean([r["rr"] for r in scored]),
        "coverage": ec.summarise_mean([r["coverage"] for r in scored]),
        "fact_coverage": ec.summarise_mean(
            [r["fact_frac"] for r in scored if r["fact_frac"] is not None]),
        "correct_refusal": ec.summarise_proportion(
            [1 if r["refused"] else 0 for r in unans]),
        "over_refusal": ec.summarise_proportion(
            [1 if r["refused"] else 0 for r in scored]),
        "api_calls": ec.summarise_mean([r["api_calls"] for r in records]),
        "input_tokens": ec.summarise_mean([r["total_input_tokens"] for r in records]),
        "latency_s": ec.summarise_mean(
            [(r["retrieval_ms"] + r["grading_ms"] + r["generation_ms"]) / 1000
             for r in records]),
        "sent_chunks": ec.summarise_mean([len(r["sent_chunks"]) for r in records]),
    }
    out["answered_unanswerable"] = [r["id"] for r in unans if not r["refused"]]
    out["wrongly_refused"] = [r["id"] for r in scored if r["refused"]]
    out["decisions"] = {
        d: sum(1 for r in records if r["decision"] == d)
        for d in ("answer", "refuse")
    }
    out["deepened"] = sum(1 for r in records if r.get("deepened"))
    out["grader_failures"] = sum(1 for r in records if r.get("grader_failed"))
    return out


def grader_quality(crag_records):
    """How good is the grader, judged against labels the test set already has.

    The only solid ground truth available per chunk is which paper it came
    from, so this measures ONE direction well and the other only loosely:

      REJECTION (solid): a chunk from a paper that is not a correct source for
        the question almost certainly does not contain the answer, so the
        grader should call it irrelevant. On unanswerable questions every
        chunk is in this category, which makes them a clean test.

      ACCEPTANCE (loose): a chunk from a correct paper *might* contain the
        answer -- or might be that paper's reference list. Marking such a chunk
        irrelevant is often correct, so a low number here is not necessarily a
        grader failure. Reported, but not treated as an error rate.
    """
    from_wrong_total = from_wrong_rejected = 0
    from_right_total = from_right_accepted = 0
    unans_total = unans_rejected = 0

    for r in crag_records:
        labels = r.get("labels") or []
        chunks = r.get("chunks") or []
        correct = set(r["correct_papers"])
        for chunk, label in zip(chunks, labels):
            is_irrelevant = label == grader.IRRELEVANT
            if not r["answerable"]:
                unans_total += 1
                unans_rejected += is_irrelevant
            elif chunk["source_paper"] in correct:
                from_right_total += 1
                from_right_accepted += not is_irrelevant
            else:
                from_wrong_total += 1
                from_wrong_rejected += is_irrelevant

    def prop(k, n):
        return ec.summarise_proportion([1] * k + [0] * (n - k)) if n else None

    return {
        "unanswerable_rejected": prop(unans_rejected, unans_total),
        "wrong_paper_rejected": prop(from_wrong_rejected, from_wrong_total),
        "right_paper_accepted": prop(from_right_accepted, from_right_total),
        "n_chunks_graded": unans_total + from_wrong_total + from_right_total,
    }


# --- Reporting ---------------------------------------------------------------

def write_report(results, summaries, gq, top_k):
    plain, crag = summaries["plain"], summaries["crag"]
    L = ["# Plain RAG vs Corrective RAG", ""]
    L.append(f"Both arms, same {plain['n_answerable']} answerable + "
             f"{plain['n_unanswerable']} unanswerable questions, same model "
             f"(`{config.BENCHMARK_MODEL_NAME}`), same scoring code.")
    L.append("")
    L.append("**Both are scored on the context the generator actually received**, not "
             "on everything retrieval returned. Plain RAG sends the five chunks it "
             "retrieved; CRAG grades those five, may look at fifteen, and sends its "
             "best five. Scoring CRAG on all fifteen would raise its hit rate for free "
             "-- more chunks cannot lose a paper a smaller set already found -- and "
             "would measure search depth instead of judgement. Given the same budget "
             "of five passages, which arm fills them better?")
    L.append("")

    caveat = ec.sample_size_caveat(plain["n_answerable"])
    if caveat:
        L += ["> " + caveat, ""]

    L += ["## The headline: catching out-of-scope questions", ""]
    L.append("`config.LOW_CONFIDENCE_THRESHOLD` was measured and documented as unable "
             "to do this job -- 14 of 15 unanswerable questions score above it, because "
             "cosine similarity measures whether a passage is *about* a question, not "
             "whether it *answers* it. CRAG exists to replace that judgement with one "
             "made by a model that reads the passage.")
    L.append("")
    L += [f"| | Plain RAG | CRAG |", "|---|---|---|"]
    L.append(f"| Correctly refused ({plain['n_unanswerable']} unanswerable) | "
             f"{ec.format_ci(plain['correct_refusal'])} | "
             f"{ec.format_ci(crag['correct_refusal'])} |")
    L.append(f"| Wrongly refused (answerable) | "
             f"{ec.format_ci(plain['over_refusal'])} | "
             f"{ec.format_ci(crag['over_refusal'])} |")
    L.append("")
    L.append("Both directions, because a system that refuses everything scores "
             "perfectly on the first row and terribly on the second.")
    L.append("")
    for name, s in (("Plain RAG", plain), ("CRAG", crag)):
        if s["answered_unanswerable"]:
            L.append(f"- **{name}** answered anyway: "
                     f"{', '.join('`' + i + '`' for i in s['answered_unanswerable'])}")
        if s["wrongly_refused"]:
            L.append(f"- **{name}** refused a question it could have answered: "
                     f"{', '.join('`' + i + '`' for i in s['wrongly_refused'])}")
    L.append("")

    L += ["## Retrieval, on what the model was given", "",
          "| Metric | Plain RAG | CRAG |", "|---|---|---|"]
    for label, key, pct in ((f"Hit Rate@{top_k}", f"hit@{top_k}", True),
                            ("MRR", "mrr", False),
                            ("Paper coverage", "coverage", True),
                            ("Chunks sent to the model", "sent_chunks", False)):
        L.append(f"| {label} | {ec.format_ci(plain[key], pct=pct)} | "
                 f"{ec.format_ci(crag[key], pct=pct)} |")
    L.append("")
    L.append("Stage 1 CRAG can only drop chunks or look further down the same ranked "
             "list -- it cannot invent a better query. Large movement here was not "
             "expected and its absence is not a failure; it is the boundary between "
             "Stage 1 and query rewriting.")
    L.append("")

    L += ["## Answers", "", "| Metric | Plain RAG | CRAG |", "|---|---|---|"]
    L.append(f"| Required facts present | {ec.format_ci(plain['fact_coverage'])} | "
             f"{ec.format_ci(crag['fact_coverage'])} |")
    L.append("")

    L += ["## Is the grader any good?", ""]
    L.append(f"Judged against labels already in the test set, over "
             f"{gq['n_chunks_graded']} graded chunks.")
    L.append("")
    L += ["| Check | Result |", "|---|---|"]
    if gq["unanswerable_rejected"]:
        L.append(f"| Chunks for *unanswerable* questions marked irrelevant | "
                 f"{ec.format_ci(gq['unanswerable_rejected'])} |")
    if gq["wrong_paper_rejected"]:
        L.append(f"| Chunks from a *wrong* paper marked irrelevant | "
                 f"{ec.format_ci(gq['wrong_paper_rejected'])} |")
    if gq["right_paper_accepted"]:
        L.append(f"| Chunks from a *correct* paper kept | "
                 f"{ec.format_ci(gq['right_paper_accepted'])} |")
    L.append("")
    L.append("The first two rows are the meaningful ones. A chunk from a paper that "
             "cannot answer the question almost certainly does not contain the answer, "
             "so the grader should reject it. The third row is looser: a chunk from the "
             "right paper might be that paper's reference list, and marking it "
             "irrelevant is correct. A low number there is not automatically an error.")
    L.append("")

    L += ["## What the correction cost", "",
          "| | Plain RAG | CRAG |", "|---|---|---|"]
    L.append(f"| API calls per question | {ec.format_ci(plain['api_calls'], pct=False)} | "
             f"{ec.format_ci(crag['api_calls'], pct=False)} |")
    L.append(f"| Input tokens per question | "
             f"{plain['input_tokens']['value']:.0f} | "
             f"{crag['input_tokens']['value']:.0f} |")
    L.append(f"| Latency (s) | {ec.format_ci(plain['latency_s'], pct=False)} | "
             f"{ec.format_ci(crag['latency_s'], pct=False)} |")
    L.append("")
    ratio = (crag["input_tokens"]["value"] / plain["input_tokens"]["value"]
             if plain["input_tokens"]["value"] else 0)
    L.append(f"CRAG costs **{ratio:.1f}x** the input tokens. It deepened on "
             f"{crag['deepened']} question(s) and refused on "
             f"{crag['decisions']['refuse']}, and a refusal skips generation entirely "
             f"-- so the extra grading call is partly paid back on exactly the "
             f"questions plain RAG would have spent a generation call getting wrong.")
    if crag["grader_failures"]:
        L.append("")
        L.append(f"The grader failed to return usable JSON on "
                 f"{crag['grader_failures']} question(s) and fell back to plain-RAG "
                 f"behaviour, which is the intended failure mode.")
    L.append("")

    L += ["## Per question", "",
          "| id | type | plain | CRAG | CRAG decision |", "|---|---|---|---|---|"]
    by_id = {r["id"]: r for r in results["crag"]}
    for p in results["plain"]:
        c = by_id[p["id"]]
        pf = "-" if p["fact_frac"] is None else f"{p['fact_frac']:.2f}"
        cf = "-" if c["fact_frac"] is None else f"{c['fact_frac']:.2f}"
        note = c["decision"] + (" (deepened)" if c.get("deepened") else "")
        L.append(f"| `{p['id']}` | {p['type']} | {pf} | {cf} | {note} |")
    L.append("")

    path = config.PROJECT_ROOT / "eval" / "crag_vs_rag.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def print_summary(summaries, gq, top_k):
    plain, crag = summaries["plain"], summaries["crag"]
    print()
    print(f"{'':<28}{'plain RAG':>20}{'CRAG':>20}")
    rows = [("correct refusal", "correct_refusal", True),
            ("over-refusal", "over_refusal", True),
            (f"hit@{top_k} (sent context)", f"hit@{top_k}", True),
            ("paper coverage", "coverage", True),
            ("required facts", "fact_coverage", True),
            ("api calls", "api_calls", False),
            ("latency (s)", "latency_s", False)]
    for label, key, pct in rows:
        print(f"{label:<28}{ec.format_ci(plain[key], pct=pct):>20}"
              f"{ec.format_ci(crag[key], pct=pct):>20}")
    print(f"{'input tokens':<28}{plain['input_tokens']['value']:>20.0f}"
          f"{crag['input_tokens']['value']:>20.0f}")
    print()
    print(f"  CRAG decisions : {crag['decisions']}, deepened on {crag['deepened']}")
    if gq["unanswerable_rejected"]:
        print(f"  grader rejects unanswerable chunks : "
              f"{ec.format_ci(gq['unanswerable_rejected'])}")
    if gq["wrong_paper_rejected"]:
        print(f"  grader rejects wrong-paper chunks  : "
              f"{ec.format_ci(gq['wrong_paper_rejected'])}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare plain RAG against Corrective RAG on the labelled set.")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--pause", type=float, default=config.API_MIN_PAUSE_SECONDS)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N questions, for a cheap smoke test")
    args = parser.parse_args()

    questions = ec.load_questions()
    if args.limit:
        questions = questions[:args.limit]

    print(f"Running {len(questions)} questions through both arms "
          f"(~{len(questions) * 3} API calls) ...")
    results = run(questions, args.top_k, args.pause)

    summaries = {name: summarise_arm(recs, args.top_k)
                 for name, recs in results.items()}
    gq = grader_quality(results["crag"])

    ec.update_summary("crag_vs_rag", {
        "top_k": args.top_k,
        "plain": {k: summaries["plain"][k] for k in
                  (f"hit@{args.top_k}", "mrr", "coverage", "fact_coverage",
                   "correct_refusal", "over_refusal", "api_calls", "latency_s")},
        "crag": {k: summaries["crag"][k] for k in
                 (f"hit@{args.top_k}", "mrr", "coverage", "fact_coverage",
                  "correct_refusal", "over_refusal", "api_calls", "latency_s")},
        "plain_input_tokens": summaries["plain"]["input_tokens"]["value"],
        "crag_input_tokens": summaries["crag"]["input_tokens"]["value"],
        "crag_decisions": summaries["crag"]["decisions"],
        "crag_deepened": summaries["crag"]["deepened"],
        "grader": {k: v for k, v in gq.items() if k != "n_chunks_graded"},
        "n_chunks_graded": gq["n_chunks_graded"],
    })

    path = write_report(results, summaries, gq, args.top_k)
    print_summary(summaries, gq, args.top_k)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
