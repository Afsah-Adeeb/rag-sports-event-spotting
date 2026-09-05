"""
Benchmark: semantic RAG vs agentic RAG on the same questions.

THE EXPERIMENT
----------------
Two retrieval strategies, same corpus, same questions, same model, same
ground-truth labels:

  Semantic (retrieve.py + generate.py)
      Pre-embed every chunk. Embed the question. Return the k nearest.
      One retrieval step, decided by vector similarity.

  Agentic (agent_tools.py + agent_rag.py)
      No index at all. Give the model grep/list/read and let it search,
      read results, and decide what to search next until it can answer.

The claim usually made about agentic retrieval is that it is better for code
and exact identifiers but weaker on prose, and that it costs several times
more. This measures whether that holds on a corpus of research papers.

WHAT IS HELD CONSTANT, AND WHY IT MATTERS
-------------------------------------------
A benchmark is only worth running if the difference it reports is caused by
the thing being tested. Held fixed here:

  - **The same model** for both arms (config.BENCHMARK_MODEL_NAME). Comparing
    two retrieval strategies while also varying the model would measure the
    models.
  - **The same text.** export_text.py reuses ingest.py's extraction,
    boilerplate stripping, and paragraph joining, so neither side gets a
    cleaner copy of the papers than the other.
  - **The same questions and the same ground truth** (eval/test_questions.json,
    the file evaluate.py already uses), so retrieval quality is scored by one
    definition of correct.

WHAT IS MEASURED
------------------
  Hit Rate  -- did the correct paper appear in what was retrieved? This is the
               one metric that is directly comparable, because "retrieved" is
               definable on both sides: the top-k chunks' source papers for
               semantic, the papers actually grepped or read for agentic.
  Latency   -- wall clock per question.
  Tokens    -- input + output, summed across every API call a question needed.
               This is where the two differ most: agentic resends the whole
               conversation, including every tool result, on every turn.
  Turns     -- API round trips per question. Always 1 for semantic.

Precision@k is deliberately NOT reported. Semantic retrieval returns exactly
k chunks, so precision is well-defined; agentic retrieval returns however
many papers the model chose to look at, so the same formula would mean
something different for each arm. Reporting one number computed two ways
would look rigorous and be meaningless.

FAITHFULNESS IS STILL NOT AUTOMATED
-------------------------------------
Same position as evaluate.py: no LLM-as-judge. The report pairs every
question with both answers side by side for manual review.

    cd src
    ../.venv/Scripts/python.exe compare_rag.py
"""

import json
import time
from statistics import fmean, median

# This script lives in a subfolder but imports the pipeline modules that sit in
# src/ (config, retrieve, generate, telemetry). Running a script puts its OWN
# folder on the import path, not its parent, so the parent is added explicitly.
# Siblings inside this folder import normally.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import config
from agent_rag import DailyQuotaExhausted, agentic_answer
from generate import answer_traced


def load_questions():
    with open(config.TEST_QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_semantic(question, model):
    """One semantic-RAG answer, normalised to the shared result shape."""
    result = answer_traced(question, top_k=config.DEFAULT_TOP_K, model=model)
    return {
        "answer": result["answer"],
        "papers_used": sorted({c["source_paper"] for c in result["chunks"]}),
        "total_ms": result["retrieval_ms"] + result["generation_ms"],
        "input_tokens": result.get("input_tokens") or 0,
        "output_tokens": result.get("output_tokens") or 0,
        "turns": 1,  # one API call, always
        "detail": [
            f"{c['source_paper']} p.{c['page_start']}-{c['page_end']} ({c['score']:.3f})"
            for c in result["chunks"]
        ],
    }


def run_agentic(question, model):
    """One agentic-RAG answer, normalised to the shared result shape."""
    result = agentic_answer(question, model=model)
    return {
        "answer": result["answer"],
        "papers_used": result["papers_used"],
        "total_ms": result["total_ms"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "turns": result["turns"],
        "hit_cap": result["hit_cap"],
        "detail": [
            f"{c['tool']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})"
            f" -> {c['chars']} chars"
            for c in result["tool_calls"]
        ],
    }


def hit(papers_used, correct_papers):
    """Did retrieval surface at least one correct source paper?"""
    return int(any(p in correct_papers for p in papers_used))


def summarise(runs):
    """Aggregate one arm's per-question results."""
    if not runs:
        return None
    return {
        "questions": len(runs),
        "hit_rate": fmean(r["hit"] for r in runs),
        "median_ms": median(r["total_ms"] for r in runs),
        "mean_input_tokens": fmean(r["input_tokens"] for r in runs),
        "mean_output_tokens": fmean(r["output_tokens"] for r in runs),
        "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in runs),
        "mean_turns": fmean(r["turns"] for r in runs),
    }


def benchmark(model=None, pause=0.0):
    """Run both arms over every labelled question."""
    model = model or config.BENCHMARK_MODEL_NAME
    questions = load_questions()

    print(f"Benchmarking {len(questions)} questions on {model}")
    print(f"  semantic: top-{config.DEFAULT_TOP_K} vector retrieval")
    print(f"  agentic : grep/read loop, max {config.AGENT_MAX_TURNS} turns\n")

    rows = []
    for i, q in enumerate(questions, start=1):
        question, correct = q["question"], q["correct_papers"]
        print(f"[{i}/{len(questions)}] {question[:64]}")

        row = {"question": question, "correct_papers": correct}
        for arm, runner in (("semantic", run_semantic), ("agentic", run_agentic)):
            try:
                result = runner(question, model)
            except DailyQuotaExhausted as exc:
                # Partial results are still worth reporting -- stop cleanly and
                # write up what completed rather than losing the whole run.
                print(f"\n  STOPPED: {exc}")
                print(f"  Reporting the {len(rows)} question(s) completed so far.\n")
                return rows, model
            result["hit"] = hit(result["papers_used"], correct)
            row[arm] = result
            print(f"    {arm:9s} hit={result['hit']}  {result['total_ms'] / 1000:5.1f}s  "
                  f"{result['input_tokens']:>7,} in / {result['output_tokens']:>4,} out  "
                  f"turns={result['turns']}")
            if pause:
                time.sleep(pause)
        rows.append(row)

    return rows, model


def _sample_size_caveat(n, semantic, agentic):
    """State plainly how much the hit-rate comparison can actually support.

    Cost and latency ratios are stable even on a handful of questions -- they
    are averages of a per-question quantity that barely varies. Hit Rate is a
    proportion over n questions, so with n small its confidence interval spans
    most of the range and a tie means "not enough evidence", not "equal". A
    report that prints 1.00 vs 1.00 without saying so invites exactly the wrong
    conclusion.
    """
    if n >= 15:
        return (f"Based on {n} questions — enough for the Hit Rate comparison to carry "
                "some weight, though still a small sample.")

    tied = abs(semantic["hit_rate"] - agentic["hit_rate"]) < 1e-9
    verdict = (
        "Both arms scored identically, which at this sample size means the questions "
        "did not separate them — not that the approaches are equivalent."
        if tied else
        "The Hit Rate gap here is well within what chance produces at this sample size."
    )
    return (
        f"> **Sample size warning: only {n} question(s).** {verdict} Roughly 15-20 "
        "labelled questions are needed before Hit Rate is worth quoting. The token, "
        "latency, and round-trip ratios are far more trustworthy at this n, because "
        "they average a per-question cost that varies little rather than a "
        "pass/fail proportion."
    )


def write_report(rows, model):
    """Write the side-by-side comparison, including answers for manual review."""
    semantic = summarise([r["semantic"] for r in rows if "semantic" in r])
    agentic = summarise([r["agentic"] for r in rows if "agentic" in r])
    if not semantic or not agentic:
        print("Nothing completed; no report written.")
        return None

    lines = [
        "# Semantic RAG vs Agentic RAG",
        "",
        f"Corpus: {len(list(config.PAPERS_DIR.glob('*.pdf')))} papers. "
        f"Questions: {len(rows)}. Model (both arms): `{model}`.",
        "",
        "**Semantic** pre-embeds every chunk and returns the "
        f"top-{config.DEFAULT_TOP_K} nearest to the question, in one step. "
        "**Agentic** has no index and searches the papers with grep/read tools "
        "in a loop, deciding what to look at next.",
        "",
        "## Results",
        "",
        "| Metric | Semantic | Agentic | Ratio |",
        "|---|---|---|---|",
        f"| Hit Rate (correct paper retrieved) | **{semantic['hit_rate']:.2f}** | "
        f"**{agentic['hit_rate']:.2f}** | — |",
        f"| Median latency | {semantic['median_ms'] / 1000:.1f}s | "
        f"{agentic['median_ms'] / 1000:.1f}s | "
        f"{agentic['median_ms'] / max(semantic['median_ms'], 1e-9):.1f}x |",
        f"| Mean input tokens | {semantic['mean_input_tokens']:,.0f} | "
        f"{agentic['mean_input_tokens']:,.0f} | "
        f"{agentic['mean_input_tokens'] / max(semantic['mean_input_tokens'], 1e-9):.1f}x |",
        f"| Mean output tokens | {semantic['mean_output_tokens']:,.0f} | "
        f"{agentic['mean_output_tokens']:,.0f} | "
        f"{agentic['mean_output_tokens'] / max(semantic['mean_output_tokens'], 1e-9):.1f}x |",
        f"| Total tokens (all questions) | {semantic['total_tokens']:,} | "
        f"{agentic['total_tokens']:,} | "
        f"{agentic['total_tokens'] / max(semantic['total_tokens'], 1e-9):.1f}x |",
        f"| Mean API round trips | {semantic['mean_turns']:.1f} | "
        f"{agentic['mean_turns']:.1f} | "
        f"{agentic['mean_turns'] / max(semantic['mean_turns'], 1e-9):.1f}x |",
        "",
        "**Hit Rate** = fraction of questions where at least one correct source paper "
        "appeared in what was retrieved (top-k chunk sources for semantic; papers "
        "actually grepped or read for agentic).",
        "",
        "Precision@k is not reported: semantic returns exactly k chunks so precision "
        "is well-defined, while agentic returns however many papers the model chose to "
        "open. The same formula would mean different things on each side.",
        "",
        _sample_size_caveat(len(rows), semantic, agentic),
        "",
        "## Per-question detail",
        "",
        "Both answers are shown for manual faithfulness review — as in `evaluate.py`, "
        "answer quality is deliberately not auto-scored.",
        "",
    ]

    for i, row in enumerate(rows, start=1):
        s, a = row["semantic"], row["agentic"]
        lines += [
            f"### Q{i}: {row['question']}",
            "",
            f"**Correct paper(s):** {', '.join(row['correct_papers'])}",
            "",
            f"| | Semantic | Agentic |",
            "|---|---|---|",
            f"| Found correct paper | {'yes' if s['hit'] else 'NO'} | "
            f"{'yes' if a['hit'] else 'NO'} |",
            f"| Latency | {s['total_ms'] / 1000:.1f}s | {a['total_ms'] / 1000:.1f}s |",
            f"| Tokens (in/out) | {s['input_tokens']:,}/{s['output_tokens']:,} | "
            f"{a['input_tokens']:,}/{a['output_tokens']:,} |",
            f"| API round trips | {s['turns']} | {a['turns']}"
            f"{' (hit cap)' if a.get('hit_cap') else ''} |",
            "",
            "<details><summary>What each retrieved</summary>",
            "",
            "*Semantic — top-k chunks:*",
            "",
        ]
        lines += [f"- {d}" for d in s["detail"]]
        lines += ["", "*Agentic — tool calls made:*", ""]
        lines += [f"- {d}" for d in a["detail"]] or ["- (none)"]
        lines += [
            "",
            "</details>",
            "",
            f"**Semantic answer:**\n\n{s['answer']}",
            "",
            f"**Agentic answer:**\n\n{a['answer']}",
            "",
            "**Faithfulness check (fill in manually):** "
            "semantic [ ] faithful [ ] unsupported · "
            "agentic [ ] faithful [ ] unsupported",
            "",
            "---",
            "",
        ]

    config.COMPARISON_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.COMPARISON_RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    return semantic, agentic


def main():
    rows, model = benchmark()
    if not rows:
        return
    result = write_report(rows, model)
    if not result:
        return
    semantic, agentic = result

    print(f"\n{'':12s} {'semantic':>12s} {'agentic':>12s}")
    print(f"{'hit rate':12s} {semantic['hit_rate']:>12.2f} {agentic['hit_rate']:>12.2f}")
    print(f"{'median s':12s} {semantic['median_ms'] / 1000:>12.1f} "
          f"{agentic['median_ms'] / 1000:>12.1f}")
    print(f"{'in tokens':12s} {semantic['mean_input_tokens']:>12,.0f} "
          f"{agentic['mean_input_tokens']:>12,.0f}")
    print(f"{'round trips':12s} {semantic['mean_turns']:>12.1f} "
          f"{agentic['mean_turns']:>12.1f}")
    print(f"\nFull report written to {config.COMPARISON_RESULTS_PATH}")


if __name__ == "__main__":
    main()
