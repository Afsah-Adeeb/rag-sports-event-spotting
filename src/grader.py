"""
The corrective step: decide whether retrieved chunks actually answer the question.

WHY THIS EXISTS
-----------------
Plain RAG hands whatever retrieval returned straight to the model. If four of
the five chunks are junk, the model still sees all five, and the only defence
is a line in the prompt asking it to notice. Nothing inspects the chunks, and
nothing is recorded about whether they were any good.

Corrective RAG (CRAG) inserts a grading step between retrieval and generation,
then branches on the result: answer from the good chunks, search deeper, or
decline. This module is the grading half -- the "how do I judge a chunk"
logic. crag.py is the branching half.

WHY AN LLM GRADER AND NOT A SIMILARITY CUTOFF
-----------------------------------------------
The cheap version is "if the cosine score is below X, call it irrelevant".
This project already measured that and it does not work here.

config.LOW_CONFIDENCE_THRESHOLD was derived from real score distributions, and
the evaluation suite then showed that 14 of 15 questions the corpus genuinely
CANNOT answer still scored above it -- the highest at 0.706, while real
answerable questions drop as low as 0.332. The two groups overlap completely.

The reason is not a badly-chosen threshold. Cosine similarity measures whether
a passage is *about* the question. It has no way to express whether the
passage *answers* it, and a chunk can be squarely on-topic while containing
nothing that resolves the question. That distinction is exactly what a model
reading the passage can make and a distance metric cannot.

So the grader reads. The prompt calls that distinction out explicitly, because
it is the whole point of the component.

COST
------
All chunks are graded in ONE call, not one call each. Five separate calls per
question would quintuple the cost of every query for no extra signal, and the
free tier allows 500 requests a day. The trade is that the grader sees all
passages together, so position within the list could bias it -- worth knowing,
and worth re-checking if the grades ever look suspiciously ordered.

FAILING OPEN, NOT CLOSED
--------------------------
If the API errors, or the reply cannot be parsed, every chunk is graded
`relevant` and the system behaves exactly like plain RAG. The opposite default
-- treating an unparseable reply as "nothing is relevant" -- would turn a
transient JSON hiccup into a refusal to answer a perfectly good question.
A broken grader should cost you the correction, not the answer.
"""

import json
import re
import time

import config
from generate import _get_client

# The three grades. Ordered from most to least useful, which `rank()` relies on.
RELEVANT = "relevant"
PARTIAL = "partial"
IRRELEVANT = "irrelevant"
LABELS = (RELEVANT, PARTIAL, IRRELEVANT)

_ORDER = {RELEVANT: 0, PARTIAL: 1, IRRELEVANT: 2}


def build_grading_prompt(question, chunks):
    """Ask for a grade per chunk, in one call, as JSON.

    The instruction to separate "about the topic" from "contains the answer"
    is the load-bearing line -- it is the judgement a similarity score cannot
    make, and the only reason this component earns its API call.
    """
    passages = "\n\n".join(
        f"PASSAGE {i}:\n{chunk['text']}" for i, chunk in enumerate(chunks, start=1)
    )

    return f"""You are grading retrieved passages for a question-answering system.

For each numbered passage, judge how much it helps answer the QUESTION:

- "relevant"   -- contains information that directly helps answer the question
- "partial"    -- related and mildly useful, but does not contain the answer
- "irrelevant" -- does not help answer the question

Judge only what each passage actually contains. A passage can be about exactly \
the right topic and still be irrelevant, because being on-topic is not the same \
as containing the answer. Do not use outside knowledge, and do not reward a \
passage for merely mentioning words from the question.

QUESTION:
{question}

{passages}

Reply with JSON and nothing else:
{{"grades": [{{"passage": 1, "label": "relevant", "why": "short reason"}}, ...]}}

Include exactly one entry for each of the {len(chunks)} passages.
"""


def _parse_grades(text, n_chunks):
    """Pull one label per chunk out of the reply, tolerating fences and prose.

    Anything missing or unrecognised becomes `relevant` -- see the module
    docstring on failing open. Returns (labels, reasons, parsed_ok).
    """
    labels = [RELEVANT] * n_chunks
    reasons = [""] * n_chunks

    if not text:
        return labels, reasons, False

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return labels, reasons, False
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return labels, reasons, False

    entries = data.get("grades")
    if not isinstance(entries, list):
        return labels, reasons, False

    seen = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            position = int(entry.get("passage", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= position < n_chunks:
            continue
        label = str(entry.get("label", "")).strip().lower()
        if label not in LABELS:
            continue
        labels[position] = label
        reasons[position] = str(entry.get("why", ""))[:200]
        seen += 1

    return labels, reasons, seen > 0


def grade_chunks(question, chunks, model=None):
    """Grade every chunk against the question in a single API call.

    Returns a dict with the labels, the per-chunk reasons, timing, token
    usage, and whether the reply parsed. The caller (crag.py) decides what to
    do with the grades -- this function only judges, it never branches.
    """
    if not chunks:
        return {
            "labels": [], "reasons": [], "parsed": True,
            "grading_ms": 0.0, "grading_input_tokens": 0,
            "grading_output_tokens": 0, "graded": False,
        }

    prompt = build_grading_prompt(question, chunks)
    started = time.perf_counter()

    try:
        response = _get_client().models.generate_content(
            model=model or config.BENCHMARK_MODEL_NAME, contents=prompt)
        text = response.text
        usage = getattr(response, "usage_metadata", None)
        in_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        out_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    except Exception:  # noqa: BLE001
        # Fail open: behave like plain RAG rather than refusing a good question
        # because the grader had a bad minute.
        return {
            "labels": [RELEVANT] * len(chunks), "reasons": [""] * len(chunks),
            "parsed": False, "grading_ms": (time.perf_counter() - started) * 1000,
            "grading_input_tokens": None, "grading_output_tokens": None,
            "graded": False,
        }

    labels, reasons, parsed = _parse_grades(text, len(chunks))
    return {
        "labels": labels,
        "reasons": reasons,
        "parsed": parsed,
        "grading_ms": (time.perf_counter() - started) * 1000,
        "grading_input_tokens": in_tokens,
        "grading_output_tokens": out_tokens,
        "graded": True,
    }


# --- Using the grades --------------------------------------------------------

def count_labels(labels):
    """How many chunks got each grade."""
    return {label: labels.count(label) for label in LABELS}


def keep_useful(chunks, labels):
    """Drop the irrelevant chunks, keeping the original ranking order.

    Partials are kept. They do not contain the answer, but they carry context
    the model can use to frame one, and dropping everything but `relevant`
    was measurably too aggressive on questions whose answer is spread thin.
    """
    kept = [c for c, label in zip(chunks, labels) if label != IRRELEVANT]
    return kept or list(chunks)  # never hand the model an empty context


def rank(labels):
    """Sort positions best-graded first, keeping retrieval order within a grade."""
    return sorted(range(len(labels)), key=lambda i: (_ORDER[labels[i]], i))


def decide(labels, already_retried, min_relevant=None):
    """Choose the next action from the grades. Pure function, no API calls.

    Three outcomes:
      "answer"  -- enough relevant material; generate from what survived
      "deepen"  -- too little; search further down the same ranked list and
                   re-grade. Costs nothing to retrieve (it is a local FAISS
                   search) and one call to re-grade.
      "refuse"  -- nothing relevant even after deepening; decline instead of
                   answering from material the grader just called irrelevant

    `deepen` is deliberately NOT query rewriting. Rewriting is the Stage 2
    behaviour and costs an extra model call to produce a new question. Going
    deeper reuses the same query against the same index and is free, so it is
    the cheapest correction available and the right one to try first.
    """
    min_relevant = config.CRAG_MIN_RELEVANT if min_relevant is None else min_relevant
    counts = count_labels(labels)

    if counts[RELEVANT] >= min_relevant:
        return "answer"
    if not already_retried:
        return "deepen"
    # Second pass and still short. Answer if anything at all survived the
    # grading; refuse only when every chunk was called irrelevant.
    if counts[RELEVANT] or counts[PARTIAL]:
        return "answer"
    return "refuse"


REFUSAL_TEXT = (
    "I could not find anything in these papers that answers that question. "
    "The passages that came back are either about a different topic or mention "
    "it without covering the specific detail you asked for."
)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(config.PROJECT_ROOT / "src"))
    from retrieve import retrieve

    question = " ".join(sys.argv[1:]) or "What is the E2E-Spot architecture?"
    found = retrieve(question, top_k=config.DEFAULT_TOP_K)

    print(f"Question: {question}\n")
    result = grade_chunks(question, found)
    for chunk, label, why in zip(found, result["labels"], result["reasons"]):
        print(f"  [{label:<10}] {chunk['source_paper'][:52]} "
              f"(p.{chunk['page_start']}, score={chunk['score']:.3f})")
        if why:
            print(f"               {why}")

    print(f"\n  counts   : {count_labels(result['labels'])}")
    print(f"  decision : {decide(result['labels'], already_retried=False)}")
    print(f"  kept     : {len(keep_useful(found, result['labels']))}/{len(found)} chunks")
    print(f"  cost     : {result['grading_ms']:.0f}ms, "
          f"{result['grading_input_tokens']} in / "
          f"{result['grading_output_tokens']} out")
