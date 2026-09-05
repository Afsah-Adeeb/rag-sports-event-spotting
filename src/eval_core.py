"""
The measuring tape: metric primitives shared by every evaluation script.

WHY THIS FILE EXISTS
----------------------
evaluate.py used to compute its own metrics inline. That was fine while
there was one evaluation. There are now several planned -- baselines,
a no-retrieval control, a refusal test, a settings sweep -- and every one
of them has to score things *identically* or the comparisons between them
are meaningless. So the scoring lives here, once, and the scripts differ
only in what they feed it.

WHAT IS MEASURED, AND WHY EACH ONE
------------------------------------
Retrieval, given a labelled set of correct source papers per question:

  - Hit Rate@k    : did ANY of the top-k chunks come from a correct paper?
                    "Did we find it at all."
  - Precision@k   : what fraction of the top-k came from a correct paper?
                    "How much noise came along."
  - Reciprocal Rank / MRR : 1/(position of the first correct chunk).
                    "WHERE did we find it." Hit Rate@5 scores a correct
                    paper at position 1 and position 5 identically, which
                    hides the case where the system is scraping in at the
                    bottom and would break if top_k were lowered. MRR is
                    the metric that exposes that.

Answers, given hand-written required facts per question:

  - Fact coverage : what fraction of the required facts appear in the
                    answer. This is a simplified form of the "nugget"
                    method TREC uses -- extract the facts that matter,
                    check the answer against each one -- with their
                    vital/okay split kept as must_mention/nice_to_mention.

WHY EACH FACT IS A LIST OF SPELLINGS
--------------------------------------
A fact is stored as ["Figure Skating", "FigureSkating"] and counts as
present if ANY spelling appears. This is not defensive over-engineering:
this corpus already produced a real failure of exactly this kind, where
searching for "temporal discriminability" found nothing in the T-DEED
paper because that paper only ever writes "Temporal-Discriminability".
A single hyphen silently zeroed a score. Accepting a list of forms per
fact makes the metric measure the answer rather than the punctuation.

TWO DIFFERENT CONFIDENCE INTERVALS, ON PURPOSE
------------------------------------------------
A score with no error bar is not a result, it is an anecdote -- especially
at 40 questions. But the right interval depends on what is being measured:

  - Proportions (hit rate, refusal rate) use a WILSON interval. The obvious
    choice, bootstrapping, breaks badly here: if all 40 questions are hits,
    every resample is also all hits, so the bootstrap reports [1.00, 1.00]
    -- perfect certainty from 40 data points, which is nonsense. Wilson is
    a closed-form interval that stays sensibly wide at the boundaries.
  - Means of continuous values (precision@k, MRR, latency) use a BOOTSTRAP,
    which makes no assumption about the shape of the distribution. These
    metrics are not proportions and are often not normally distributed.

The bootstrap is seeded, so re-running the same evaluation twice reports
the same interval. An error bar that wobbles between runs is worse than
no error bar, because it invites re-rolling until the number looks good.
"""

import json
import math
import random
import time
from collections import defaultdict

import config

# Re-exported rather than redefined. compare_rag.py already catches this exact
# class, and a second class with the same name in another module would not be
# caught by that `except` -- a benchmark would die on a quota error it thinks
# it handles.
from agent_rag import DailyQuotaExhausted  # noqa: F401

# Fixed so intervals are reproducible across runs. See module docstring.
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_RESAMPLES = 2000
Z_95 = 1.959963985  # two-sided 95% normal quantile


# --- Loading -----------------------------------------------------------------

def load_questions(path=None, include_unanswerable=True):
    """Read the labelled test set.

    The file has a leading "_schema" key documenting the format for whoever
    edits it by hand; the questions themselves live under "questions".
    """
    path = path or config.TEST_QUESTIONS_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"] if isinstance(data, dict) else data
    if not include_unanswerable:
        questions = [q for q in questions if q.get("answerable", True)]
    return questions


def answerable(questions):
    return [q for q in questions if q.get("answerable", True)]


def unanswerable(questions):
    return [q for q in questions if not q.get("answerable", True)]


# --- Retrieval metrics -------------------------------------------------------

def hit_at_k(retrieved_chunks, correct_papers, k):
    """1 if any of the top-k chunks came from a correct paper, else 0."""
    return int(any(c["source_paper"] in correct_papers for c in retrieved_chunks[:k]))


def precision_at_k(retrieved_chunks, correct_papers, k):
    """Fraction of the top-k chunks that came from a correct paper.

    Divided by k rather than by len(retrieved), so a system that returns
    fewer than k chunks is penalised for the shortfall instead of being
    quietly rewarded for it.
    """
    if k == 0:
        return 0.0
    relevant = sum(1 for c in retrieved_chunks[:k] if c["source_paper"] in correct_papers)
    return relevant / k


def first_correct_rank(retrieved_chunks, correct_papers):
    """1-based position of the first correct chunk, or None if there is none."""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        if chunk["source_paper"] in correct_papers:
            return i
    return None


def reciprocal_rank(retrieved_chunks, correct_papers):
    """1/rank of the first correct chunk; 0.0 if no correct chunk was retrieved."""
    rank = first_correct_rank(retrieved_chunks, correct_papers)
    return 0.0 if rank is None else 1.0 / rank


def papers_covered(retrieved_chunks, correct_papers):
    """How many of the correct papers are represented at all in the results.

    Matters for the multi_paper questions, where the right answer needs
    three or four different papers present. Hit Rate@k scores those as a
    full success the moment ONE correct paper shows up, which overstates
    how well retrieval did on exactly the questions it finds hardest.
    """
    if not correct_papers:
        return 0.0
    found = {c["source_paper"] for c in retrieved_chunks} & set(correct_papers)
    return len(found) / len(correct_papers)


def top_score(retrieved_chunks):
    """Best similarity score in the results, or 0.0 if nothing was retrieved."""
    return max((c.get("score", 0.0) for c in retrieved_chunks), default=0.0)


# --- Answer metrics ----------------------------------------------------------

def fact_present(answer, fact_spellings):
    """True if ANY accepted spelling of one fact appears in the answer."""
    lowered = (answer or "").lower()
    return any(str(spelling).lower() in lowered for spelling in fact_spellings)


def fact_coverage(answer, facts):
    """Fraction of required facts present in the answer, plus the raw counts.

    Returns (fraction, found, total). A question with no listed facts
    returns (None, 0, 0) rather than a misleading 1.0 -- "nothing was
    required" is not the same as "everything required was delivered", and
    averaging those in as perfect scores would inflate the headline number.
    """
    if not facts:
        return None, 0, 0
    found = sum(1 for fact in facts if fact_present(answer, fact))
    return found / len(facts), found, len(facts)


def missing_facts(answer, facts):
    """The facts that did NOT appear, for the per-question report.

    The aggregate number says how well it did; this says what it got wrong,
    which is the part that tells you what to fix.
    """
    return [fact for fact in facts if not fact_present(answer, fact)]


# --- Statistics --------------------------------------------------------------

def wilson_ci(successes, n, z=Z_95):
    """95% Wilson score interval for a proportion.

    Used instead of a bootstrap for pass/fail metrics because the bootstrap
    degenerates at the boundaries: 40 out of 40 hits resamples to 40 out of
    40 every time, reporting [1.00, 1.00]. Wilson stays honest there --
    40/40 gives roughly [0.91, 1.00], which is the correct amount of
    confidence to have from 40 observations.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_ci(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """95% percentile bootstrap interval for the mean of `values`.

    Resample the observations with replacement, take the mean, repeat, then
    read off the 2.5th and 97.5th percentiles. Makes no assumption about the
    distribution's shape, which matters because per-question precision and
    reciprocal rank are lumpy, not bell-shaped.
    """
    values = list(values)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (_percentile(means, 2.5), _percentile(means, 97.5))


def _percentile(sorted_values, pct):
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


def summarise_proportion(flags):
    """Aggregate a list of 0/1 (or bool) outcomes with a Wilson interval."""
    flags = [int(bool(f)) for f in flags]
    n = len(flags)
    successes = sum(flags)
    lo, hi = wilson_ci(successes, n)
    return {
        "n": n,
        "value": successes / n if n else 0.0,
        "lo": lo,
        "hi": hi,
        "kind": "proportion",
    }


def summarise_mean(values):
    """Aggregate a list of continuous values with a bootstrap interval."""
    values = [v for v in values if v is not None]
    n = len(values)
    if n == 0:
        return {"n": 0, "value": 0.0, "lo": 0.0, "hi": 0.0, "kind": "mean"}
    lo, hi = bootstrap_ci(values)
    return {
        "n": n,
        "value": sum(values) / n,
        "lo": lo,
        "hi": hi,
        "kind": "mean",
    }


def format_ci(summary, pct=True):
    """Render a summary dict as '0.85 [0.71-0.94]' for reports and terminals."""
    if summary["n"] == 0:
        return "n/a"
    if pct:
        return (f"{summary['value']:.2f} "
                f"[{summary['lo']:.2f}-{summary['hi']:.2f}]")
    return (f"{summary['value']:.3f} "
            f"[{summary['lo']:.3f}-{summary['hi']:.3f}]")


# --- Grouping ----------------------------------------------------------------

def group_by_type(records):
    """Bucket per-question records by their question type.

    The whole point of the type tags. An 85% average can be 100% on simple
    questions and 0% on multi-paper ones, and only the breakdown shows that.
    Aggregates hide the failure mode; this is where the findings live.
    """
    buckets = defaultdict(list)
    for record in records:
        buckets[record.get("type", "untyped")].append(record)
    return dict(buckets)


def sample_size_caveat(n, floor=15):
    """A warning line when n is too small for a proportion to mean anything.

    Kept here rather than written per-script so no report can quietly omit
    it. compare_rag.py already learned this lesson at n=3, where both arms
    scored 1.00 and the tie said nothing at all.
    """
    if n >= floor:
        return None
    return (f"**Sample size warning: only {n} question(s).** Proportions at this "
            f"size carry intervals so wide they cannot separate two systems. "
            f"Treat these as a smoke test, not a result.")


# --- Calling the model, safely -----------------------------------------------

def call_with_retry(fn, attempts=None, base_seconds=None, label=""):
    """Run one API call, retrying per-minute rate limits, failing fast on daily ones.

    Every evaluation script here makes dozens of sequential calls, so it will
    trip the free tier's 5-requests-per-minute limit sooner or later. That one
    is transient and worth sleeping through -- an unattended 55-question run
    only finishes if it backs off and carries on.

    The per-DAY quota is not transient. agent_rag.py learned this the
    expensive way: treating both 429s identically burned five escalating
    sleeps against a daily cap that had hours left on it. So they are told
    apart, and the daily one raises immediately.
    """
    attempts = attempts or config.API_RETRY_ATTEMPTS
    base_seconds = base_seconds or config.API_RETRY_BASE_SECONDS

    for attempt in range(attempts):
        try:
            return fn()
        except DailyQuotaExhausted:
            raise
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            is_quota = "429" in message or "RESOURCE_EXHAUSTED" in message
            # Transient server-side failures, which are NOT quota problems and
            # were not handled by the first version of this function: a
            # 110-call closed-book run died two thirds of the way through on a
            # single 503, wasting every request it had already paid for. The
            # API being briefly unavailable is exactly what a retry is for.
            is_transient = ("503" in message or "UNAVAILABLE" in message
                            or "500" in message or "INTERNAL" in message
                            or "504" in message or "DEADLINE_EXCEEDED" in message)

            if is_quota and "PerDay" in message:
                raise DailyQuotaExhausted(
                    f"Free-tier daily quota exhausted{' during ' + label if label else ''}. "
                    f"Wait for the reset, or point config.BENCHMARK_MODEL_NAME at "
                    f"another model."
                ) from exc
            if not (is_quota or is_transient) or attempt == attempts - 1:
                raise

            wait = base_seconds * (attempt + 1)
            reason = "rate limited" if is_quota else "server unavailable"
            print(f"    {reason}, waiting {wait}s ...", flush=True)
            time.sleep(wait)


# --- Caching generated answers ------------------------------------------------

def save_records(records, path):
    """Persist per-question records so a later script can reuse the answers.

    Generation is the only expensive part of this pipeline. Without a cache,
    every script that wants to analyse answers pays to regenerate them, and
    worse, analyses a DIFFERENT set of answers than the previous script did --
    so any disagreement between two reports could be the analysis or could
    just be resampling. Caching makes the comparison exact.

    Retrieved chunk text is dropped: it is large, and it is reproducible for
    free by re-running retrieval.
    """
    slim = []
    for r in records:
        keep = {k: v for k, v in r.items() if k != "retrieved"}
        keep["retrieved"] = [
            {"source_paper": c["source_paper"], "page_start": c["page_start"],
             "page_end": c["page_end"], "score": c.get("score", 0.0),
             "text": c["text"]}
            for c in r.get("retrieved", [])
        ]
        slim.append(keep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_records(path):
    """Read back a cache written by save_records, or None if it is not there."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- Shared headline summary --------------------------------------------------

SUMMARY_PATH = config.PROJECT_ROOT / "eval" / "summary.json"


def update_summary(section, data):
    """Merge one script's headline numbers into eval/summary.json.

    Each evaluation script owns one section and rewrites only its own. The
    app's Evaluation tab reads this file instead of parsing the markdown
    reports, so changing a report's wording cannot break the dashboard.

    Merged rather than overwritten because the scripts are run independently
    and at different times -- running the sweep must not erase the baseline
    numbers from an hour earlier.
    """
    try:
        current = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) \
            if SUMMARY_PATH.exists() else {}
    except (json.JSONDecodeError, OSError):
        current = {}
    current[section] = data
    current[section]["updated"] = time.strftime("%Y-%m-%d %H:%M")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(current, indent=1), encoding="utf-8")
    return SUMMARY_PATH


def read_summary():
    """Read eval/summary.json, or {} if no evaluation has been run yet."""
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return {}
