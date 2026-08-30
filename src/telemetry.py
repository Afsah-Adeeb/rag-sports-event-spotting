"""
Monitoring layer: record what the system actually did, so quality is
observable instead of assumed.

WHY THIS EXISTS
-----------------
evaluate.py measures retrieval quality *offline*, against a fixed set of
hand-labeled questions. That answers "is the system good?" once, on
questions I chose. It cannot answer:

  - How slow is it in practice, and *which stage* is slow?
  - What are real users asking, versus what I tested?
  - Did retrieval actually find anything relevant for those questions?
  - Did the model answer confidently even when retrieval found nothing?
  - Which papers in the corpus are never used?

Those are production questions, and they need a log of real traffic.
This module is that log.

DESIGN: AN APPEND-ONLY EVENT LOG, NOT A TABLE OF ROWS
-------------------------------------------------------
Every write is a single JSON line appended to one file. Nothing is ever
updated in place. Two kinds of events share the file:

    {"event": "query",    "query_id": "...", ...}   written when answered
    {"event": "feedback", "query_id": "...", ...}   written if the user rates it

Feedback arrives *after* the answer was already logged, so the obvious
design would be "find that row and set a rating column". Append-only is
better here for two reasons:

  1. Concurrency. Streamlit serves multiple sessions from one process, and
     several could write at once. Appending one short line is effectively
     atomic on both Windows and Linux; read-modify-rewrite of a whole file
     is a race that silently loses records.
  2. History. An event log keeps the fact that feedback came 30 seconds
     after the answer. Overwriting a column throws that away.

The cost is that reads have to reduce events into records -- that's what
build_records() does.

STORAGE HONESTY
-----------------
This writes to a local file, which is the right call for a portfolio
project but has a real limitation worth stating out loud: on Streamlit
Community Cloud the filesystem is ephemeral, so the log is wiped whenever
the app restarts or redeploys. Metrics there cover the current instance's
lifetime only, and the Metrics tab says so. Locally the file persists
normally. Swapping in a database means reimplementing _append() and
load_events() -- nothing else in the project touches the storage format.

FAILURES ARE SWALLOWED ON PURPOSE
-----------------------------------
Monitoring must never take down the thing it monitors. Every public
function here is wrapped so that a full disk, a permissions error, or a
corrupt line degrades the metrics rather than breaking a user's question.
"""

import json
import statistics
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone

import config

# Phrases that indicate the model declined to answer rather than guessing.
# generate.py's prompt explicitly instructs it to say so when the context is
# insufficient, so these are the shapes that instruction produces in practice.
#
# This is a heuristic, and it is worth being honest about that: it can miss a
# refusal phrased in a new way. It is used only to *triage* which queries a
# human should look at, never to score the system.
REFUSAL_MARKERS = (
    "does not contain",
    "doesn't contain",
    "does not provide",
    "doesn't provide",
    "does not include",
    "does not mention",
    "doesn't mention",
    "not enough information",
    "insufficient information",
    "no information",
    "no relevant information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "i don't know",
    "is not specified",
    "are not specified",
    "not detailed in the provided",
)

# Matching those phrases *anywhere* in the answer was the obvious first
# implementation, and it was wrong. Measured against real logged answers:
#
#   "The provided context does not contain information about the capital
#    of Brazil."                          marker at char 21, answer 78 chars  -> refusal
#   "I am sorry, but the provided context does not contain information
#    about how to bake sourdough bread."  marker at char 37, answer 211 chars -> refusal
#   "Optical flow is used in two-stream experiments... median flow is
#    subtracted, values clamped to [-20,+20]... The context does not
#    contain further details."            marker at char 248, answer 384 chars -> NOT a refusal
#
# The third one answers the question and then adds a closing caveat, but the
# naive check flagged it, which in turn produced a false "over-refusal" alert.
# A real refusal *leads* with the refusal; a caveat comes after the substance.
# So the position of the marker is the signal, not its presence.
REFUSAL_LEAD_CHARS = 150   # comfortably above the 37 seen, below the 248 seen
REFUSAL_SHORT_ANSWER = 250  # a short answer containing a marker is a refusal wherever it sits


def new_query_id():
    """Short unique id linking a query event to its later feedback event."""
    return uuid.uuid4().hex[:12]


def looks_like_refusal(answer):
    """True if the answer declines to answer, rather than answering with a caveat.

    Position-aware on purpose (see the note above REFUSAL_LEAD_CHARS): a
    refusal opens with the refusal, whereas a real answer that ends "...the
    context does not specify X" is still an answer and must not be flagged.

    Heuristic, used for triage only -- never to score the system.
    """
    lowered = (answer or "").lower()
    positions = [lowered.find(m) for m in REFUSAL_MARKERS if m in lowered]
    if not positions:
        return False
    if len(lowered) <= REFUSAL_SHORT_ANSWER:
        return True
    return min(positions) <= REFUSAL_LEAD_CHARS


def _append(record):
    """Append one JSON line. Never raises -- monitoring must not break the app."""
    if not config.TELEMETRY_ENABLED:
        return
    try:
        config.TELEMETRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        # One open/write/close per event, in append mode: the OS appends the
        # whole short line at the current end of file, so concurrent writers
        # interleave lines rather than corrupting each other.
        with open(config.TELEMETRY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring
        print(f"[telemetry] failed to write event: {exc}", file=sys.stderr)


def log_query(
    question,
    answer,
    retrieved_chunks,
    retrieval_ms,
    generation_ms,
    top_k,
    usage=None,
    surface="app",
    session_id=None,
    query_id=None,
):
    """Record one answered question. Returns the query_id used (for feedback)."""
    query_id = query_id or new_query_id()
    scores = [c["score"] for c in retrieved_chunks]
    top_score = max(scores) if scores else 0.0
    weak_retrieval = top_score < config.LOW_CONFIDENCE_THRESHOLD
    refused = looks_like_refusal(answer)

    _append({
        "event": "query",
        "query_id": query_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "surface": surface,          # "app" (Streamlit) or "cli"
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "answer_chars": len(answer or ""),
        "top_k": top_k,
        # --- retrieval quality signals ---
        "top_score": round(top_score, 4),
        "mean_score": round(statistics.fmean(scores), 4) if scores else 0.0,
        "weak_retrieval": weak_retrieval,
        "refused": refused,
        # The metric this whole module exists for: retrieval found nothing
        # that matched, and the model produced a confident answer anyway.
        # That answer is built on irrelevant context -- review it.
        "hallucination_risk": weak_retrieval and not refused,
        # Strong retrieval but the model declined: the opposite failure,
        # suggesting the prompt is too strict or chunks are cut badly.
        "over_refusal": (not weak_retrieval) and refused,
        # --- performance ---
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "total_ms": round(retrieval_ms + generation_ms, 1),
        # --- cost ---
        "input_tokens": (usage or {}).get("input_tokens"),
        "output_tokens": (usage or {}).get("output_tokens"),
        # --- provenance: which papers/pages were actually used ---
        "sources": [
            {
                "paper": c["source_paper"],
                "pages": [c["page_start"], c["page_end"]],
                "score": round(c["score"], 4),
            }
            for c in retrieved_chunks
        ],
    })
    return query_id


def log_feedback(query_id, rating, note=None):
    """Record a thumbs up/down against a previously logged query."""
    _append({
        "event": "feedback",
        "query_id": query_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rating": rating,  # "up" | "down"
        "note": note,
    })


def load_events():
    """Read every event. Skips unparseable lines rather than failing."""
    path = config.TELEMETRY_LOG_PATH
    if not path.exists():
        return []
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # A partially-written final line (app killed mid-write)
                    # shouldn't make the whole dashboard unreadable.
                    continue
    except Exception as exc:  # noqa: BLE001
        print(f"[telemetry] failed to read log: {exc}", file=sys.stderr)
        return []
    return events


def _derive_flags(record):
    """Recompute the judgement flags from the raw signals stored on a record.

    The flags are also written into the log at query time (an audit trail of
    what the system concluded then), but the dashboard uses *these* values,
    recomputed on read. That split matters: raw signals (top_score, the answer
    text, timings) are facts and never change, while `refused` /
    `hallucination_risk` are opinions produced by a heuristic that gets
    improved over time. Recomputing on read means a better classifier
    retroactively improves every historical record instead of leaving the
    dashboard showing verdicts from an older, worse rule.

    This actually happened here: the first refusal heuristic matched anywhere
    in the answer and mislabelled a good answer that ended with a caveat (see
    the note above REFUSAL_LEAD_CHARS). Fixing looks_like_refusal() corrected
    the already-logged records too, because of this.
    """
    weak = record.get("top_score", 0.0) < config.LOW_CONFIDENCE_THRESHOLD
    refused = looks_like_refusal(record.get("answer", ""))
    record["weak_retrieval"] = weak
    record["refused"] = refused
    record["hallucination_risk"] = weak and not refused
    record["over_refusal"] = (not weak) and refused
    return record


def build_records(events=None):
    """Reduce the event stream into one record per query, newest last.

    Feedback events are folded into the query they reference; this is the
    read-side cost of the append-only design described in the module docstring.
    """
    events = load_events() if events is None else events

    queries = {}
    order = []
    for e in events:
        if e.get("event") == "query":
            qid = e.get("query_id")
            if qid and qid not in queries:
                queries[qid] = _derive_flags({**e, "rating": None, "note": None})
                order.append(qid)

    for e in events:
        if e.get("event") == "feedback":
            record = queries.get(e.get("query_id"))
            if record:
                # Last rating wins -- the user is allowed to change their mind.
                record["rating"] = e.get("rating")
                record["note"] = e.get("note")

    return [queries[qid] for qid in order]


def summarize(records):
    """Aggregate records into the numbers the Metrics tab displays.

    Pure Python (no pandas) so this stays importable from cli.py and testable
    without Streamlit -- same framework-agnostic rule the rest of src/ follows.
    """
    if not records:
        return None

    total = len(records)
    total_ms = [r["total_ms"] for r in records]
    retrieval_ms = [r["retrieval_ms"] for r in records]
    generation_ms = [r["generation_ms"] for r in records]
    top_scores = [r["top_score"] for r in records]

    rated = [r for r in records if r.get("rating") in ("up", "down")]
    ups = sum(1 for r in rated if r["rating"] == "up")

    in_tok = [r["input_tokens"] for r in records if r.get("input_tokens")]
    out_tok = [r["output_tokens"] for r in records if r.get("output_tokens")]

    # Which papers actually get retrieved. Papers that never appear are
    # either off-topic for real questions or chunked badly -- both worth knowing.
    paper_hits = Counter()
    for r in records:
        for s in r.get("sources", []):
            paper_hits[s["paper"]] += 1

    return {
        "total_queries": total,
        "median_total_ms": statistics.median(total_ms),
        "p95_total_ms": _percentile(total_ms, 95),
        "median_retrieval_ms": statistics.median(retrieval_ms),
        "median_generation_ms": statistics.median(generation_ms),
        "mean_top_score": statistics.fmean(top_scores),
        "weak_retrieval_count": sum(1 for r in records if r.get("weak_retrieval")),
        "refused_count": sum(1 for r in records if r.get("refused")),
        "hallucination_risk_count": sum(1 for r in records if r.get("hallucination_risk")),
        "over_refusal_count": sum(1 for r in records if r.get("over_refusal")),
        "rated_count": len(rated),
        "up_count": ups,
        "down_count": len(rated) - ups,
        "satisfaction": (ups / len(rated)) if rated else None,
        "total_input_tokens": sum(in_tok) if in_tok else None,
        "total_output_tokens": sum(out_tok) if out_tok else None,
        "paper_hits": paper_hits,
    }


def _percentile(values, pct):
    """Nearest-rank percentile. Small-sample friendly, no numpy needed."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[k]


def score_histogram(records, bins=10):
    """Bucket top-1 similarity scores into [0,1] bins for a bar chart."""
    counts = [0] * bins
    for r in records:
        idx = min(bins - 1, max(0, int(r["top_score"] * bins)))
        counts[idx] += 1
    labels = [f"{i / bins:.1f}-{(i + 1) / bins:.1f}" for i in range(bins)]
    return labels, counts


if __name__ == "__main__":
    # Quick terminal summary: `python telemetry.py`
    records = build_records()
    stats = summarize(records)
    if not stats:
        print(f"No telemetry recorded yet at {config.TELEMETRY_LOG_PATH}")
        raise SystemExit(0)

    print(f"Queries logged      : {stats['total_queries']}")
    print(f"Median latency      : {stats['median_total_ms'] / 1000:.2f}s "
          f"(retrieval {stats['median_retrieval_ms']:.0f}ms + "
          f"generation {stats['median_generation_ms'] / 1000:.2f}s)")
    print(f"p95 latency         : {stats['p95_total_ms'] / 1000:.2f}s")
    print(f"Mean top-1 score    : {stats['mean_top_score']:.3f}")
    print(f"Weak retrieval      : {stats['weak_retrieval_count']}")
    print(f"Hallucination risk  : {stats['hallucination_risk_count']}")
    print(f"Model declined      : {stats['refused_count']}")
    if stats["satisfaction"] is not None:
        print(f"Satisfaction        : {stats['satisfaction']:.0%} "
              f"({stats['up_count']}/{stats['rated_count']} rated)")
