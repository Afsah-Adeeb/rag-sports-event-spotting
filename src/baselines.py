"""
Two deliberately unintelligent retrievers, to give the real one something to beat.

WHY THIS EXISTS
-----------------
"Hit Rate@5 is 0.93" is not a result. It is a number with nothing to
compare it against. 0.93 could mean the embeddings are doing excellent
work, or it could mean the questions are easy enough that anything would
score 0.93. Without a floor and without a cheap alternative, there is no
way to tell -- and "we used embeddings" is an unsupported claim rather
than a measured decision.

So this module implements two alternative retrievers with the exact same
interface as retrieve.retrieve(), scored by the exact same code:

  RANDOM  -- pick k chunks at random. This is the floor. Anything at or
             near this number means the system is not working at all.
             It also calibrates the metric itself: with 9 papers and 827
             chunks, random Hit Rate@5 is not 0, and knowing what it
             actually is tells you how much of the real score is free.

  BM25    -- the standard keyword ranking function from classical search
             engines. Matches words, not meaning.

WHY BM25 AND NOT A WORD-OVERLAP COUNT
---------------------------------------
The tempting cheap version is "count how many question words appear in
the chunk". That baseline is a strawman: it ignores how rare a word is,
so "the" counts as much as "SGP-Mixer", and it favours long chunks purely
for being long. Beating it would prove nothing.

BM25 is what an actual keyword search engine uses. It weights rare terms
far more heavily than common ones (via IDF) and normalises for chunk
length, which removes both of those flaws. If the embeddings only just
edge out BM25, that is a genuine and worth-reporting finding -- and it is
one the strawman version would have hidden behind a flattering margin.

No new dependency: BM25 is about thirty lines of arithmetic, and writing
it out keeps the mechanism visible, the same reason this project uses
FAISS directly instead of a vector-database client.

WHAT IS HELD CONSTANT
-----------------------
Both baselines search the SAME chunks the FAISS index was built from,
read from chunk_metadata.jsonl, and return the same dict shape. So the
comparison isolates one variable: how candidates are ranked. Chunking,
cleaning, the corpus, the questions, and the scoring code are identical
across all three arms.
"""

import argparse
import json
import math
import random
import re
import zlib
from collections import Counter

import config
import eval_core as ec

_chunks = None
_bm25 = None

# Standard BM25 constants. k1 controls how fast term-frequency saturates
# (a word appearing 10 times is not 10x more relevant than once); b controls
# how strongly long chunks are penalised. These are the usual defaults and are
# not tuned here -- tuning the baseline to lose would defeat the point.
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")


def load_chunks():
    """Load the exact chunk set the FAISS index was built from."""
    global _chunks
    if _chunks is None:
        with open(config.CHUNK_METADATA_PATH, encoding="utf-8") as f:
            _chunks = [json.loads(line) for line in f]
    return _chunks


def tokenize(text):
    """Lowercase alphanumeric tokens, dropping single characters.

    No stopword list on purpose: BM25's IDF term already drives words like
    "the" to near-zero weight, so a hand-maintained stoplist would add a
    tuning knob without adding accuracy.
    """
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) > 1]


# --- BM25 --------------------------------------------------------------------

class BM25:
    """Classical BM25 ranking over the chunk set."""

    def __init__(self, documents):
        self.docs = [tokenize(d) for d in documents]
        self.n = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avgdl = sum(self.lengths) / self.n if self.n else 0.0
        self.freqs = [Counter(d) for d in self.docs]

        df = Counter()
        for doc in self.docs:
            df.update(set(doc))
        # Smoothed IDF. The +1 inside the log keeps it non-negative for terms
        # that appear in more than half the corpus, which the classic form
        # makes negative and which would let a common word push a chunk DOWN.
        self.idf = {
            term: math.log(1 + (self.n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def scores(self, query):
        terms = tokenize(query)
        out = [0.0] * self.n
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.freqs):
                f = freq.get(term, 0)
                if not f:
                    continue
                norm = 1 - BM25_B + BM25_B * (self.lengths[i] / self.avgdl)
                out[i] += idf * (f * (BM25_K1 + 1)) / (f + BM25_K1 * norm)
        return out


def _get_bm25():
    global _bm25
    if _bm25 is None:
        chunks = load_chunks()
        _bm25 = BM25([c["text"] for c in chunks])
    return _bm25


def bm25_retrieve(question, top_k=config.DEFAULT_TOP_K):
    """Keyword retrieval. Same signature and return shape as retrieve.retrieve()."""
    chunks = load_chunks()
    scores = _get_bm25().scores(question)
    ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])[:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in ranked if scores[i] > 0]


# --- Random ------------------------------------------------------------------

def random_retrieve(question, top_k=config.DEFAULT_TOP_K, seed=None):
    """Pick k chunks at random. The floor.

    Seeded from the question text so the same question always draws the same
    chunks. Otherwise the floor would move between runs and could not be
    quoted alongside the other arms.

    Uses crc32 rather than the built-in hash(): Python salts string hashing
    with a per-process random seed, so hash("...") differs between runs. The
    first version of this function used it and the multi_paper floor moved
    from 0.86 to 1.00 between two runs of the same code -- a "reproducible"
    baseline that silently was not.
    """
    chunks = load_chunks()
    stable = zlib.crc32(question.encode("utf-8"))
    rng = random.Random(seed if seed is not None else stable)
    picked = rng.sample(range(len(chunks)), min(top_k, len(chunks)))
    return [{**chunks[i], "score": 0.0} for i in picked]


# --- Three-way comparison ----------------------------------------------------

def _semantic_retrieve(question, top_k):
    from retrieve import retrieve  # imported lazily: loads a model and an index
    return retrieve(question, top_k=top_k)


ARMS = {
    "semantic": _semantic_retrieve,
    "bm25": lambda q, k: bm25_retrieve(q, top_k=k),
    "random": lambda q, k: random_retrieve(q, top_k=k),
}


def run_arm(retriever, questions, top_k):
    """Score one retriever over the answerable questions."""
    records = []
    for q in ec.answerable(questions):
        correct = q.get("correct_papers", [])
        if not correct:
            continue
        chunks = retriever(q["question"], top_k)
        records.append({
            "id": q["id"],
            "type": q.get("type", "untyped"),
            "question": q["question"],
            "correct_papers": correct,
            f"hit@{top_k}": ec.hit_at_k(chunks, correct, top_k),
            "hit@1": ec.hit_at_k(chunks, correct, 1),
            "rr": ec.reciprocal_rank(chunks, correct),
            "p@k": ec.precision_at_k(chunks, correct, top_k),
            "paper_coverage": ec.papers_covered(chunks, correct),
        })
    return records


def summarise_arm(records, top_k):
    return {
        "n": len(records),
        "hit@1": ec.summarise_proportion([r["hit@1"] for r in records]),
        f"hit@{top_k}": ec.summarise_proportion([r[f"hit@{top_k}"] for r in records]),
        "mrr": ec.summarise_mean([r["rr"] for r in records]),
        "precision": ec.summarise_mean([r["p@k"] for r in records]),
        "paper_coverage": ec.summarise_mean([r["paper_coverage"] for r in records]),
    }


def compare(top_k=config.DEFAULT_TOP_K):
    questions = ec.load_questions()
    results, summaries = {}, {}
    for name, retriever in ARMS.items():
        print(f"  running {name} ...")
        results[name] = run_arm(retriever, questions, top_k)
        summaries[name] = summarise_arm(results[name], top_k)
    return results, summaries


def write_report(results, summaries, top_k):
    L = ["# Retrieval Baselines", ""]
    L.append(f"Three retrievers over the same {summaries['semantic']['n']} answerable "
             f"questions, the same chunks, and the same scoring code. The only thing "
             f"that differs is how candidates are ranked.")
    L.append("")
    L += ["| Arm | What it does |", "|---|---|",
          "| **semantic** | The real system. Embeds the question, returns the nearest chunks by cosine similarity. |",
          "| **bm25** | Classical keyword search. Matches words, weighted by rarity and chunk length. No embeddings. |",
          "| **random** | Picks chunks at random. The floor. |", ""]

    caveat = ec.sample_size_caveat(summaries["semantic"]["n"])
    if caveat:
        L += ["> " + caveat, ""]

    L += ["## Results", "",
          f"| Metric | semantic | bm25 | random |", "|---|---|---|---|"]
    rows = [("Hit Rate@1", "hit@1", True), (f"Hit Rate@{top_k}", f"hit@{top_k}", True),
            ("MRR", "mrr", False), (f"Precision@{top_k}", "precision", True),
            ("Paper coverage", "paper_coverage", True)]
    for label, key, pct in rows:
        cells = " | ".join(ec.format_ci(summaries[a][key], pct=pct)
                           for a in ("semantic", "bm25", "random"))
        L.append(f"| {label} | {cells} |")
    L.append("")

    sem, bm, rnd = (summaries[a][f"hit@{top_k}"]["value"] for a in ("semantic", "bm25", "random"))
    L.append(f"Semantic retrieval scores **{sem - rnd:+.2f}** over random and "
             f"**{sem - bm:+.2f}** over keyword search on Hit Rate@{top_k}.")
    L.append("")
    if bm >= sem:
        L.append("**BM25 matches or beats the embeddings here.** That is a real result "
                 "and worth stating plainly: at this corpus size, on these questions, "
                 "the vector index is not earning its complexity. The place to look "
                 "next is the per-type table below -- the advantage of embeddings is "
                 "supposed to show up on reworded questions, not on ones that quote "
                 "the paper's own terminology.")
    else:
        L.append("The gap over BM25 is what the embeddings actually buy. The gap over "
                 "random is what any working retrieval buys, and quoting only that "
                 "second number would flatter the system.")
    L.append("")

    types = {}
    for name, recs in results.items():
        for qtype, group in ec.group_by_type(recs).items():
            entry = types.setdefault(qtype, {"n": len(group)})
            entry[name] = ec.summarise_proportion([r[f"hit@{top_k}"] for r in group])
            entry[name + "_cov"] = ec.summarise_mean(
                [r["paper_coverage"] for r in group])

    order = [t for t in ("simple", "paraphrase", "comparison", "multi_paper")
             if t in types]

    L += ["## By question type", "",
          f"### Hit Rate@{top_k}", "",
          "| Type | n | semantic | bm25 | random |", "|---|---|---|---|---|"]
    for qtype in order:
        e = types[qtype]
        cells = " | ".join(ec.format_ci(e[a]) for a in ("semantic", "bm25", "random"))
        L.append(f"| {qtype} | {e['n']} | {cells} |")
    L.append("")
    L.append("**The `paraphrase` row is the one that matters.** Those questions are "
             "deliberately worded so the paper's own vocabulary never appears -- "
             "\"a racket sport played indoors on a small table\" instead of \"table "
             "tennis\". Keyword search has nothing to match on. If embeddings are "
             "worth having, this is the row where it shows.")
    L.append("")

    L += [f"### Paper coverage (fraction of ALL correct papers reached)", "",
          "| Type | n | semantic | bm25 | random |", "|---|---|---|---|---|"]
    for qtype in order:
        e = types[qtype]
        cells = " | ".join(ec.format_ci(e[a + "_cov"]) for a in ("semantic", "bm25", "random"))
        L.append(f"| {qtype} | {e['n']} | {cells} |")
    L.append("")

    mp = types.get("multi_paper")
    if mp and mp["random"]["value"] >= mp["semantic"]["value"]:
        L.append(f"**Read those two tables together, because the `multi_paper` row is "
                 f"the worst result in this report.** Random retrieval scores "
                 f"{mp['random']['value']:.2f} on Hit Rate@{top_k} against the real "
                 f"system's {mp['semantic']['value']:.2f}, and "
                 f"{mp['random_cov']['value']:.2f} on paper coverage against "
                 f"{mp['semantic_cov']['value']:.2f}. The intervals overlap almost "
                 f"completely, so the honest reading is not \"random wins\" -- it is "
                 f"that **on questions needing several papers at once, semantic "
                 f"retrieval is statistically indistinguishable from picking chunks at "
                 f"random.**")
        L.append("")
        L.append("Two separate things are going on, and they should not be conflated:")
        L.append("")
        L.append(f"1. **Hit Rate@{top_k} is the wrong metric here.** These questions "
                 f"accept three or four papers out of nine, so five random chunks are "
                 f"almost guaranteed to touch one, and Hit Rate scores that as a full "
                 f"success. It should not be quoted for multi-paper questions at all.")
        L.append(f"2. **Top-k retrieval concentrates, and breadth questions need the "
                 f"opposite.** The five nearest chunks to a question tend to come from "
                 f"whichever single paper matches best, so semantic search reaches only "
                 f"{mp['semantic_cov']['value']:.2f} of the papers a complete answer "
                 f"needs. Random spreads across the corpus by construction, which is "
                 f"why it is not obviously worse here. Similarity ranking has no term "
                 f"for diversity, and that is a fixable property of the retriever "
                 f"(cap chunks per paper, or re-rank for spread) rather than a limit "
                 f"of embeddings.")
        L.append("")

    L += ["## Where they disagree", "",
          "Questions one arm found and the other missed. More useful than the "
          "averages: it names the specific failure.", ""]
    by_id = {a: {r["id"]: r for r in results[a]} for a in results}
    sem_only, bm_only = [], []
    for qid, r in by_id["semantic"].items():
        s, b = r[f"hit@{top_k}"], by_id["bm25"][qid][f"hit@{top_k}"]
        if s and not b:
            sem_only.append((qid, r["type"], r["question"]))
        elif b and not s:
            bm_only.append((qid, r["type"], r["question"]))

    L.append(f"**Semantic found, BM25 missed ({len(sem_only)}):**")
    L += [f"- `{i}` *({t})* {q}" for i, t, q in sem_only] or ["- none"]
    L.append("")
    L.append(f"**BM25 found, semantic missed ({len(bm_only)}):**")
    L += [f"- `{i}` *({t})* {q}" for i, t, q in bm_only] or ["- none"]
    L.append("")

    path = config.BASELINE_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def print_summary(summaries, top_k):
    print()
    print(f"{'metric':<18}{'semantic':>20}{'bm25':>20}{'random':>20}")
    for label, key, pct in [("Hit Rate@1", "hit@1", True),
                            (f"Hit Rate@{top_k}", f"hit@{top_k}", True),
                            ("MRR", "mrr", False),
                            (f"Precision@{top_k}", "precision", True),
                            ("Paper coverage", "paper_coverage", True)]:
        cells = "".join(f"{ec.format_ci(summaries[a][key], pct=pct):>20}"
                        for a in ("semantic", "bm25", "random"))
        print(f"{label:<18}{cells}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare semantic retrieval against BM25 and random. No API calls.")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    print(f"Comparing three retrievers at top-k={args.top_k} (no API calls) ...")
    results, summaries = compare(top_k=args.top_k)
    ec.update_summary("baselines", {
        "top_k": args.top_k,
        "arms": {a: {"hit": summaries[a][f"hit@{args.top_k}"],
                     "mrr": summaries[a]["mrr"],
                     "paper_coverage": summaries[a]["paper_coverage"]}
                 for a in summaries},
    })
    path = write_report(results, summaries, args.top_k)
    print_summary(summaries, args.top_k)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
