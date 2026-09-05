"""
The settings sweep: turning two guessed numbers into two measured ones.

WHAT THIS ANSWERS
-------------------
config.py sets CHUNK_SIZE = 1000 and DEFAULT_TOP_K = 5. Both were chosen
by judgement and defended in prose. An interviewer asking "why 1000?"
currently gets a paragraph of plausible reasoning; after this script runs,
they get a table.

That distinction is most of the difference between a project and an
experiment, and it is cheap to close: scoring retrieval touches no API at
all, so the entire sweep is free. The only cost is CPU time re-embedding
the corpus once per chunk size.

Published ablations disagree with the current setting, which is what makes
it worth running rather than assuming: several report ~500 characters as
the sweet spot and clear degradation past 2000, and at least one found
retrieval quality peaking at top-k=3 and falling after. Those studies were
not run on dense academic PDFs, so they do not settle the question here --
they just mean the question is open.

IT DOES NOT TOUCH THE PRODUCTION INDEX
----------------------------------------
Every candidate index is built in memory and thrown away. Nothing under
data/vector_store/ is read for the alternatives or written at any point.
The obvious implementation -- re-run ingest.py and embed_store.py per
setting -- would leave the committed index rebuilt at whatever setting
happened to run last, silently changing the deployed app to a
configuration nobody chose.

OVERLAP IS HELD PROPORTIONAL, NOT FIXED
-----------------------------------------
The corpus is re-chunked with overlap kept at a constant fraction of chunk
size (the 200/1000 ratio currently in config.py), not at a constant 200
characters. A fixed 200 would mean 40% redundancy at 500 characters and
13% at 1500, so a "chunk size" comparison would really be measuring two
variables at once and crediting the difference to the wrong one.
"""

import argparse
import time

import faiss

import config
import eval_core as ec
from ingest import chunks_from_pages, extract_pages
from retrieve import embed_texts

# Candidate settings. Ranges chosen to bracket the current values rather than
# to flatter them: 1000 and 5 must be able to lose.
CHUNK_SIZES = [500, 1000, 1500]
TOP_KS = [3, 5, 10]

OVERLAP_RATIO = config.CHUNK_OVERLAP / config.CHUNK_SIZE  # 0.2 as committed

_pages_cache = None


def load_pages():
    """Extract every PDF once. Re-extracting per setting would dominate runtime."""
    global _pages_cache
    if _pages_cache is None:
        _pages_cache = []
        for pdf in sorted(config.PAPERS_DIR.glob("*.pdf")):
            _pages_cache.append((pdf.name, extract_pages(pdf)))
    return _pages_cache


def build_corpus(chunk_size):
    """Re-chunk and re-embed the whole corpus at one chunk size, in memory."""
    overlap = int(round(chunk_size * OVERLAP_RATIO))
    chunks = []
    for name, pages in load_pages():
        chunks.extend(chunks_from_pages(pages, name,
                                        chunk_size=chunk_size, overlap=overlap))

    vectors = embed_texts([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks, overlap


def search(index, chunks, question, top_k):
    """Same search as retrieve.search(), against an in-memory index."""
    query = embed_texts([question])
    scores, positions = index.search(query, top_k)
    out = []
    for s, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue
        out.append({**chunks[pos], "score": float(s)})
    return out


def score_setting(index, chunks, questions, top_k):
    """Score one (chunk_size, top_k) combination."""
    hits, rrs, precs, covs, confs = [], [], [], [], []
    unans_conf = []

    for q in questions:
        results = search(index, chunks, q["question"], top_k)
        correct = q.get("correct_papers", [])
        if q.get("answerable", True) and correct:
            hits.append(ec.hit_at_k(results, correct, top_k))
            rrs.append(ec.reciprocal_rank(results, correct))
            precs.append(ec.precision_at_k(results, correct, top_k))
            covs.append(ec.papers_covered(results, correct))
            confs.append(ec.top_score(results))
        else:
            unans_conf.append(ec.top_score(results))

    return {
        "hit": ec.summarise_proportion(hits),
        "mrr": ec.summarise_mean(rrs),
        "precision": ec.summarise_mean(precs),
        "coverage": ec.summarise_mean(covs),
        "conf_ans": ec.summarise_mean(confs),
        "conf_unans": ec.summarise_mean(unans_conf),
    }


def run_sweep(chunk_sizes=None, top_ks=None):
    chunk_sizes = chunk_sizes or CHUNK_SIZES
    top_ks = top_ks or TOP_KS
    questions = ec.load_questions()

    grid, meta = {}, {}
    for size in chunk_sizes:
        started = time.perf_counter()
        print(f"  chunk_size={size} : chunking and embedding ...", flush=True)
        index, chunks, overlap = build_corpus(size)
        meta[size] = {
            "n_chunks": len(chunks),
            "overlap": overlap,
            "build_s": time.perf_counter() - started,
            "mean_chars": sum(len(c["text"]) for c in chunks) / len(chunks),
        }
        for k in top_ks:
            grid[(size, k)] = score_setting(index, chunks, questions, k)
        print(f"    {len(chunks)} chunks, "
              f"{meta[size]['build_s']:.1f}s", flush=True)

    return grid, meta, chunk_sizes, top_ks


# --- Reporting ---------------------------------------------------------------

def _best(grid, metric):
    return max(grid.items(), key=lambda kv: kv[1][metric]["value"])


def write_report(grid, meta, chunk_sizes, top_ks):
    L = ["# Settings Sweep: chunk size and top-k", ""]
    L.append(f"Every combination of chunk size {chunk_sizes} and top-k {top_ks}, scored "
             f"on the same labelled questions. The corpus is re-chunked and re-embedded "
             f"from the PDFs for each chunk size.")
    L.append("")
    L.append(f"**Currently committed:** `CHUNK_SIZE = {config.CHUNK_SIZE}`, "
             f"`DEFAULT_TOP_K = {config.DEFAULT_TOP_K}`.")
    L.append("")
    L.append("Free to run -- retrieval scoring makes no API calls. Overlap is held at a "
             "constant fraction of chunk size "
             f"({OVERLAP_RATIO:.0%}), so this measures chunk size rather than chunk size "
             "and redundancy together. Nothing under `data/vector_store/` is modified: "
             "every candidate index is built in memory and discarded.")
    L.append("")

    L += ["## Corpus shape at each chunk size", "",
          "| Chunk size | Overlap | Chunks | Mean chars | Build time |",
          "|---|---|---|---|---|"]
    for size in chunk_sizes:
        m = meta[size]
        L.append(f"| {size} | {m['overlap']} | {m['n_chunks']} | "
                 f"{m['mean_chars']:.0f} | {m['build_s']:.1f}s |")
    L.append("")

    for metric, label, pct in [("hit", "Hit Rate@k", True), ("mrr", "MRR", False),
                               ("coverage", "Paper coverage", True),
                               ("precision", "Precision@k", True)]:
        L += [f"## {label}", "",
              "| chunk size \\ top-k | " + " | ".join(str(k) for k in top_ks) + " |",
              "|---" * (len(top_ks) + 1) + "|"]
        for size in chunk_sizes:
            cells = " | ".join(
                ec.format_ci(grid[(size, k)][metric], pct=pct) for k in top_ks)
            L.append(f"| **{size}** | {cells} |")
        L.append("")

    best_hit = _best(grid, "hit")
    best_mrr = _best(grid, "mrr")
    current = grid.get((config.CHUNK_SIZE, config.DEFAULT_TOP_K))

    L += ["## What the sweep says", ""]
    L.append(f"- Best Hit Rate: **chunk {best_hit[0][0]}, top-k {best_hit[0][1]}** at "
             f"{ec.format_ci(best_hit[1]['hit'])} -- but note this will almost always "
             f"pick the largest k in the grid, because Hit Rate cannot fall as k rises. "
             f"It is not evidence that k={best_hit[0][1]} is the right choice.")
    L.append(f"- Best MRR: **chunk {best_mrr[0][0]}, top-k {best_mrr[0][1]}** at "
             f"{ec.format_ci(best_mrr[1]['mrr'], pct=False)}")
    if current:
        L.append(f"- Currently committed ({config.CHUNK_SIZE}/{config.DEFAULT_TOP_K}): "
                 f"Hit Rate {ec.format_ci(current['hit'])}, "
                 f"MRR {ec.format_ci(current['mrr'], pct=False)}")
    L.append("")

    if current:
        gap = best_hit[1]["hit"]["value"] - current["hit"]["value"]
        overlapping = (best_hit[1]["hit"]["lo"] <= current["hit"]["hi"])
        if gap <= 0.001:
            L.append("**The committed settings are already the best in this grid.** "
                     "Worth knowing, and worth being able to show rather than assert.")
        elif overlapping:
            L.append(f"The best cell beats the committed one by {gap:+.2f} on Hit Rate, "
                     f"but the confidence intervals overlap, so this grid **cannot "
                     f"establish** that the difference is real. The honest conclusion is "
                     f"that retrieval quality is not very sensitive to these two knobs "
                     f"on this corpus -- which is itself the answer to \"why 1000?\": "
                     f"within measurement error, it does not matter much.")
        else:
            L.append(f"The best cell beats the committed one by {gap:+.2f} on Hit Rate "
                     f"with non-overlapping intervals. That is a real difference and "
                     f"`config.py` should be changed to match.")
    L.append("")

    L.append("**Reading the top-k columns.** Hit Rate can only rise with larger k -- "
             "more slots cannot lose a paper that a smaller k already found -- so a "
             "higher Hit Rate at k=10 is not evidence that k=10 is better. MRR and "
             "precision are the columns that can genuinely fall, and every extra chunk "
             "is paid for in prompt tokens on every single query. Pick the smallest k "
             "whose MRR has stopped improving.")
    L.append("")
    L.append("**This table is corpus-specific.** Re-run it after adding papers; the "
             "best chunk size for nine papers is not necessarily the best for fifty.")
    L.append("")

    path = config.SWEEP_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def print_summary(grid, chunk_sizes, top_ks):
    print()
    for metric, label, pct in [("hit", "Hit Rate@k", True), ("mrr", "MRR", False)]:
        print(f"{label}")
        header = "".join(f"{'k=' + str(k):>10}" for k in top_ks)
        print(f"  {'chunk':<8}{header}")
        for size in chunk_sizes:
            cells = "".join(f"{grid[(size, k)][metric]['value']:>10.3f}" for k in top_ks)
            print(f"  {size:<8}{cells}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Sweep chunk size and top-k. No API calls; does not touch the "
                    "committed index.")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--top-ks", type=int, nargs="+", default=None)
    args = parser.parse_args()

    print("Sweeping settings (no API calls, production index untouched) ...")
    grid, meta, sizes, ks = run_sweep(args.chunk_sizes, args.top_ks)
    ec.update_summary("sweep", {
        "chunk_sizes": sizes, "top_ks": ks,
        "committed": {"chunk_size": config.CHUNK_SIZE, "top_k": config.DEFAULT_TOP_K},
        "grid": {f"{s}x{k}": {"hit": grid[(s, k)]["hit"]["value"],
                              "mrr": grid[(s, k)]["mrr"]["value"],
                              "coverage": grid[(s, k)]["coverage"]["value"]}
                 for s in sizes for k in ks},
        "best_hit": f"{_best(grid, 'hit')[0][0]}x{_best(grid, 'hit')[0][1]}",
        "best_mrr": f"{_best(grid, 'mrr')[0][0]}x{_best(grid, 'mrr')[0][1]}",
    })
    path = write_report(grid, meta, sizes, ks)
    print_summary(grid, sizes, ks)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
