"""
Robustness: does retrieval survive being asked badly?

THE BLIND SPOT THIS CLOSES
----------------------------
Every question in the test set is written in careful academic English,
with correct spelling, full punctuation, and the field's own terminology.
That is the easiest possible input, and it is not what a real user types.
Scoring only on those questions measures the system under laboratory
conditions and quietly reports the result as if it were field performance.

So each question is re-asked in three degraded forms and re-scored. All
four forms are generated deterministically from the original, so the test
is reproducible and no variant was hand-picked to flatter or punish the
system.

  original   what the test set says
  typo       character-level corruption at a fixed rate -- transpositions,
             deletions, doubled letters, applied inside words only
  casual     lowercase, punctuation stripped, and the polite question
             frame removed ("what is the ..." -> "the ...")
  keywords   stopwords removed entirely, leaving a bag of content words,
             which is how a lot of people actually use a search box

WHAT IS ACTUALLY BEING MEASURED
---------------------------------
Two different things, and conflating them would hide the interesting half:

  ACCURACY  -- does the degraded question still retrieve a correct paper?
               A drop here is a real quality loss.
  STABILITY -- does it retrieve the SAME top paper as the clean question?
               This can fall even when accuracy does not, because several
               papers may be acceptable answers. Instability without an
               accuracy drop means the system is luckier than it is robust,
               which matters once the corpus grows and near-duplicate
               papers start competing.

Free to run: no API calls, retrieval only.
"""

import argparse
import random
import re
import string
import zlib

# This script lives in a subfolder but imports the pipeline modules that sit in
# src/ (config, retrieve, generate, telemetry). Running a script puts its OWN
# folder on the import path, not its parent, so the parent is added explicitly.
# Siblings inside this folder import normally.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import config
import eval_core as ec
from retrieve import retrieve

# One typo per this many characters. Roughly a realistic mistyping rate --
# high enough to matter, low enough that the question stays readable.
TYPO_EVERY_N_CHARS = 12

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "do", "does",
    "for", "from", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "them", "these", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "you", "your", "here", "there",
    "any", "all", "can", "did", "has", "have", "not", "they", "we", "us",
    "much", "many", "some", "each", "other", "than", "then", "so", "if",
}

# Polite question frames, stripped for the "casual" variant.
_FRAME = re.compile(
    r"^(what (is|are|was|were|does|do|accuracy)|how (is|are|does|do|many|much)|"
    r"which|why|when|where|is there|are there|do any|can you|could you|"
    r"tell me about|compare)\s+", re.IGNORECASE)


def _rng(question):
    """Deterministic per-question randomness. crc32, not hash(): see baselines.py."""
    return random.Random(zlib.crc32(question.encode("utf-8")))


def add_typos(question):
    """Corrupt characters inside words at a fixed rate.

    Only word interiors are touched. Mangling first and last letters produces
    strings a human would never type, which would make this an adversarial
    test rather than a realistic one.
    """
    rng = _rng(question)
    chars = list(question)
    n_typos = max(1, len(question) // TYPO_EVERY_N_CHARS)

    # Positions that sit inside a word, so a typo never destroys a word boundary.
    interior = [
        i for i in range(1, len(chars) - 1)
        if chars[i].isalpha() and chars[i - 1].isalpha() and chars[i + 1].isalpha()
    ]
    if not interior:
        return question

    for pos in rng.sample(interior, min(n_typos, len(interior))):
        kind = rng.choice(("swap", "drop", "double"))
        if kind == "swap":
            chars[pos], chars[pos - 1] = chars[pos - 1], chars[pos]
        elif kind == "drop":
            chars[pos] = ""
        else:
            chars[pos] = chars[pos] * 2
    return "".join(chars)


def make_casual(question):
    """Lowercase, strip punctuation, and drop the polite question frame."""
    text = _FRAME.sub("", question.strip())
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.lower().split())


def keywords_only(question):
    """Strip stopwords, leaving the content words -- search-box style."""
    words = re.findall(r"[A-Za-z0-9\-]+", question)
    kept = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(kept) if kept else question


VARIANTS = {
    "original": lambda q: q,
    "typo": add_typos,
    "casual": make_casual,
    "keywords": keywords_only,
}


def run(top_k=None):
    top_k = top_k or config.DEFAULT_TOP_K
    questions = ec.answerable(ec.load_questions())
    questions = [q for q in questions if q.get("correct_papers")]

    records = []
    for q in questions:
        entry = {"id": q["id"], "type": q["type"], "question": q["question"],
                 "correct_papers": q["correct_papers"], "variants": {}}
        baseline_top = None
        for name, transform in VARIANTS.items():
            text = transform(q["question"])
            chunks = retrieve(text, top_k=top_k)
            top_paper = chunks[0]["source_paper"] if chunks else None
            if name == "original":
                baseline_top = top_paper
            entry["variants"][name] = {
                "text": text,
                "hit": ec.hit_at_k(chunks, q["correct_papers"], top_k),
                "rr": ec.reciprocal_rank(chunks, q["correct_papers"]),
                "coverage": ec.papers_covered(chunks, q["correct_papers"]),
                "top_paper": top_paper,
                "same_top": int(top_paper == baseline_top),
                "conf": ec.top_score(chunks),
            }
        records.append(entry)
    return records


def summarise(records):
    out = {}
    for name in VARIANTS:
        vs = [r["variants"][name] for r in records]
        out[name] = {
            "hit": ec.summarise_proportion([v["hit"] for v in vs]),
            "mrr": ec.summarise_mean([v["rr"] for v in vs]),
            "same_top": ec.summarise_proportion([v["same_top"] for v in vs]),
            "conf": ec.summarise_mean([v["conf"] for v in vs]),
        }
    return out


def write_report(records, summary, top_k):
    L = ["# Robustness: retrieval under badly-typed questions", ""]
    L.append(f"Every answerable question re-asked in four forms and re-scored at "
             f"top-k={top_k}. All variants are generated deterministically from the "
             f"original, so nothing here was hand-picked.")
    L.append("")
    L += ["| Variant | What it does to the question |", "|---|---|",
          "| `original` | unchanged -- careful academic English |",
          f"| `typo` | one character-level typo per ~{TYPO_EVERY_N_CHARS} characters, "
          "inside words only |",
          "| `casual` | lowercased, punctuation removed, polite question frame stripped |",
          "| `keywords` | stopwords removed -- a bag of content words, search-box style |",
          ""]

    L += ["## Results", "",
          "| Variant | Hit Rate@%d | MRR | Same top paper as original | Mean confidence |"
          % top_k, "|---|---|---|---|---|"]
    for name in VARIANTS:
        s = summary[name]
        same = "-" if name == "original" else ec.format_ci(s["same_top"])
        L.append(f"| `{name}` | {ec.format_ci(s['hit'])} | "
                 f"{ec.format_ci(s['mrr'], pct=False)} | {same} | "
                 f"{ec.format_ci(s['conf'], pct=False)} |")
    L.append("")

    base = summary["original"]
    L.append("**Accuracy and stability are different questions.** Hit Rate can hold "
             "steady while the *same top paper* column falls, because several papers "
             "may be acceptable for one question. That combination means the system is "
             "landing on a different-but-still-correct source -- fine today, fragile "
             "once the corpus grows and near-duplicate papers start competing.")
    L.append("")

    for name in VARIANTS:
        if name == "original":
            continue
        drop = base["hit"]["value"] - summary[name]["hit"]["value"]
        overlapping = summary[name]["hit"]["hi"] >= base["hit"]["lo"]
        verdict = ("within measurement error" if overlapping
                   else "a real drop, outside the intervals")
        L.append(f"- **`{name}`**: Hit Rate {drop:+.2f} vs original ({verdict}); "
                 f"kept the same top paper on "
                 f"{summary[name]['same_top']['value']:.0%} of questions.")
    L.append("")

    L += ["## Per-type", "",
          "| Type | " + " | ".join(f"`{n}`" for n in VARIANTS) + " |",
          "|---" * (len(VARIANTS) + 1) + "|"]
    for qtype, group in sorted(ec.group_by_type(records).items()):
        cells = " | ".join(
            f"{ec.summarise_proportion([r['variants'][n]['hit'] for r in group])['value']:.2f}"
            for n in VARIANTS)
        L.append(f"| {qtype} ({len(group)}) | {cells} |")
    L.append("")

    L += ["## Questions that broke", "",
          "Questions the original form got right and a degraded form got wrong. "
          "These name the actual failure rather than averaging it away.", ""]
    any_broke = False
    for r in records:
        broke = [n for n in VARIANTS
                 if n != "original" and r["variants"]["original"]["hit"]
                 and not r["variants"][n]["hit"]]
        if not broke:
            continue
        any_broke = True
        L.append(f"**`{r['id']}`** ({r['type']}) -- broke under: "
                 f"{', '.join('`' + b + '`' for b in broke)}")
        L.append(f"- original: {r['question']}")
        for b in broke:
            L.append(f"- {b}: {r['variants'][b]['text']}")
        L.append("")
    if not any_broke:
        L.append("None. Every question the clean form retrieved correctly survived all "
                 "three degradations.")
        L.append("")

    path = config.ROBUSTNESS_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def print_summary(summary, top_k):
    print()
    print(f"{'variant':<12}{'Hit@'+str(top_k):>20}{'MRR':>20}{'same top':>20}")
    for name in VARIANTS:
        s = summary[name]
        same = "-" if name == "original" else ec.format_ci(s["same_top"])
        print(f"{name:<12}{ec.format_ci(s['hit']):>20}"
              f"{ec.format_ci(s['mrr'], pct=False):>20}{same:>20}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Re-ask every question badly and re-score. No API calls.")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    print("Testing robustness to typos, casual phrasing and keyword-only input "
          "(no API calls) ...")
    records = run(top_k=args.top_k)
    summary = summarise(records)
    ec.update_summary("robustness", {
        "top_k": args.top_k,
        "variants": {n: {"hit": summary[n]["hit"], "mrr": summary[n]["mrr"],
                         "same_top": summary[n]["same_top"]}
                     for n in VARIANTS},
    })
    path = write_report(records, summary, args.top_k)
    print_summary(summary, args.top_k)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
