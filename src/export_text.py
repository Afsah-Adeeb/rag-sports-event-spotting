"""
Export each PDF to a plain-text file that the agentic retriever can grep.

WHY THIS STAGE EXISTS
-----------------------
The semantic pipeline never needs the papers as text on disk -- ingest.py
extracts, chunks, and embeds them in one pass, and after that everything
lives in the FAISS index. The agentic retriever (agent_rag.py) works the
opposite way: it has no index at all, and instead searches the source text
directly, the way a person would with Ctrl+F. That needs actual text files.

So this stage converts data/papers/*.pdf -> data/papers_text/*.txt.

KEEPING THE COMPARISON FAIR
-----------------------------
This deliberately reuses ingest.py's extraction and boilerplate stripping
rather than doing its own. If the agentic side searched raw PyMuPDF output
while the semantic side searched cleaned text, any difference in the
benchmark could just be preprocessing quality rather than the retrieval
strategy -- which is the thing we actually want to measure.

Page markers are written into the text ("[page 12]") for the same reason:
the semantic pipeline cites page ranges from chunk metadata, so the agentic
side needs page information available too, or it would look worse at
citation purely for lack of the data rather than for lack of ability.

ONE PARAGRAPH PER LINE, AND WHY IT MATTERS A LOT
--------------------------------------------------
PyMuPDF emits PDF text as it is laid out on the page -- wrapped at whatever
column width the paper was typeset in. Written out verbatim, a phrase that
spans a line break simply cannot be found by a line-based search. Measured
on this corpus: the T-DEED paper contains 32 lines mentioning
"discriminab...", but grepping the phrase "temporal discriminability"
matched **zero** of them, because the words land on different lines
("...Temporal-\nDiscriminability Enhancer..."). The paper that coined the
term was invisible to a search for the term.

That is a line-wrapping artifact, not a real property of agentic retrieval,
and benchmarking against it would be a strawman. ingest.py already joins
wrapped lines into flowing paragraphs before chunking (see
pages_to_paragraphs, which does `.replace("\\n", " ")`), so the semantic
side never sees the wrapping. Writing raw wrapped lines here would hand the
semantic side a preprocessing advantage and then report it as a retrieval
result.

So this applies the same join, then re-splits the flowing text one sentence
per line. Both arms of the comparison see identical *characters*; only where
the newlines fall differs, and newline placement matters to nothing except
grep.

Sentence granularity rather than paragraph granularity because paragraphs
turned out far too coarse: PDF pages frequently contain no blank lines at
all, so joining collapsed an entire page into one line -- whole papers came
out as 20 lines averaging 1,500 characters, with the longest at 6,432. That
makes grep output unreadable, blows the output budget on a single match, and
leaves read_paper's line offsets useless as a way to page through a document.
Sentences give ~100-200 character lines: phrases still match, results stay
readable, and line numbers mean something.

(A related artifact is left in place deliberately: words hyphenated across a
line break rejoin with a space, so "bro-\\nken" becomes "bro ken". Ugly, but
ingest.py does exactly the same thing, so both sides are affected equally
and the comparison stays symmetric.)

    cd src
    ../.venv/Scripts/python.exe export_text.py
"""

import re

import config
from ingest import (
    extract_pages,
    find_boilerplate_lines,
    pages_to_paragraphs,
    strip_boilerplate,
)

# Split after sentence-ending punctuation followed by whitespace and something
# that starts a new sentence (capital letter or an opening bracket/citation).
# Requiring the capital keeps "Fig. 3", "0.5 mAP", and "et al. [12]" intact,
# which a naive split on ". " would shred. Not linguistically perfect, and it
# does not need to be -- an occasional extra line break costs nothing here.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(paragraph):
    """Break one flowing paragraph into sentence-sized lines."""
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(paragraph) if s.strip()]


def pdf_to_text(pdf_path):
    """Extract one PDF to page-marked text, one sentence per line.

    Reuses ingest.py's extraction, boilerplate stripping, and paragraph
    joining so the agentic retriever searches the same text the semantic
    pipeline embedded, then re-splits it at sentence boundaries so grep
    results and line offsets are usable.
    """
    pages = extract_pages(pdf_path)
    pages = strip_boilerplate(pages, find_boilerplate_lines(pages))
    paragraphs = pages_to_paragraphs(pages)  # already line-joined into flowing text

    lines = []
    current_page = None
    for page_num, para in paragraphs:
        if page_num != current_page:
            lines.append(f"\n[page {page_num}]")
            current_page = page_num
        lines.extend(split_sentences(para))
    return "\n".join(lines).strip() + "\n"


def main():
    pdf_paths = sorted(config.PAPERS_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {config.PAPERS_DIR}.")
        return

    config.PAPERS_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    for pdf_path in pdf_paths:
        text = pdf_to_text(pdf_path)
        out_path = config.PAPERS_TEXT_DIR / (pdf_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        total_chars += len(text)
        print(f"  {len(text):>8,} chars  {out_path.name}")

    print(f"\nWrote {len(pdf_paths)} text files ({total_chars:,} chars total) "
          f"to {config.PAPERS_TEXT_DIR}")


if __name__ == "__main__":
    main()
