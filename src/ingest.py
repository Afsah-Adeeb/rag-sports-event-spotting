"""
Step 1: Load PDFs and split them into overlapping text chunks.

CHUNKING STRATEGY (fixed-size, paragraph-aware, with overlap)
---------------------------------------------------------------
There are two common approaches:

  1. Fixed-size chunking: slice text into chunks of ~N characters/tokens,
     optionally overlapping consecutive chunks so an idea that falls on a
     boundary still appears whole in at least one chunk. Simple, fast,
     no extra model needed, and it's the standard baseline for RAG projects.

  2. Semantic chunking: use an embedding model (or an LLM) to detect topic
     shifts and cut at natural "meaning boundaries" instead of a fixed size.
     Produces more coherent chunks, but is slower, needs an extra model
     call per document, and is harder to reason about / explain simply.

For this project we use fixed-size chunking (~1000 characters, 200-char
overlap), but we don't slice blindly through the middle of a paragraph.
We first split each page into paragraphs, then greedily pack whole
paragraphs into a chunk until adding the next one would exceed the size
budget. This is still "fixed-size chunking" (no embedding model involved)
but respects natural text boundaries, which matters a lot for academic
papers with dense, structured writing.

Why this over pure semantic chunking, given the constraints of this project:
research papers already have strong structural signal (headings, paragraph
breaks, section numbers), so a simple paragraph-aware fixed-size approach
captures most of the benefit of semantic chunking without the added
complexity/cost -- and it's much easier to explain and defend in an
interview than "an embedding model decided where to cut."

Each chunk is tagged with its source PDF filename and the page range it
came from, so later, when the LLM answers a question, we can cite exactly
which paper and pages the answer came from.
"""

import json
import re
from collections import Counter

import fitz  # PyMuPDF
from tqdm import tqdm

import config


def extract_pages(pdf_path):
    """Return a list of (page_number, page_text) tuples for one PDF."""
    doc = fitz.open(pdf_path)
    pages = [(page_num, page.get_text()) for page_num, page in enumerate(doc, start=1)]
    doc.close()
    return pages


def _normalize_line(line):
    """Collapse digits so a running footer like 'Paper Title 5' matches
    the same footer on another page that reads 'Paper Title 12'."""
    return re.sub(r"\d+", "#", line.strip())


def find_boilerplate_lines(pages, min_page_fraction=0.4, max_line_len=200):
    """
    PDFs repeat running headers/footers (paper title, page number, venue
    line) on almost every page. PyMuPDF extracts these as their own short
    text lines, but with no blank line separating them from the body text
    -- so our paragraph splitter (which only breaks on blank lines) was
    gluing them onto the start of real sentences, e.g.:
        "Giancola2024DeepLearning Deep learning for ... videos 5 set of N
         untrimmed videos ..."
    Here "Giancola2024DeepLearning", the title, and "5" (page number) are
    boilerplate; "set of N untrimmed videos ..." is the real sentence.

    We detect boilerplate generically: any short line whose text (with
    digits normalized, so page numbers don't break the match) recurs on a
    large fraction of this PDF's pages is almost certainly a running
    header/footer rather than unique body content, and gets dropped.
    """
    if len(pages) < 3:
        return set()  # too few pages to tell a repeating pattern from coincidence

    counts = Counter()
    for _, text in pages:
        seen_this_page = set()
        for line in text.split("\n"):
            norm = _normalize_line(line)
            if norm and len(norm) <= max_line_len and norm not in seen_this_page:
                counts[norm] += 1
                seen_this_page.add(norm)

    threshold = max(2, int(len(pages) * min_page_fraction))
    return {norm for norm, c in counts.items() if c >= threshold}


def strip_boilerplate(pages, boilerplate_signatures):
    """Remove lines matching a detected header/footer signature from every page."""
    cleaned = []
    for page_num, text in pages:
        kept = [line for line in text.split("\n") if _normalize_line(line) not in boilerplate_signatures]
        cleaned.append((page_num, "\n".join(kept)))
    return cleaned


def pages_to_paragraphs(pages):
    """
    Flatten a list of (page_number, page_text) into a list of
    (page_number, paragraph_text), splitting each page on blank lines.
    """
    paragraphs = []
    for page_num, text in pages:
        for para in text.split("\n\n"):
            para = para.strip().replace("\n", " ")
            # Skip empty strings and junk fragments (e.g. stray page numbers, headers)
            if len(para) > 20:
                paragraphs.append((page_num, para))
    return paragraphs


def chunk_paragraphs(paragraphs, chunk_size, overlap):
    """
    Greedily pack (page_number, paragraph_text) tuples into chunks of
    at most `chunk_size` characters, then prepend the tail of the previous
    chunk to each chunk (except the first) to create `overlap` characters
    of shared context between consecutive chunks.

    Returns a list of dicts: {text, page_start, page_end}.
    """
    raw_chunks = []  # each item: {"text": str, "page_start": int, "page_end": int}
    current_text = ""
    current_pages = []

    def flush():
        if current_text:
            raw_chunks.append({
                "text": current_text,
                "page_start": min(current_pages),
                "page_end": max(current_pages),
            })

    for page_num, para in paragraphs:
        if len(current_text) + len(para) + 2 <= chunk_size:
            current_text = f"{current_text}\n\n{para}".strip() if current_text else para
            current_pages.append(page_num)
        else:
            flush()
            if len(para) > chunk_size:
                # Rare case: a single paragraph is bigger than the whole chunk budget
                # (e.g. a giant unbroken block of text). Hard-split it by characters.
                step = chunk_size - overlap
                for i in range(0, len(para), step):
                    raw_chunks.append({
                        "text": para[i:i + chunk_size],
                        "page_start": page_num,
                        "page_end": page_num,
                    })
                current_text, current_pages = "", []
            else:
                current_text, current_pages = para, [page_num]
    flush()

    # Add overlap: prepend the tail of the previous chunk's text to the current one.
    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev_tail = raw_chunks[i - 1]["text"][-overlap:]
            overlapped.append({
                "text": f"{prev_tail}\n\n{chunk['text']}",
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
            })
    return overlapped


def process_paper(pdf_path):
    """Turn one PDF into a list of chunk dicts, ready to embed."""
    pages = extract_pages(pdf_path)
    boilerplate = find_boilerplate_lines(pages)
    pages = strip_boilerplate(pages, boilerplate)
    paragraphs = pages_to_paragraphs(pages)
    chunks = chunk_paragraphs(paragraphs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "chunk_id": f"{pdf_path.stem}__{i}",
            "source_paper": pdf_path.name,
            "chunk_index": i,
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "text": chunk["text"],
        })
    return records


def main():
    pdf_paths = sorted(config.PAPERS_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {config.PAPERS_DIR}. Add papers there and re-run.")
        return

    all_chunks = []
    for pdf_path in tqdm(pdf_paths, desc="Chunking PDFs"):
        try:
            all_chunks.extend(process_paper(pdf_path))
        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")

    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks from {len(pdf_paths)} papers to {config.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
