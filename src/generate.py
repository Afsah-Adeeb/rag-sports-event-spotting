"""
Step 4: Pass retrieved chunks + the question to an LLM to produce an answer.

This is the "G" in RAG. Retrieval found the relevant text; this step asks
Gemini to read that text and write a grounded answer, instead of relying
on whatever it memorized during training.

PROMPT DESIGN
---------------
The prompt has two deliberate rules baked in:
  1. Only answer from the provided context (this is what makes it
     "retrieval-augmented" rather than just chatting with a generic LLM --
     the model's job is to synthesize the retrieved text, not recall facts
     from its own training data).
  2. Say so explicitly if the context doesn't contain the answer, instead
     of guessing. This matters a lot for evaluation later (Step 5): we want
     to be able to tell "hallucinated" apart from "correctly said I don't know."

The model is deliberately NOT asked to cite sources inline in the answer
text (no "[paper.pdf, p.5]" scattered through the prose) -- that reads as
clutter compared to a normal AI chat answer. Grounding is still fully
checkable: the retrieved chunks (source_paper/page metadata tracked all the
way from ingest.py) are returned alongside the answer and shown in a
collapsed "Sources" section by the UI, the same pattern ChatGPT/Claude/
Gemini use for citations.

We use gemini-2.0-flash: it's on Gemini's free tier, fast, and more than
capable for summarizing a handful of retrieved paragraphs.
"""

import os
import time

from dotenv import load_dotenv
from google import genai

import config
from retrieve import retrieve, search

load_dotenv()  # reads GEMINI_API_KEY from a local .env file (see .env.example)

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and add your key "
                "(get one free at https://aistudio.google.com/apikey)."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def build_prompt(question, retrieved_chunks):
    """Combine the question with retrieved context into one prompt for Gemini."""
    context_blocks = [chunk["text"] for chunk in retrieved_chunks]
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are a research assistant answering questions about a collection of \
academic papers on sports video event-spotting / temporal action localization.

Answer the question using ONLY the context below. Follow these rules:
- If the context does not contain enough information to answer, say so explicitly \
instead of guessing or using outside knowledge.
- Write a clean, natural answer, like a normal AI chat assistant -- do NOT insert \
inline citations, brackets, or paper filenames into the text. Sources are shown \
separately by the UI, not in the prose.
- Be concise and technical -- this is for a researcher, not a general audience.

Context:
{context}

Question: {question}

Answer:"""


def _extract_usage(response):
    """Pull token counts off a Gemini response, tolerating SDK shape changes.

    Token counts are how the monitoring layer reports cost, but they're
    metadata -- if a future SDK version renames these fields, that must not
    break answering. Missing counts simply show as unknown in the dashboard.
    """
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {"input_tokens": None, "output_tokens": None}
    return {
        "input_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
    }


def generate_with_usage(question, retrieved_chunks, model=None):
    """Answer from chunks, returning the text plus cost/latency metadata.

    This is the core call. It returns a dict rather than a bare string so the
    monitoring layer (telemetry.py) can record how long generation took and
    how many tokens it burned -- measured here, at the actual API boundary,
    rather than estimated by the caller.

    `model` overrides config.GEMINI_MODEL_NAME. compare_rag.py uses it to pin
    both arms of the benchmark to one model, so the comparison measures
    retrieval strategy rather than the difference between two models.

    Returns: {"answer": str, "generation_ms": float,
              "input_tokens": int|None, "output_tokens": int|None}
    """
    prompt = build_prompt(question, retrieved_chunks)
    client = _get_client()

    started = time.perf_counter()
    response = client.models.generate_content(
        model=model or config.GEMINI_MODEL_NAME,
        contents=prompt,
    )
    generation_ms = (time.perf_counter() - started) * 1000

    return {
        "answer": response.text,
        "generation_ms": generation_ms,
        **_extract_usage(response),
    }


def generate_answer_from_chunks(question, retrieved_chunks):
    """Ask Gemini to answer `question` using an already-retrieved list of chunks.

    Split out from generate_answer() so callers that already retrieved chunks
    (e.g. evaluate.py, which needs the chunks anyway for precision@k) don't
    pay for a second, redundant embedding + FAISS search of the same question.

    Thin wrapper over generate_with_usage() for callers that only want the text.
    """
    return generate_with_usage(question, retrieved_chunks)["answer"]


def answer_traced(question, top_k=config.DEFAULT_TOP_K, index=None, metadata=None, model=None):
    """Run the full pipeline and time each stage separately.

    Both front ends (cli.py, app.py) go through this so the monitoring layer
    sees identical fields no matter where a question came from. Splitting the
    two timings matters: retrieval and generation have wildly different cost
    profiles, and reporting only a combined number hides which one to optimize.

    Pass `index`/`metadata` to search a caller-built index instead of the
    committed one -- that's how app.py includes session-uploaded papers.

    Returns: {"answer", "chunks", "retrieval_ms", "generation_ms",
              "input_tokens", "output_tokens"}
    """
    started = time.perf_counter()
    if index is not None:
        chunks = search(question, index, metadata, top_k=top_k)
    else:
        chunks = retrieve(question, top_k=top_k)
    retrieval_ms = (time.perf_counter() - started) * 1000

    result = generate_with_usage(question, chunks, model=model)
    return {"chunks": chunks, "retrieval_ms": retrieval_ms, **result}


def generate_answer(question, top_k=config.DEFAULT_TOP_K):
    """Retrieve relevant chunks, then ask Gemini to answer using only that context.

    Returns (answer_text, retrieved_chunks) so callers can show sources alongside the answer.
    """
    retrieved_chunks = retrieve(question, top_k=top_k)
    answer = generate_answer_from_chunks(question, retrieved_chunks)
    return answer, retrieved_chunks


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is the E2E-Spot architecture?"
    answer, sources = generate_answer(question)

    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}\n")
    print("Sources used:")
    for chunk in sources:
        print(f"  - {chunk['source_paper']} (p.{chunk['page_start']}-{chunk['page_end']}, score={chunk['score']:.3f})")
