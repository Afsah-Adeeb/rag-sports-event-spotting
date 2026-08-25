"""
Step 4: Pass retrieved chunks + the question to an LLM to produce an answer.

This is the "G" in RAG. Retrieval found the relevant text; this step asks
Gemini to read that text and write a grounded answer, instead of relying
on whatever it memorized during training.

PROMPT DESIGN
---------------
The prompt has three deliberate rules baked in:
  1. Only answer from the provided context (this is what makes it
     "retrieval-augmented" rather than just chatting with a generic LLM --
     the model's job is to synthesize the retrieved text, not recall facts
     from its own training data).
  2. Say so explicitly if the context doesn't contain the answer, instead
     of guessing. This matters a lot for evaluation later (Step 5): we want
     to be able to tell "hallucinated" apart from "correctly said I don't know."
  3. Cite which paper(s) it drew from, using the source_paper/page metadata
     we tracked all the way from ingest.py. This is what lets you (and an
     interviewer) verify an answer is actually grounded, not just plausible.

We use gemini-2.0-flash: it's on Gemini's free tier, fast, and more than
capable for summarizing a handful of retrieved paragraphs.
"""

import os

from dotenv import load_dotenv
from google import genai

import config
from retrieve import retrieve

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
    context_blocks = []
    for chunk in retrieved_chunks:
        source = f"[{chunk['source_paper']}, p.{chunk['page_start']}-{chunk['page_end']}]"
        context_blocks.append(f"{source}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are a research assistant answering questions about a collection of \
academic papers on sports video event-spotting / temporal action localization.

Answer the question using ONLY the context below. Follow these rules:
- If the context does not contain enough information to answer, say so explicitly \
instead of guessing or using outside knowledge.
- Cite which paper(s) you used for each part of your answer, using the filenames \
shown in brackets before each context block.
- Be concise and technical -- this is for a researcher, not a general audience.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer_from_chunks(question, retrieved_chunks):
    """Ask Gemini to answer `question` using an already-retrieved list of chunks.

    Split out from generate_answer() so callers that already retrieved chunks
    (e.g. evaluate.py, which needs the chunks anyway for precision@k) don't
    pay for a second, redundant embedding + FAISS search of the same question.
    """
    prompt = build_prompt(question, retrieved_chunks)
    client = _get_client()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL_NAME,
        contents=prompt,
    )
    return response.text


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
