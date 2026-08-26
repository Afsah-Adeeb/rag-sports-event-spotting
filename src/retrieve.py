"""
Step 3: Given a question, find the most relevant chunks.

HOW THIS WORKS
----------------
1. Embed the question with the *same* model used for the chunks
   (all-MiniLM-L6-v2). Question and chunks must live in the same vector
   space for similarity comparison to mean anything.
2. Ask FAISS for the top-k chunk vectors closest to the question vector
   (closest = highest cosine similarity, since both are unit-normalized).
3. Use the returned positions to look up the actual chunk text + source
   paper + page range from our metadata file (see embed_store.py for why
   this two-file split exists).

This module is deliberately just a function library (no CLI) -- Step 4
(generate.py) and Step 6 (cli.py) both import `retrieve()` from here.
"""

import json

import faiss
from sentence_transformers import SentenceTransformer

import config

_model = None
_index = None
_metadata = None


def _load_resources():
    """Lazily load the embedding model, FAISS index, and metadata once per process."""
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    if _index is None:
        _index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    if _metadata is None:
        with open(config.CHUNK_METADATA_PATH, encoding="utf-8") as f:
            _metadata = [json.loads(line) for line in f]
    return _model, _index, _metadata


def embed_texts(texts):
    """Embed a list of texts with the same model/normalization used for the corpus.

    Exposed so callers that build their own index (e.g. the Streamlit app adding
    session-uploaded papers) embed identically to embed_store.py -- vectors from
    two different code paths must live in the same space to be comparable.
    """
    model, _, _ = _load_resources()
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


def base_index_and_metadata():
    """Return the on-disk FAISS index and its metadata list (loaded once per process)."""
    _, index, metadata = _load_resources()
    return index, metadata


def search(question, index, metadata, top_k=config.DEFAULT_TOP_K):
    """Search an arbitrary (index, metadata) pair -- the corpus one, or one the
    caller extended with extra vectors. `metadata[i]` must describe vector `i`."""
    query_vector = embed_texts([question])
    scores, positions = index.search(query_vector, top_k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:  # FAISS pads with -1 if there are fewer than top_k vectors total
            continue
        results.append({**metadata[pos], "score": float(score)})
    return results


def retrieve(question, top_k=config.DEFAULT_TOP_K):
    """
    Return the top_k most relevant chunks for `question`, each as a dict:
    {chunk_id, source_paper, page_start, page_end, text, score}.
    `score` is cosine similarity in [-1, 1] (in practice ~[0, 1] for real text).
    """
    index, metadata = base_index_and_metadata()
    return search(question, index, metadata, top_k=top_k)


if __name__ == "__main__":
    # Quick manual sanity check: run `python retrieve.py "your question"`.
    import sys

    question = " ".join(sys.argv[1:]) or "What is the E2E-Spot architecture?"
    print(f"Question: {question}\n")
    for i, r in enumerate(retrieve(question), start=1):
        print(f"[{i}] score={r['score']:.3f}  {r['source_paper']}  (p.{r['page_start']}-{r['page_end']})")
        print(f"    {r['text'][:200]}...\n")
