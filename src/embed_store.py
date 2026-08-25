"""
Step 2: Turn chunks into embeddings and build a FAISS vector index.

WHAT IS AN EMBEDDING?
----------------------
An embedding model maps a piece of text to a fixed-length vector of numbers
(here, 384 numbers) such that texts with similar meaning end up as vectors
that point in similar directions. "What is the E2E-Spot architecture?" and
"Describe the E2E-Spot model design" would land close together, even though
they share few exact words -- this is what lets us do *semantic* search
instead of keyword matching.

We use all-MiniLM-L6-v2 (via sentence-transformers): it runs locally on
CPU, is free, and is small/fast enough to embed ~800 chunks in seconds.

WHY FAISS + COSINE SIMILARITY
-------------------------------
We normalize every embedding to unit length, then use FAISS's
IndexFlatIP (inner product). For unit-length vectors, inner product is
mathematically equivalent to cosine similarity -- it measures the angle
between two vectors (how similar in *direction*, ignoring magnitude),
which is the standard similarity measure for text embeddings.

"Flat" means FAISS does an exact brute-force comparison against every
stored vector -- no approximation. That's fine (and simplest to reason
about) at our scale: a few thousand chunks. Approximate indexes (IVF,
HNSW) only start to matter once you have hundreds of thousands+ vectors
and brute force gets too slow.

WHY WE ALSO SAVE A SEPARATE METADATA FILE
-------------------------------------------
FAISS only stores vectors and hands back integer positions (e.g. "result
#482") -- it knows nothing about source papers or page numbers. So we
save chunk metadata (source_paper, page range, text) to a parallel JSONL
file, in the exact same order the vectors were added to the index. Row i
in that file always corresponds to vector i in the FAISS index -- that's
how we translate a FAISS search result back into "this came from CALF.pdf,
page 4."
"""

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


def load_chunks():
    chunks = []
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_chunks(chunks, model):
    texts = [c["text"] for c in chunks]
    # normalize_embeddings=True gives unit-length vectors, required for
    # inner product to behave as cosine similarity (see module docstring).
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")  # FAISS requires float32


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine similarity for unit vectors
    index.add(embeddings)
    return index


def main():
    chunks = load_chunks()
    if not chunks:
        print(f"No chunks found at {config.CHUNKS_PATH}. Run ingest.py first.")
        return

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks, model)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    # Save metadata in the same order as the vectors, so FAISS result
    # position i maps directly to metadata line i.
    with open(config.CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved index ({index.ntotal} vectors, dim={index.d}) to {config.FAISS_INDEX_PATH}")
    print(f"Saved metadata to {config.CHUNK_METADATA_PATH}")


if __name__ == "__main__":
    main()
