"""
Central place for every path and knob used across the project.
Keeping these here (instead of scattered magic numbers in each script)
means you only change one place when tuning the pipeline, and it's the
first file you'd show an interviewer to explain your setup.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"          # drop your PDFs here
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"    # output of ingest.py
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNK_METADATA_PATH = VECTOR_STORE_DIR / "chunk_metadata.jsonl"

# --- Chunking ------------------------------------------------------------
# Measured in characters, not tokens, to keep things simple and dependency-free.
# ~1000 chars is roughly 200-250 words / ~250-300 tokens for English text.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # chars shared between consecutive chunks, to avoid cutting ideas in half

# --- Embedding model -------------------------------------------------------
# Local, free, CPU-friendly sentence-transformers model. 384-dim output.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Generation model (Gemini) --------------------------------------------
GEMINI_MODEL_NAME = "gemini-3.6-flash"

# --- Retrieval -------------------------------------------------------------
DEFAULT_TOP_K = 5

# --- Evaluation --------------------------------------------------------
TEST_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "test_questions.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "eval" / "results.md"
EVAL_K_VALUES = [1, 3, 5]  # report precision/hit rate at each of these k values
