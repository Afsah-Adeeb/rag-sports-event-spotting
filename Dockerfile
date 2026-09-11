# Container image for deploying the Streamlit app to Google Cloud Run.
#
# WHY THIS FILE LOOKS THE WAY IT DOES
# ------------------------------------
# Two decisions here are worth more than the rest of the file combined, and
# both are about cold-start time. Cloud Run scales to zero, so an idle app has
# no running instance and the next visitor pays for the whole startup. Getting
# that wrong is the difference between a demo that feels broken and one that
# feels slow-but-fine.
#
#   1. CPU-ONLY PYTORCH, INSTALLED FIRST.
#      sentence-transformers depends on torch. `pip install torch` defaults to
#      the CUDA build, which drags in several gigabytes of NVIDIA libraries
#      that are pure dead weight on Cloud Run -- there is no GPU. Installing
#      the CPU wheel first means torch is already satisfied when
#      requirements.txt is processed, and the image drops from roughly 4GB to
#      well under 1GB. A smaller image is pulled and started faster on every
#      cold start.
#
#   2. THE EMBEDDING MODEL IS BAKED IN AT BUILD TIME.
#      all-MiniLM-L6-v2 is ~90MB and sentence-transformers downloads it from
#      HuggingFace on first use. Left to runtime that download happens inside
#      the first request of every cold start, and fails outright if HuggingFace
#      is slow or down. Downloading it during the build makes startup a local
#      file read and removes a third-party service from the request path.
#
# The FAISS index and the PDFs are already committed to the repo, so there is
# no build step for them -- they are copied in and the app serves immediately.

FROM python:3.11-slim

# PORT is what Cloud Run injects and expects the container to listen on.
# Defaulted here so `docker run` works locally without setting it.
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Keep HuggingFace's cache at a fixed path so the model baked in below is
    # the same one found at runtime.
    HF_HOME=/opt/hf \
    # The app never trains or fine-tunes; telemetry from these libraries is
    # noise in Cloud Logging.
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /app

# See note 1 above: CPU-only torch, before anything that depends on it.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# See note 2 above: pull the embedding model now, not on the first request.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# Application code and the committed corpus + prebuilt FAISS index.
COPY src/ ./src/
COPY data/ ./data/
COPY eval/ ./eval/
COPY .streamlit/ ./.streamlit/

# Streamlit defaults are wrong for a container:
#   --server.address=0.0.0.0  bind all interfaces, not just localhost, or Cloud
#                             Run's health check cannot reach the process
#   --server.port=$PORT       Cloud Run chooses the port; hardcoding 8501 fails
#   --server.headless=true    do not try to open a browser or prompt for email
#   --fileWatcherType=none    the source never changes in an image; the watcher
#                             just burns CPU and inotify handles
#   --enableCORS=false        Cloud Run terminates TLS upstream, so Streamlit's
#                             origin check sees a mismatch and rejects the
#                             websocket the app needs to function
# Shell form on purpose: $PORT has to be expanded at runtime, not at build.
CMD streamlit run src/app.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --server.headless=true \
    --server.fileWatcherType=none \
    --server.enableCORS=false
