# =============================================================================
#  ARES · Autonomous Research & Multi-Agent Evaluation Engine
#  Container image
# =============================================================================
# A slim, non-root Python image that runs the ARES CLI. Configuration is passed
# at runtime via environment variables (see .env.example) rather than baked into
# the image, so the same image works for the NVIDIA NIM and Ollama providers.
#
# Build:
#   docker build -t ares:latest .
#
# Run (interactive, NVIDIA provider):
#   docker run --rm -it --env-file .env \
#     -v ares_data:/data ares:latest --topic "..."
#
# See README.md · "🐳 Running with Docker" for the full guide.
# =============================================================================

FROM python:3.11-slim AS base

# --- OS-level hardening / runtime hygiene ------------------------------------
# PYTHONUNBUFFERED  : stream logs immediately (important for the live CLI).
# PYTHONDONTWRITEBYTECODE : keep the image/layers clean of .pyc files.
# PIP_NO_CACHE_DIR  : smaller image.
# USE_TF=0          : ARES sets this in-code too, but exporting it early keeps
#                     transformers from ever probing for TensorFlow at import.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    USE_TF=0 \
    # Default the persistent artefacts into a mountable /data volume so runs
    # survive container restarts (override any of these at `docker run` time).
    CHECKPOINT_DB=/data/ares_checkpoints.sqlite \
    ARES_OUTPUT=/data/research_report.md

WORKDIR /app

# --- Python dependencies (cached layer) --------------------------------------
# Copy only requirements first so `pip install` is cached unless deps change.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# --- Application source -------------------------------------------------------
COPY ARES.py .
COPY LICENSE README.md ./

# --- Persistent data volume ---------------------------------------------------
# SQLite checkpoints + generated reports live here. Declared as a VOLUME so
# `docker run -v ares_data:/data ...` (or a compose named volume) persists them.
RUN mkdir -p /data
VOLUME ["/data"]

# --- Non-root runtime user ----------------------------------------------------
# Run as an unprivileged user and hand it ownership of the app + data dirs.
RUN useradd --create-home --uid 10001 ares && \
    chown -R ares:ares /app /data
USER ares

# The container IS the ARES CLI: `docker run ares:latest <flags>` forwards flags
# straight to `python ARES.py`. Override the entrypoint for a shell if needed.
ENTRYPOINT ["python", "ARES.py"]
# Default flags when none are supplied (non-interactive demo run). Override by
# passing your own args after the image name, e.g. `... ares:latest --topic "X"`.
CMD ["--help"]
