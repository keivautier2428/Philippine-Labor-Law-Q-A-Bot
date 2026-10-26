# syntax=docker/dockerfile:1
# Philippine Labor Law Q&A Bot - Checkpoint 4
# Two stages: "builder" has pip + the model download + index build; "runtime" is a
# slim image holding only the finished virtualenv, the model cache, the baked
# vector index and the source. The API key is NEVER baked in: it arrives at run time.

# ---------- stage 1: build dependencies, model cache and vector index ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/opt/hf
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch. The default Linux wheel drags in GBs of CUDA libraries that a CPU host cannot use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Bake the embedding model into the image so the container never downloads at start-up.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Build the vector index with the SAME pipeline used in development.
WORKDIR /home/user/app
COPY src ./src
COPY data/raw ./data/raw
COPY data/incoming ./data/incoming
RUN python -m src.ingestion.pipeline

# ---------- stage 2: slim runtime ----------
FROM python:3.12-slim AS runtime
# Hugging Face Spaces runs containers as UID 1000; running non-root is good practice anywhere.
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf HF_HUB_OFFLINE=1 PORT=7860
WORKDIR /home/user/app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf
COPY --from=builder --chown=user:user /home/user/app /home/user/app
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('PORT','7860'), timeout=3)"

# GEMINI_API_KEY (and optional APP_* limits) are supplied at run time: docker run -e / Space secrets.
CMD ["sh", "-c", "streamlit run src/app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
