# ResearchLens retrieval/ingestion service (Person 2).
# NOTE: full ML stack (torch/transformers/chromadb) is heavy (~3-4 GB image,
# ~2 GB RAM at runtime). Render free (512 MB) cannot run it reliably;
# Hugging Face Spaces CPU Basic (16 GB) can. See DEPLOY.md.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=0

WORKDIR /srv/retrieval

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY data/processed_papers.json ./data/processed_papers.json
COPY data/evaluation_questions.json ./data/evaluation_questions.json

RUN useradd -m -u 10001 appuser && mkdir -p /srv/retrieval/data/ingested /tmp/p2_ingest && chown -R appuser:appuser /srv/retrieval /tmp/p2_ingest
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
