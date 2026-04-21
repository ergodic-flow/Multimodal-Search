FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY .python-version ./

RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

COPY knn.py embedder.py ./
COPY static/ static/

ENV HF_HOME=/app/.hf_cache

EXPOSE 8000 8001 8002
