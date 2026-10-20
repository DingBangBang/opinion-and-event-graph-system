# syntax=docker/dockerfile:1
FROM python:3.10-slim-bookworm

ARG INSTALL_ML=false
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install --upgrade pip --index-url "$PIP_INDEX_URL" \
    && pip install --index-url "$PIP_INDEX_URL" -r requirements.txt
# Optional ~2GB Transformer stack; enable with:
#   docker compose build --build-arg INSTALL_ML=true
RUN if [ "$INSTALL_ML" = "true" ]; then pip install --index-url "$PIP_INDEX_URL" -r requirements-ml.txt; fi

COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY notebooks ./notebooks
COPY README.md ./

RUN mkdir -p data reports data/logs .cache/huggingface

EXPOSE 8000

CMD ["python", "-m", "src.stream_processor"]
