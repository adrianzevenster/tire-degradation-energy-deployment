FROM python:3.13-slim

ARG F1_BUILD_SHA=unknown
ARG F1_BUILD_DATE=unknown

LABEL org.opencontainers.image.title="f1-tire-energy-strategy" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.revision="${F1_BUILD_SHA}" \
      org.opencontainers.image.created="${F1_BUILD_DATE}"

ENV F1_BUILD_SHA=${F1_BUILD_SHA} \
    F1_BUILD_DATE=${F1_BUILD_DATE} \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -e ".[api,observability,persistence,ml]"

EXPOSE 8000
CMD ["uvicorn", "f1_strategy.api:app", "--host", "0.0.0.0", "--port", "8000"]
