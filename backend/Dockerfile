FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies against a stub package first, so code changes reuse this cached layer.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py && pip install ".[onnx]" && rm -rf app

# uvicorn puts the working directory first on sys.path, so this copy is what runs.
COPY app ./app
COPY models ./models

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app:app /data
USER app

EXPOSE 8000
# The port is published on 127.0.0.1 only, so every caller is the host's Nginx (or the frontend
# container); its X-Forwarded-For carries the visitor's address for rate limits.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
