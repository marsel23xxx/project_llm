FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install fastapi "uvicorn>=0.30" duckdb pandas numpy scipy scikit-learn

WORKDIR /sandbox
COPY app.py runner.py ./

# A different uid from the backend: dataset files are world-readable but never writable here.
RUN useradd --no-create-home --uid 1001 sandbox
USER sandbox

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
