FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HOME=/tmp

RUN pip install tensorflow-cpu fastapi "uvicorn>=0.30"

WORKDIR /forecast
COPY app.py forecasting.py ./

RUN useradd --no-create-home --uid 1002 forecast
USER forecast

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
