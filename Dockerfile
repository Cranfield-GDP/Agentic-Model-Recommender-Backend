FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-alpine

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8000

ENV MODEL="chatgpt"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
