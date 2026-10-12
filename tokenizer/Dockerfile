FROM python:3.11-slim

LABEL org.opencontainers.image.title="train GPT tokenizer"

WORKDIR /app

RUN pip install --no-cache-dir numpy tiktoken

COPY tokenizer/tokenize.py /app/tokenize_dataset.py
COPY input.txt /input.txt

ENTRYPOINT ["python", "/app/tokenize_dataset.py"]
