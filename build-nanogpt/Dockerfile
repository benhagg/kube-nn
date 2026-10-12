FROM python:3.11-slim

LABEL org.opencontainers.image.title="train GPT"

WORKDIR /app

# Install the CPU-only PyTorch wheel so the image does not include CUDA libraries.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir numpy python-dotenv

COPY . /app
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
