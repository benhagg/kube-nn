FROM python:3.11-slim

WORKDIR /app

# Install the CPU-only PyTorch wheel so the image does not include CUDA libraries.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir numpy tiktoken

COPY . /app
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
