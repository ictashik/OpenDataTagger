# ODT Stable Diffusion server. Heavy ML deps (torch/diffusers) live only here —
# see README.md for the GPU vs CPU tradeoffs, they apply the same way in Docker.
FROM python:3.11-slim

WORKDIR /app

# For CUDA, pass the matching wheel index at build time, e.g.:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .
# Left empty, torch installs its default (CPU-only) build from requirements.txt.
ARG TORCH_INDEX_URL=""

COPY requirements.txt .
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
      pip install --no-cache-dir torch --index-url "$TORCH_INDEX_URL"; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
