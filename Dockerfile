ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

COPY uv.lock pyproject.toml ./

ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && /install.sh && rm /install.sh
ENV PATH="/root/.local/bin:$PATH"

RUN uv sync --frozen --no-install-project --no-dev


WORKDIR /app

FROM ${BASE_IMAGE}
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY configs/ ./configs/
COPY src/ ./src/
COPY proto/ ./proto/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    TORCH_INDEX_URL=https://pytorch.org

EXPOSE 50051

ENTRYPOINT ["python", "src/server.py"]
