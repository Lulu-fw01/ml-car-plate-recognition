ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# Install system dependencies including curl for uv
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY uv.lock pyproject.toml ./

# Install dependencies (uses cache if lock unchanged)
RUN uv sync --frozen --no-install-project

COPY configs/ ./configs/
COPY src/ ./src/
COPY proto/ ./proto/

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

EXPOSE 50051

ENTRYPOINT ["uv", "run", "python", "src/server.py"]
