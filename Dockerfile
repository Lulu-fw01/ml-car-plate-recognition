ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /install.sh
RUN sh /install.sh && rm /install.sh
ENV PATH="/root/.local/bin:$PATH"

# Copy lock file and pyproject.toml first (for layer caching)
COPY uv.lock pyproject.toml ./

# Install all dependencies (uv syncs from lock file)
RUN uv sync --refresh

# Copy remaining project files
COPY configs/ ./configs/
COPY src/ ./src/
COPY proto/ ./proto/
COPY runs/ ./runs/
COPY mlartifacts/ ./mlartifacts/

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

EXPOSE 50051

ENTRYPOINT ["uv", "run", "python", "src/server.py"]
