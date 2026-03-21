FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up project
WORKDIR /workspace
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy autoresearch files
COPY autoresearch/prepare.py autoresearch/train.py autoresearch/

# Git init for the experiment loop
RUN git config --global user.email "ykumards@gmail.com" && \
    git config --global user.name "Yogesh Kumar" && \
    cd autoresearch && git init && git add -A && git commit -m "baseline"

WORKDIR /workspace/autoresearch

# Default: run one experiment
CMD ["uv", "run", "--no-sync", "python", "train.py"]
