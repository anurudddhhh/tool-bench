# ==============================================================================
# toolbench: Reproducible MCP LLM Tool Selection Benchmarking Suite (F11g)
# ==============================================================================

FROM python:3.12-slim AS base

# Install system dependencies (git for git MCP server, curl, ca-certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary from official Astral distribution
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

# Set working directory and non-root execution path
WORKDIR /app

# Enable bytecode compilation and unbuffered IO
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Copy package dependency definition
COPY pyproject.toml ./

# Install dependencies into virtual environment
RUN uv sync --no-dev --no-cache

# Copy project source code and default configuration
COPY src/ ./src/
COPY config.yaml ./

# Install local package in non-editable mode
RUN uv pip install --no-deps .

# Create persistent state directories
RUN mkdir -p data results cache

# Default entrypoint for CLI command execution
ENTRYPOINT ["toolbench"]
CMD ["--help"]
