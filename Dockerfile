# syntax=docker/dockerfile:1

# ---- Base image -----------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install uv for fast, reproducible dependency installs.
RUN pip install --no-cache-dir uv

# Copy project metadata first to leverage Docker layer caching.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the project (and its runtime dependencies) into the system env.
RUN uv pip install --system --no-cache .

# Copy the remaining source.
COPY . .

EXPOSE 8000 8501

# Default command runs the API; docker-compose overrides for the UI service.
CMD ["uvicorn", "agentic_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
