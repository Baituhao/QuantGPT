# syntax=docker/dockerfile:1.7

# ── Stage 1: Build frontend ──────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ─────────────────────────────────────
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ fonts-liberation

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[postgresql]"

COPY quantgpt/ ./quantgpt/
COPY scripts/ ./scripts/
COPY --from=frontend /app/frontend/dist ./frontend/dist

RUN mkdir -p data reports logs

EXPOSE 8003

ENV AUTH_DISABLED=true
ENV QUANTGPT_TASK_BACKEND=process
ENV QUANTGPT_WORKER_PROCESSES=2

CMD ["python", "-m", "quantgpt", "--transport", "http"]
