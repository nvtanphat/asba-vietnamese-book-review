# syntax=docker/dockerfile:1
# SentenAI — API image (FastAPI + PhoBERT ABSA inference)

# ---- builder: resolve + install Python deps into an isolated venv ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN pip install --no-cache-dir uv==0.10.5

# Manifests first for layer caching — the (slow, torch-heavy) install layer only
# reruns when a dependency actually changes, not on every source-code edit.
COPY pyproject.toml ./
COPY packages/absa_core/pyproject.toml packages/absa_core/pyproject.toml
COPY apps/api/pyproject.toml apps/api/pyproject.toml

RUN uv venv /opt/venv

COPY packages/absa_core ./packages/absa_core
COPY apps/api ./apps/api
COPY artifacts/final ./artifacts/final
# This service never gets GPU passthrough (see docker-compose.yml), but plain PyPI
# `torch` bundles the full CUDA runtime (~2GB of nvidia-*/triton wheels) regardless —
# pre-install the CPU-only build so the real install below has nothing left to fetch.
RUN uv pip install --python /opt/venv/bin/python torch --extra-index-url https://download.pytorch.org/whl/cpu
# apps/api's pyproject.toml marks absa-core as `{ workspace = true }`, so uv installs it
# editable (a .pth pointing back at /build/packages/absa_core) by default — that path
# doesn't exist in the runtime stage below, so the API would fail every request with
# `ModuleNotFoundError: No module named 'absa_core'`. --no-sources makes both install as
# real (non-editable) wheels instead of following the workspace path source.
RUN uv pip install --python /opt/venv/bin/python --no-sources ./packages/absa_core ./apps/api

# ---- runtime: just the built venv + app source, no compilers or package manager ----
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    WEB_CONCURRENCY=2

# Unprivileged, non-login system user — the API never needs root once installed.
RUN groupadd --system app && \
    useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/apps/api /app/apps/api
COPY --from=builder /build/artifacts/final /app/artifacts/final

WORKDIR /app/apps/api
# SQLite's default DATABASE_URL lives under ./data — must exist and be writable by
# the unprivileged user *before* USER switches away from root.
RUN mkdir -p data && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Multiple worker *processes* for real concurrency (the PhoBERT forward pass holds the
# GIL, so threads alone don't parallelize it); each process tunes its own torch thread
# count down via ABSA_TORCH_THREADS (see app/core/config.py) to avoid CPU contention
# between workers. Override WEB_CONCURRENCY at deploy time to match the host's core count.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
