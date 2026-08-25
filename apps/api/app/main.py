"""SentenAI API gateway — FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from app.core.config import DEFAULT_JWT_SECRET, get_settings
from app.core.limiter import limiter
from app.db.base import Base
from app.db.session import engine
from app.routers import absa, auth, reviews, templates
from app.routers import settings as settings_router

# Arbitrary fixed key for the Postgres advisory lock in _ensure_tables — any int64 works,
# it just has to be the same constant every time this process (or a sibling worker) calls it.
_SCHEMA_INIT_LOCK_KEY = 727271

settings = get_settings()

app = FastAPI(
    title="SentenAI API",
    version="0.3.0",
    description="PhoBERT ABSA inference gateway for the SentenAI prediction dashboard.",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _ensure_tables() -> None:
    """Create tables if missing (seed.py populates demo data separately).

    Under multiple worker *processes* (see infra/api.Dockerfile's --workers), every
    worker runs this startup hook independently. Against Postgres that's a genuine
    race: two workers can both see "table missing" and both issue CREATE TABLE, and
    the loser fails with a duplicate-sequence/index error instead of silently no-op'ing
    — reproduced while hardening the Docker setup for multi-worker deployment. A
    Postgres advisory lock serializes the DDL across workers; SQLite has no concurrent
    writers in dev, so it skips straight to create_all.
    """
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _SCHEMA_INIT_LOCK_KEY})
            try:
                Base.metadata.create_all(bind=engine)
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _SCHEMA_INIT_LOCK_KEY})
    else:
        Base.metadata.create_all(bind=engine)


@app.on_event("startup")
def _check_jwt_secret() -> None:
    if settings.jwt_secret != DEFAULT_JWT_SECRET:
        return
    if settings.environment == "production":
        raise RuntimeError(
            "Refusing to start with the default JWT_SECRET in a production environment. "
            "Set a real random secret (e.g. `openssl rand -hex 32`) via the JWT_SECRET env var."
        )
    import warnings

    warnings.warn(
        "Using the default JWT_SECRET placeholder — fine for local dev, but every token "
        "signed with it is forgeable by anyone who reads this source. Set JWT_SECRET before "
        "deploying anywhere real.",
        stacklevel=1,
    )


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": "sentenai-api"}


@app.get("/model-info", tags=["meta"])
def model_info() -> dict:
    """Expose deployed artifact identity without forcing heavy model loading."""
    import json
    from pathlib import Path

    meta = Path(settings.absa_artifact_dir) / "metadata.json"
    if not meta.exists():
        return {
            "source": "fallback",
            "model_id": settings.absa_model_id,
            "variant": settings.absa_model_variant,
        }
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return {
        "source": "promoted_artifact",
        "model": payload.get("model"),
        "selected_by": payload.get("selected_by"),
        "version": payload.get("registry_version"),
        "data_fingerprint": payload.get("data_fingerprint"),
    }


app.include_router(auth.router)
app.include_router(absa.router)
app.include_router(reviews.router)
app.include_router(settings_router.router)
app.include_router(templates.router)
