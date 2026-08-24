"""Application settings, loaded from environment / `.env`."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Exposed as a constant (not just inlined in the default=) so main.py's startup check
# can compare against the exact same value without the two literals drifting apart.
DEFAULT_JWT_SECRET = "change-me-in-production-use-openssl-rand-hex-32"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # "development" | "production" — only gates the JWT-secret startup check below.
    environment: str = "development"

    # Database (shop/user auth)
    database_url: str = "sqlite:///./data/app.db"

    # ABSA model
    absa_artifact_dir: str = "artifacts/final"
    absa_prefer_unified_artifact: bool = True
    absa_model_id: str = "hoangloc112/ABSA-TIKI-BOOK"
    absa_model_variant: str = "phobert"
    # Intra-op CPU threads PyTorch uses per process. Keep this low (1-2) when the API
    # runs with multiple worker *processes* (see infra/api.Dockerfile) — otherwise every
    # worker's torch call fans out across all CPU cores and they contend with each other,
    # which hurts p99 latency under concurrent requests far more than it helps a single one.
    absa_torch_threads: int = 1
    # Opt-in CPU cost optimization: serve from the ONNX Runtime + INT8 export instead of
    # the full-precision torch model (~4x smaller, ~2x faster on CPU in local benchmarks —
    # see packages/absa_core/scripts/export_onnx.py). Off by default: quantization trades
    # a little precision on borderline (uncalibrated) aspect-presence calls, so this is a
    # deliberate choice to make, not a silent default. Generate the file first via
    # `uv run python packages/absa_core/scripts/export_onnx.py`, then set both.
    absa_use_onnx: bool = False
    absa_onnx_model_path: str = "data/models/onnx/absa_phobert.int8.onnx"

    # Privacy-preserving model telemetry for drift/latency monitoring. Raw review text
    # is never persisted; only a short hash, coarse text features and predictions.
    absa_telemetry_enabled: bool = True
    absa_telemetry_path: str = "data/model_telemetry.jsonl"
    absa_telemetry_sample_rate: float = 1.0

    # JWT Auth
    # NOTE: this default is a placeholder for local/demo use only. Set a real random secret
    # via the JWT_SECRET env var (.env) for any deployment beyond your own laptop.
    jwt_secret: str = DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60 * 24  # 24 hours

    # Server / CORS
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

