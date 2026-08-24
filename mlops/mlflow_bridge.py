from __future__ import annotations

from pathlib import Path


def register_unified_artifact(
    artifact_dir: str | Path,
    *,
    registered_model_name: str = "sentenai-absa",
    alias: str | None = None,
    tracking_uri: str | None = None,
) -> dict:
    """Log a promoted SentenAI artifact as an MLflow 3 pyfunc model and optionally set an alias."""
    try:
        import mlflow
        import pandas as pd
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Install with: uv sync --group mlops") from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    artifact_dir = Path(artifact_dir).resolve()
    if not (artifact_dir / "metadata.json").exists():
        raise FileNotFoundError(f"Not a promoted SentenAI artifact: {artifact_dir}")

    class SentenAIPyfunc(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from absa_core.models import UnifiedArtifactPredictor
            self.predictor = UnifiedArtifactPredictor(context.artifacts["sentenai_artifact"], device="cpu")

        def predict(self, context, model_input, params=None):
            if isinstance(model_input, pd.DataFrame):
                texts = model_input["text"].astype(str).tolist()
            else:
                texts = [str(x) for x in model_input]
            return pd.DataFrame(self.predictor.predict(texts))

    mlflow.set_experiment("sentenai-absa-registry")
    with mlflow.start_run() as run:
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=SentenAIPyfunc(),
            artifacts={"sentenai_artifact": str(artifact_dir)},
            registered_model_name=registered_model_name,
            input_example=pd.DataFrame({"text": ["Sách hay, giao hàng nhanh"]}),
        )
    version = getattr(info, "registered_model_version", None)
    if alias and version is not None:
        MlflowClient().set_registered_model_alias(registered_model_name, alias, str(version))
    return {"run_id": run.info.run_id, "model_uri": info.model_uri, "version": version, "alias": alias}
