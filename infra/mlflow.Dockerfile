FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir "mlflow>=3.1,<4"
RUN useradd --system --create-home --uid 10001 mlflow && mkdir -p /mlflow/artifacts && chown -R mlflow:mlflow /mlflow
USER mlflow
WORKDIR /mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts"]
