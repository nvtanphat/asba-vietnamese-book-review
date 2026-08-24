.DEFAULT_GOAL := help
.PHONY: help install install-ml install-mlops install-web prepare validate-data train benchmark benchmark-tuned tune tune-all promote kaggle-doctor kaggle-data kaggle-run kaggle-status kaggle-logs kaggle-output kaggle-resume kaggle-collect mlops-doctor mlops-snapshot mlops-profile mlops-drift mlflow-server dvc-init api web data-dashboard dev lint test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install API + absa_core workspace
	uv sync

install-ml: ## Install full fair-benchmark/training dependencies
	uv sync --group ml

install-mlops: ## Install MLflow + DVC MLOps extras
	uv sync --group ml --group mlops

install-web: ## Install Next.js dependencies
	cd apps/web && npm install

prepare: ## Build the frozen 70/15/15 split from the raw v3 dataset
	uv run --group ml python -m ml.data.split --input data/raw/tiki-book-review_merged_fixed_v3.json --output-dir data/splits

validate-data: ## Verify split manifest and leakage constraints
	uv run --group ml python scripts/data/validate_splits.py

train: ## Train one model: make train MODEL=phobert
	uv run --group ml python -m ml.train --model $(MODEL)

benchmark: ## Train/evaluate all eight models with base configs
	uv run --group ml python -m ml.benchmark

benchmark-tuned: ## Fair benchmark using completed equal-budget tuned configs
	uv run --group ml python -m ml.benchmark --use-tuned

tune: ## Tune one model: make tune MODEL=phobert TRIALS=20
	uv run --group ml python -m ml.tune --model $(MODEL) --trials $(or $(TRIALS),20)

tune-all: ## Tune all models with the same completed-trial budget
	uv run --group ml python scripts/training/tune_all.py --trials $(or $(TRIALS),20)

promote: ## Promote best PRIMARY model selected by validation score
	uv run --group ml python scripts/deployment/promote_best.py

kaggle-doctor: ## Check official Kaggle CLI installation/authentication
	python -m tools.kaggle_cli doctor

kaggle-data: ## Create/version private Kaggle data: make kaggle-data KAGGLE_DATASET=user/sentenai-absa-data
	python -m tools.kaggle_cli sync-data --dataset $(KAGGLE_DATASET)

kaggle-run: ## Submit validation-only T4 train: make kaggle-run MODEL=phobert KAGGLE_OWNER=user KAGGLE_DATASET=user/sentenai-absa-data
	python -m tools.kaggle_cli run --owner $(KAGGLE_OWNER) --dataset $(KAGGLE_DATASET) --model $(MODEL)

kaggle-status: ## Show kernel status: make kaggle-status KERNEL=user/sentenai-phobert
	python -m tools.kaggle_cli status --kernel $(KERNEL)

kaggle-logs: ## Stream kernel logs: make kaggle-logs KERNEL=user/sentenai-phobert
	python -m tools.kaggle_cli logs --kernel $(KERNEL) --follow

kaggle-output: ## Download kernel output: make kaggle-output KERNEL=user/sentenai-phobert
	python -m tools.kaggle_cli output --kernel $(KERNEL)

kaggle-resume: ## Resume from latest Kaggle checkpoint via private checkpoint dataset
	python -m tools.kaggle_cli resume --owner $(KAGGLE_OWNER) --dataset $(KAGGLE_DATASET) --model $(MODEL)

kaggle-collect: ## Download Kaggle model output and register candidate
	python -m tools.kaggle_cli collect --owner $(KAGGLE_OWNER) --model $(MODEL) --register

mlops-doctor: ## Check local MLOps backends and optional tools
	python -m mlops doctor

mlops-snapshot: ## Validate + fingerprint current raw data/split lineage
	python -m mlops validate-data && python -m mlops snapshot-data

mlops-profile: ## Build production drift reference from training split
	python -m mlops profile-data --input data/splits/train.json

mlops-drift: ## Compare API telemetry to training reference: make mlops-drift CURRENT=apps/api/data/model_telemetry.jsonl
	python -m mlops drift --current $(CURRENT)

mlflow-server: ## Start local MLflow 3 tracking/registry server on :5000
	mlflow server --host 0.0.0.0 --port 5000

dvc-init: ## Initialize DVC and hand raw dataset to DVC tracking
	python -m mlops bootstrap-dvc

api: ## Run FastAPI gateway on :8000
	uv run uvicorn app.main:app --reload --port 8000 --app-dir apps/api

web: ## Run Next.js dashboard on :3000
	cd apps/web && npm run dev

data-dashboard: ## Optional Streamlit data-quality dashboard retained from research repo
	uv run --group ml streamlit run apps/data_quality/dashboard.py

dev: ## Show local app commands
	@echo "Terminal 1: make api"; echo "Terminal 2: make web"

lint: ## Ruff the maintained Python code (migrated notebook archive excluded)
	uv run ruff check packages apps/api ml mlops tools scripts --exclude scripts/migrated_notebooks

test: ## Unit/integration tests that do not download model weights
	uv run pytest

clean: ## Remove caches; keep experiment/artifact outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache apps/web/.next
