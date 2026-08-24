# SentenAI Unified API

FastAPI gateway for the merged SentenAI ABSA research + production system.

The inference service first looks for the artifact promoted by the fair benchmark in `artifacts/final/`. If no promoted artifact exists, it keeps backward compatibility with the original PhoBERT/ONNX predictor.

Main endpoints include:

- `POST /absa/analyze` — analyze one review;
- review/history endpoints used by the dashboard;
- authentication and settings endpoints preserved from the original dashboard repo;
- Tiki review fetching preserved from the production repo.

Run from the repository root:

```bash
make api
```

Open the generated FastAPI documentation at `/docs`.
