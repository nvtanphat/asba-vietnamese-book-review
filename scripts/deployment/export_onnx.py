"""ONNX export remains available for the legacy PhoBERT production model.
For the unified winner, export should be performed only after promotion and parity-tested;
see docs/deployment.md. This guard prevents silently exporting a mismatched architecture.
"""
raise SystemExit("Unified ONNX export is intentionally gated. Promote a model first, then follow docs/deployment.md. Legacy exporter: packages/absa_core/scripts/export_onnx.py")
