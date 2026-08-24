# BRIEFING — 2026-08-24T00:28:44Z

## Mission
Mine and verify all specifications regarding Kaggle CLI integration, Fair Benchmark Guardrails, Benchmark leaderboard & Model Card, Kaggle remote execution workflow, environment constraints, and verification commands in SentenAI-Unified.

## 🔒 My Identity
- Archetype: Specification Miner
- Roles: Specification Miner, Teamwork Specialist
- Working directory: D:\vietcv\SentenAI-Unified\.agents\miner_survey_kaggle
- Original parent: 95abf3c9-ecd1-4d03-9a74-a2869e75f3ee
- Milestone: Kaggle Tooling & Benchmark Specification Mining

## 🔒 Key Constraints
- Mine and verify all specifications thoroughly.
- Do NOT implement anything — read-only.
- Prioritize authoritative sources (files, configs, source code, docs).
- Output specification report to D:\vietcv\SentenAI-Unified\.agents\miner_survey_kaggle\handoff.md.

## Current Parent
- Conversation ID: 95abf3c9-ecd1-4d03-9a74-a2869e75f3ee
- Updated: 2026-08-24T00:28:44Z

## Task Summary
- **What to build**: Specification discovery report for Kaggle CLI tooling, Fair Benchmark guardrails, split manifest, fingerprint, benchmark script, leaderboard, model card, and remote workflows.
- **Success criteria**: Comprehensive tables (Features Discovered, Edge Cases), verbatim source quotes, exact parameters/flags, error behavior, and verification methods.
- **Interface contracts**: docs/kaggle_cli.md, tools/kaggle_cli/cli.py, docs/fair_benchmark.md, data/splits/split_manifest.json, ml/benchmark.py, experiments/benchmark/leaderboard.csv, MODEL_CARD.md
- **Code layout**: Workspace root at D:\vietcv\SentenAI-Unified

## Key Decisions Made
- Fully probed Kaggle CLI commands (`doctor`, `prepare-data`, `sync-data`, `prepare-kernel`, `run`, `status`, `logs`, `output`, `resume`, `collect`).
- Verified unit test suite passing (17 unit tests passed).
- Identified Transformer pooling bottleneck (`hidden[:, 0]`) and absent class domination in calibration as the key reasons for the performance gap vs `linear_svm`.

## Artifact Index
- D:\vietcv\SentenAI-Unified\.agents\miner_survey_kaggle\DISPATCH.md — Dispatch instructions
- D:\vietcv\SentenAI-Unified\.agents\miner_survey_kaggle\BRIEFING.md — Situational awareness
- D:\vietcv\SentenAI-Unified\.agents\miner_survey_kaggle\progress.md — Progress log
- D:\vietcv\SentenAI-Unified\.agents\miner_survey_kaggle\handoff.md — Final 5-component handoff report
