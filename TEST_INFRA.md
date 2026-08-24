# E2E Test Infra: SentenAI-Unified

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on internal implementation details.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Interaction + Real-World Workload Testing.

## Feature Inventory & Test Mapping
| # | Feature | Requirement | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Real-World) |
|---|---|---|:---:|:---:|:---:|:---:|
| 1 | Transformer Feature Pooling (Masked Mean / MHA) | R1 | 5 | 5 | ✓ | ✓ |
| 2 | Hierarchical Multi-Task Head | R1 | 5 | 5 | ✓ | ✓ |
| 3 | Present-Only Macro F1 Calibration | R2 | 5 | 5 | ✓ | ✓ |
| 4 | Minority Aspect Sensitivity (as_price, as_service) | R2 | 5 | 5 | ✓ | ✓ |
| 5 | Kaggle CLI Sync, Run, Status, Collect | R3 | 5 | 5 | ✓ | ✓ |
| 6 | Fair Benchmark Data Fingerprint & Seed | R4 | 5 | 5 | ✓ | ✓ |
| 7 | Benchmark Leaderboard & Model Promotion | R4 | 5 | 5 | ✓ | ✓ |

## Test Architecture
- Unit and Component Tests: `pytest tests/unit/`
- Smoke Tests: `pytest tests/smoke/`
- End-to-End Test Suite: `tests/e2e/` (or dedicated verification scripts)
- Validation Checks:
  - Architecture tensors and shapes across different sequence lengths and padding masks.
  - Calibration optimization avoiding 0.90 ceiling and preserving minority aspect recall.
  - Kaggle CLI commands syntax, staging, packaging `sentenai_src_bundle.dat`, and execution manifests.
  - Data fingerprint computation hash stability and split invariants.
  - Production promotion and leaderboard formatting.

## Coverage Goals
- Tier 1: >= 5 test cases per feature (happy path isolation).
- Tier 2: >= 5 test cases per feature (boundary, extreme lengths, all-pad, 0 presence, single class).
- Tier 3: Pairwise combinations across feature interactions.
- Tier 4: Realistic end-to-end ABSA workflow scenarios.
