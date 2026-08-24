# Worker M1 Fix Context
Milestone M1 Fix: Update packages/absa_core/absa_core/models/unified_predictor.py
Feedback from Challenger 2: D:\vietcv\SentenAI-Unified\.agents\challenger_m1_2\handoff.md
Task: In packages/absa_core/absa_core/models/unified_predictor.py:43, pass pooling_type=str(cfg.get("pooling_type", "masked_mean")) and head_type=str(cfg.get("head_type", "hierarchical")) to EncoderMultiTaskNetwork.
Run all tests: python -m pytest tests/ -v.
