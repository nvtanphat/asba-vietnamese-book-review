# Original User Request

## 2026-08-24T00:20:24Z

Cải tiến hiệu năng các mô hình Aspect-Based Sentiment Analysis (ABSA) trong kho lưu trữ SentenAI-Unified và điều phối huấn luyện/đánh giá thông qua nền tảng Kaggle GPU tuân thủ hướng dẫn kỹ thuật tại `docs/kaggle_cli.md`, đồng thời đảm bảo 100% tính công bằng (Fair Benchmark Protocol).

Working directory: D:\vietcv\SentenAI-Unified
Integrity mode: benchmark

Reference documentation:
- Kaggle CLI Integration Guide: docs/kaggle_cli.md
- Fair Benchmark Protocol: docs/fair_benchmark.md
- Training Pipeline & Loss Architecture: docs/training_pipeline.md

## Requirements

### R1. Tối ưu kiến trúc Transformer & Pooling
- Nâng cấp cơ chế trích xuất vector đặc trưng của các mô hình Transformer (`phobert`, `mdeberta`, `xlmr`) từ First-Token Pooling (`hidden[:, 0]`) sang **Masked Mean Pooling** hoặc **Multi-Head Attention Pooling** để nắm bắt đầy đủ ngữ cảnh các khía cạnh ở giữa và cuối câu review.
- Bổ sung kết nối phân cấp (Hierarchical Head) giữa cảm xúc tổng thể (Overall Sentiment) và cảm xúc từng khía cạnh (Aspect Sentiments).

### R2. Tối ưu thuật toán Calibrate Thresholds cho Khía cạnh thiểu số
- Điều chỉnh hàm mục tiêu trong `ml/evaluation/calibration.py` (`calibrate_absent_thresholds`) để tối đa hóa **Present-Only Macro F1** của từng khía cạnh thay vì bị chi phối bởi độ chính xác của nhãn vắng mặt (absent label), giúp cải thiện F1 của các khía cạnh ít mẫu (`as_price`, `as_service`).

### R3. Điều phối huấn luyện từ xa qua Kaggle GPU Tooling
- Sử dụng chuẩn `python -m tools.kaggle_cli` theo `docs/kaggle_cli.md`:
  - Đồng bộ dataset chuẩn (`sync-data`) chứa `sentenai_src_bundle.dat` và dữ liệu `data/splits/`.
  - Nộp job huấn luyện (`run --model <model> --accelerator NvidiaTeslaT4`).
  - Giám sát tiến độ qua `status` / `logs --follow`.
  - Thu thập kết quả về thư mục `experiments/` thông qua `collect --register`.
  - Giữ chính sách validation-only (`--no-test`) trong quá trình thử nghiệm và chỉ dùng `--run-test` khi đánh giá chốt.

### R4. Tuân thủ nghiêm ngặt nguyên tắc Đánh giá Công bằng (Fair Benchmark Guardrails)
- Mọi mô hình chạy trên Kaggle phải mount chung 1 bộ dữ liệu cố định `data/splits/split_manifest.json` với cùng mã hash `data_fingerprint` (`c32f956aee64af890c0645d37da203a9ae1f62ca14d0ef85ac8f2182e3415623`).
- Giữ nguyên random seed (`seed = 42`) và chính sách niêm phong tập test.
- Tự động chạy lại `python -m ml.benchmark` để cập nhật bảng xếp hạng `experiments/benchmark/leaderboard.csv` và `MODEL_CARD.md`.

## Acceptance Criteria

### Performance & Fairness Criteria
- [ ] Tất cả mô hình được huấn luyện/đánh giá qua quy trình Kaggle tooling với cùng `data_fingerprint`.
- [ ] F1 Combined của các mô hình Transformer (`phobert`, `mdeberta`, `xlmr`) sau cải tiến tăng trưởng so với baseline hiện tại.
- [ ] F1 của các khía cạnh thiểu số (`as_price`, `as_service`) không bị suy giảm dưới 0.40 do ngưỡng quá cao.
- [ ] Kết quả từ Kaggle được đồng bộ tự động về `experiments/` và leaderboard `experiments/benchmark/leaderboard.csv` phản ánh chính xác kết quả mới.
