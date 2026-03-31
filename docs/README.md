# Vietnamese Book Review ABSA (Tiki Books)

Dự án Phân tích Cảm xúc Đa khía cạnh (Aspect-Based Sentiment Analysis - ABSA) cho các bài đánh giá sách trên sàn thương mại điện tử Tiki.

## 📋 Giới thiệu
Project này giải quyết bài toán ABSA cho 6 khía cạnh cốt lõi của trải nghiệm mua sách:
- **Content**: Nội dung sách, dịch thuật.
- **Physical**: Hình thức, chất lượng giấy/in.
- **Price**: Giá cả, kinh tế.
- **Packaging**: Đóng gói, xốp bong bóng.
- **Delivery**: Giao hàng, shipper.
- **Service**: Tư vấn, chăm sóc khách hàng.

## 🤖 Mô hình (Models)
Các mô hình đã được huấn luyện và tải lên Hugging Face:
- **PhoBERT**: Multi-head Classifier (Sentiment + Presence + Aspect Sentiment).
- **ViT5**: Generative ABSA (LoRA Fine-tuned).

🔗 Repo Model: [hoangloc112/ABSA-TIKI-BOOK](https://huggingface.co/hoangloc112/ABSA-TIKI-BOOK)

## 🏗️ Cấu trúc dự án (Project Structure)
```text
├── src/
│   ├── preprocessing/  # Pipeline tiền xử lý văn bản
│   ├── models/         # Toàn bộ logic nạp và dự đoán mô hình
│   └── evaluation.py   # Script báo cáo đánh giá (Academic Metrics)
├── notebooks/          # Quá trình thử nghiệm các mô hình (Classic ML, PhoBERT, ViT5)
├── experiments/        # Báo cáo và kết quả thực nghiệm
├── app.py              # Web Dashboard (Streamlit)
└── requirements.txt    # Thư viện phụ thuộc
```

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Chạy Dashboard Demo
Giao diện trực quan cho phép nhập review và xem kết quả ngay lập tức:
```bash
streamlit run app.py
```

### 3. Chạy báo cáo Evaluation
Để xuất báo cáo F1-Score trên tập Test:
```bash
python -m src.evaluation
```

## 📊 Thống kê hiệu năng (F1-Score)
| Model | Overall | Aspect (Best) |
| :--- | :---: | :---: |
| Classic ML (XGBoost) | ~0.76 | ~0.65 |
| **PhoBERT (Proposed)** | **~0.82** | **~0.76** |
| **ViT5 (LoRA)** | **~0.83** | **~0.78** |

---
© 2024 Vietnamese Book Review ABSA Project | Đồ án thành viên team: Hoàng Lộc
