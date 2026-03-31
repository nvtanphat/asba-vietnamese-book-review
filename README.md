# 📊 Vietnamese Book Review ABSA - Diamond Edition

Hệ thống Phân tích Cảm xúc Đa khía cạnh (**ABSA**) cho đánh giá sách trên Tiki. Dự án tập trung vào việc trích xuất Insights từ khách hàng thông qua các mô hình Deep Learning tiên tiến.

---

## 👥 Tác giả
- [**Nguyễn Văn Tấn Phát**](https://github.com/nvtanphat)
- [**Nguyễn Hoàng Lộc**](https://github.com/hoangloc112)

---

## 🤖 Hệ thống Mô hình (4 Models)
Dự án thực nghiệm qua 4 giai đoạn công nghệ:
1.  **Classic ML**: XGBoost / SVM trên tập đặc trưng TF-IDF.
2.  **Classic DL**: BiLSTM kết hợp Word2Vec cho xử lý chuỗi.
3.  **Transformer (Encoder)**: **PhoBERT** - Multi-head Classification (Mô hình đề xuất chính).
4.  **Transformer (Generative)**: **ViT5** - Generative ABSA sử dụng kỹ thuật LoRA Fine-tuning.

---

## 🛠️ Tiền xử lý & Dán nhãn
-   **Tiền xử lý**: Chuẩn hóa nội dung (lowercasing), xử lý Emoji/Teencode, loại bỏ ký tự đặc biệt và tối ưu hóa câu cho PhoBERT/ViT5.
-   **Dán nhãn (Labeling)**: Quy trình **LLM-Assisted Labeling** (Sử dụng GPT/Claude để gán nhãn sơ bộ) kết hợp **Manual Verification** để đảm bảo độ chính xác tuyệt đối cho bộ Dataset.

---

## 🏗️ Cấu trúc Diamond-Clean Layout
```text
├── app.py                      # Diamond Dashboard (Premium UI)
├── docs/                       # Reports, Labeling Guides & Data Explanations
├── scripts/                    # Scripts Evaluation (Metrics Report)
├── assets/images/              # Branding & Diagrams
├── data/                       # Datasets (Raw & Processed)
└── src/                        # Modular Source Code (Models, UI, Preprocessing)
```

---

## 🚀 Hướng dẫn Khởi chạy (Quick Start)

### 1. Cài đặt Thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy Demo Dashboard
Giao diện kính mờ cao cấp với biểu đồ Radar hiển thị kết quả phân tích PhoBERT:
```bash
streamlit run app.py
```

### 3. Chạy Báo cáo Đánh giá (Evaluation)
```bash
python scripts/evaluation.py
```

---
© 2024 Vietnamese Book Review ABSA Project | Đồ án thành viên team: Phát - Lộc
