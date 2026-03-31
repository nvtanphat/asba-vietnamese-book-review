# 💎 Vietnamese Book Review ABSA - Diamond Edition

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/hoangloc112/ABSA-TIKI-BOOK)

Hệ thống Phân tích Cảm xúc Đa khía cạnh (**Aspect-Based Sentiment Analysis - ABSA**) dành cho các bài đánh giá sách trên sàn thương mại điện tử Tiki. Đây là phiên bản **Diamond Edition** với giao diện Dashboard cao cấp và cấu trúc mã nguồn được quy chuẩn hóa.

---

## 🖼️ Dashboard Showcase
Hệ thống cung cấp một bảng điều khiển trực quan giúp người dùng theo dõi:
- **Cảm xúc tổng thể**: Phân loại Tích cực/Trung lập/Tiêu cực với độ tin cậy cao.
- **Phân tích Khía cạnh**: Radar Chart thể hiện mức độ xuất hiện của 6 khía cạnh cốt lõi.
- **Chi tiết hiệu năng**: Xác suất dự đoán cụ thể cho từng đầu ra của mô hình PhoBERT.

## 🤖 Công nghệ & Mô hình (Technical Stack)
Dự án sử dụng kiến trúc **PhoBERT (Multi-head Classification)** đã được tinh chỉnh để giải quyết đồng thời 3 nhiệm vụ:
1. **Sentiment**: Cảm xúc chủ đạo của cả câu.
2. **Presence**: Xác định khía cạnh nào được nhắc đến (Content, Physical, Price, Packaging, Delivery, Service).
3. **Aspect Sentiment**: Cảm xúc cụ thể cho từng khía cạnh được nhận diện.

> [!TIP]
> Mô hình được huấn luyện và lưu trữ tại Hugging Face: [hoangloc112/ABSA-TIKI-BOOK](https://huggingface.co/hoangloc112/ABSA-TIKI-BOOK)

## 🏗️ Cấu trúc dự án (Diamond-Clean Layout)
Dự án được tổ chức theo tiêu chuẩn công nghiệp nhằm tối ưu hóa việc bảo trì và mở rộng:
```text
├── app.py                      # Main Entry Point (Premium Dashboard)
├── requirements.txt            # Dependencies
├── docs/                       # Báo cáo thực nghiệm & Hướng dẫn chi tiết
├── scripts/                    # Scripts đánh giá hiệu năng (Metrics)
├── assets/images/              # Branding, Logos & Diagrams
├── data/                       # Dữ liệu Test & Training (Cleaned)
└── src/                        # Modular Source Code
    ├── models/                 # Logic Predictor & Architectures
    ├── ui/                     # Design System (Glassmorphism CSS)
    └── preprocessing/          # Text Cleaning Pipelines
```

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.9+ và tiến hành cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Diamond Dashboard
```bash
streamlit run app.py
```

### 3. Chạy báo cáo Metrics (Academic Reporting)
Để kiểm tra độ chính xác (F1-Score, Precision, Recall) trên tập dữ liệu Test:
```bash
python scripts/evaluation.py
```

---

## 🏆 Đóng góp (Credits)
Dự án được thực hiện trong khuôn khổ **Đồ án 2 - Data Preprocessing**.
- **Tác giả**: **Hoàng Lộc** 🎓
- **Chủ đề**: Phân tích dữ liệu TMĐT Tiki (ABSA).

---
© 2024 Vietnamese Book Review ABSA Project | Diamond Premium UI Framework v1.5
