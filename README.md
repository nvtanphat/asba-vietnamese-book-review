# Vietnamese Book Review ABSA

Hệ thống phân tích cảm xúc đa khía cạnh cho đánh giá sách Tiki, kết hợp mô hình học sâu, tiền xử lý văn bản tiếng Việt và hai giao diện Streamlit:

- `app.py`: dashboard ABSA để nhập review và xem dự đoán.
- `dashboard.py`: dashboard kiểm tra dữ liệu và báo cáo chất lượng tập train / test.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

## Mục Tiêu

Dự án giải quyết hai bài toán chính:

1. Phân loại cảm xúc tổng thể của một review thành `tiêu cực`, `trung lập`, hoặc `tích cực`.
2. Phát hiện cảm xúc theo từng khía cạnh của trải nghiệm mua sách:
   - Nội dung
   - Hình thức
   - Giá cả
   - Đóng gói
   - Giao hàng
   - Dịch vụ

## Tính Năng

- Dự đoán sentiment tổng thể từ một review tiếng Việt.
- Nhận diện khía cạnh nào được nhắc đến trong review.
- Hiển thị xác suất, độ tin cậy và khía cạnh nổi bật.
- Tiền xử lý text trước khi đưa vào model.
- Dashboard kiểm tra dữ liệu với các báo cáo:
  - tổng quan dữ liệu,
  - giá trị thiếu,
  - độ dài văn bản,
  - mã hóa / Unicode,
  - nhiễu text,
  - emoji,
  - từ vựng bất thường,
  - trùng lặp,
  - phân bố nhãn.

## Kiến Trúc Mô Hình

### `app.py` - ABSA inference

`src/models/predictor.py` đang dùng model ABSA dạng multi-head:

- Head 1: sentiment tổng thể với 3 lớp.
- Head 2: nhận diện khía cạnh xuất hiện hay không.
- Head 3: sentiment cho từng khía cạnh.

Model được tải từ Hugging Face:

- Repo: `hoangloc112/ABSA-TIKI-BOOK`
- Subfolder sử dụng: `phobert`

Ngưỡng hiện tại để coi một khía cạnh là “được nhắc đến”:

- `presence_conf > 0.65`

Nếu một khía cạnh không đạt ngưỡng, hệ thống sẽ không gán sentiment cho khía cạnh đó.

### `dashboard.py` - Data report dashboard

Dashboard này đọc file JSON report đã sinh từ bước scan dữ liệu và hiển thị:

- thống kê tổng quan,
- missing values,
- length analysis,
- encoding issues,
- noise patterns,
- emoji statistics,
- vocab anomalies,
- duplicate analysis,
- label distribution,
- JSON gốc.

## Pipeline Tiền Xử Lý

Phần tiền xử lý nằm trong `src/preprocessing/` và được gọi qua `src/preprocessing/pipeline.py`.

Các bước chính:

- chuẩn hóa Unicode,
- làm sạch noise text,
- chuẩn hóa emoji,
- chuẩn hóa từ vựng / teencode,
- format lại text,
- lower-case nếu cần,
- lọc các dòng quá ngắn hoặc trùng lặp.

Hàm chính:

- `clean_text_series(...)`
- `preprocess_dataframe(...)`
- `preprocess_file(...)`

## Cấu Trúc Dự Án

```text
├── app.py                       # Dashboard ABSA cho review sách
├── dashboard.py                 # Dashboard kiểm tra dữ liệu / report
├── README.md                    # Tài liệu tổng quan dự án
├── requirements.txt             # Danh sách thư viện cần cài
├── .streamlit/
│   └── config.toml              # Theme và cấu hình giao diện Streamlit
├── data/
│   ├── raw/                     # Dữ liệu thô ban đầu
│   ├── interim/                 # Dữ liệu trung gian sau khi split
│   ├── processed/               # Dữ liệu đã làm sạch để train/eval
│   └── error_analysis_test.csv  # Mẫu dữ liệu phục vụ phân tích lỗi
├── docs/                        # Tài liệu mô tả dự án
├── experiments/                 # Report và kết quả thực nghiệm
│   └── reports/                 # JSON report / prediction report
├── notebooks/                   # Notebook thử nghiệm mô hình
├── scripts/
│   ├── check.py                 # Kiểm tra phân bố nhãn
│   ├── evaluation.py            # Đánh giá mô hình trên tập test
│   └── vit5_parse.txt           # Ghi chú / parse cho ViT5
├── src/
│   ├── analysis/
│   │   ├── data_scanner.py      # Core scanner tạo report chất lượng dữ liệu
│   │   ├── scan_cli.py          # CLI chạy scan và xuất JSON report
│   │   ├── overview_check.py    # Kiểm tra tổng quan
│   │   ├── missing_values_check.py
│   │   ├── length_check.py
│   │   ├── encoding_check.py
│   │   ├── noise_pattern_check.py
│   │   ├── emoji_check.py
│   │   ├── vocab_check.py
│   │   ├── duplicate_check.py
│   │   ├── label_distribution_check.py
│   │   ├── scan_constants.py
│   │   ├── scan_dataframe.py
│   │   └── helpers.py
│   ├── models/
│   │   ├── architectures.py     # Định nghĩa model ABSA
│   │   └── predictor.py         # Load model và suy luận
│   ├── preprocessing/
│   │   ├── pipeline.py          # Pipeline tiền xử lý chính
│   │   ├── cli.py               # CLI clean train/val/test
│   │   ├── split_dataset.py     # Tách dữ liệu train/val/test
│   │   ├── unicode_norm.py      # Chuẩn hóa Unicode
│   │   ├── emoji_norm.py        # Chuẩn hóa emoji
│   │   ├── vocab_norm.py        # Chuẩn hóa từ vựng / teencode
│   │   ├── noise_cleaner.py     # Làm sạch noise text
│   │   ├── quality_filter.py    # Lọc chất lượng dòng dữ liệu
│   │   ├── formatters.py        # Chuẩn hóa định dạng cuối
│   │   └── map_utils.py         # Hàm hỗ trợ map / transform
│   ├── features/                # Thư mục dự phòng cho feature engineering
│   ├── ui/
│   │   └── styles.py            # CSS cho dashboard
│   └── utils/                   # Hàm tiện ích dùng chung
├── web_crapping/
│   └── crawler.py               # Crawl review sách từ Tiki
├── crawl_data/                  # Dữ liệu crawl ra từ crawler
└── __pycache__/                 # File cache Python
```

Tóm lại:

- `app.py` là màn hình demo ABSA.
- `dashboard.py` là màn hình report / data quality.
- `src/` chứa toàn bộ mã nguồn chính.
- `data/` chứa dữ liệu đầu vào, trung gian và đã xử lý.
- `experiments/` chứa report kết quả và prediction output.
- `scripts/` là các lệnh chạy kiểm tra, đánh giá.

## Cách Chạy

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Crawl dữ liệu gốc từ Tiki

Script crawl lấy review sách từ Tiki và lưu mặc định vào `crawl_data/tiki_books_reviews_v2.csv`.

```bash
python web_crapping/crawler.py
```

### 3. Tách dữ liệu train / val / test

Script này:

- đọc dữ liệu đã gán nhãn trong `data/raw/`,
- tách thành `train`, `val`, `test`,
- lưu cả bản raw interim và bản đã clean.

```bash
python src/preprocessing/split_dataset.py
```

Kết quả mặc định:

- `data/interim/raw_train/train.json`
- `data/interim/raw_val/val.json`
- `data/interim/raw_test/test.json`
- `data/processed/train_clean.json`
- `data/processed/val_clean.json`
- `data/processed/test_clean.json`

### 4. Tiền xử lý dữ liệu

Nếu bạn muốn chạy riêng từng split qua CLI:

```bash
python -m src.preprocessing.cli --split train
python -m src.preprocessing.cli --split val
python -m src.preprocessing.cli --split test
```

Script này làm các bước:

- chuẩn hóa Unicode,
- làm sạch noise text,
- chuẩn hóa emoji,
- chuẩn hóa từ vựng,
- lower-case nếu bật,
- lọc dòng quá ngắn và trùng lặp.

### 5. Scan / phân tích chất lượng dữ liệu

Tạo report JSON để dùng cho `dashboard.py`:

```bash
python -m src.analysis.scan_cli --input data/interim/raw_train/train.json --output experiments/reports/train_scan.json
```

Nếu muốn scan một file khác, đổi `--input` và `--output` tương ứng.

### 6. Huấn luyện / thực nghiệm mô hình

Phần train model hiện được tổ chức trong notebook và các report thực nghiệm trong `notebooks/` và `experiments/`.

- Classic ML / baseline: dùng đặc trưng truyền thống.
- PhoBERT: mô hình ABSA đa đầu ra.
- ViT5: hướng generative ABSA.

### 7. Đánh giá mô hình

Chạy script đánh giá trên tập test đã clean:

```bash
python scripts/evaluation.py
```

Script này đang dùng mặc định:

- `data/processed/test_clean.json`

### 8. Kiểm tra phân bố nhãn

```bash
python scripts/check.py
```

### 9. Chạy dashboard ABSA

```bash
streamlit run app.py
```

### 10. Chạy dashboard kiểm tra dữ liệu

```bash
streamlit run dashboard.py
```

## Luồng Chạy Toàn Dự Án

1. Crawl review từ Tiki bằng `web_crapping/crawler.py`.
2. Chuẩn bị dữ liệu gốc trong `data/raw/`.
3. Tách train / val / test bằng `src/preprocessing/split_dataset.py`.
4. Tiền xử lý dữ liệu bằng `src/preprocessing/cli.py` hoặc pipeline trong `src/preprocessing/pipeline.py`.
5. Scan chất lượng dữ liệu bằng `src.analysis.scan_cli`.
6. Huấn luyện / thực nghiệm mô hình trong notebook hoặc phần experiment.
7. Đánh giá bằng `scripts/evaluation.py`.
8. Kiểm tra phân bố nhãn bằng `scripts/check.py`.
9. Mở `app.py` để demo ABSA.
10. Mở `dashboard.py` để xem dashboard phân tích dữ liệu.

## File Dữ Liệu / Report Quan Trọng

- `data/processed/train_clean.json`
- `data/processed/val_clean.json`
- `data/processed/test_clean.json`
- `experiments/reports/train_scan.json`
- `experiments/reports/train_clean_scan.json`
- `experiments/reports/fasttext_xgboost_v3_nosmote_predictions.json`
- `experiments/reports/fasttext_xgboost_v4_two_stage_neutral_predictions.json`

## Công Nghệ Sử Dụng

- `Python`
- `Streamlit`
- `PyTorch`
- `Transformers`
- `Pandas`
- `Plotly`
- `scikit-learn`
- `Hugging Face Hub`

## Giao diện Dashboard

Hệ thống cung cấp hai giao diện Dashboard chuyên biệt:

### 1. Dashboard Kiểm tra dữ liệu (Data Quality Dashboard)
Cho phép phân tích đặc điểm tập dữ liệu, phát hiện nhiễu và kiểm tra phân bố nhãn trước khi huấn luyện.
![Data Quality Dashboard](assets/images/dashboard_data.png)

### 2. Dashboard Phân tích ABSA (ABSA Inference Dashboard)
Giao diện người dùng cuối để nhập liệu và phân tích cảm xúc đa khía cạnh thời gian thực.
![ABSA Inference Dashboard](assets/images/dashboard_absa.png)

## Ghi Chú

- `app.py` là giao diện demo ABSA.
- `dashboard.py` là giao diện báo cáo dữ liệu.
- `.streamlit/config.toml` chứa cấu hình theme của Streamlit.
- Các model được load trực tiếp từ Hugging Face khi chạy predictor.

## Tác Giả

- Nguyễn Văn Tấn Phát
- Nguyễn Hoàng Lộc
