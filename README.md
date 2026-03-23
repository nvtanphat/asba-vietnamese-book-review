# Đồ án Phân tích Cảm xúc Đa khía cạnh (ABSA) - Tiki Book Reviews

## 1. Giới thiệu Project
Dự án tập trung vào việc tiền xử lý văn bản, trực quan hóa dữ liệu và xây dựng mô hình Deep Learning (**PhoBERT**) để phân loại cảm xúc từ các đánh giá sách thực tế trên Tiki. Mô hình không chỉ nhận diện cảm xúc (Tích cực/Tiêu cực/Trung lập) mà còn phân tích sâu vào 6 khía cạnh của một đơn hàng:
- Nội dung sách (Content), Khổ giấy/Bìa (Physical), Giá cả (Price), Đóng gói (Packaging), Giao hàng (Delivery), Chăm sóc khách hàng (Service).
- Tích hợp một quy trình **Pipeline Làm Sạch Dữ Liệu độc lập (Stateless)** mạnh mẽ và một **Web Dashboard (Streamlit)** để kiểm tra biến rỗng, nhiễu, lỗi đánh máy, từ lóng (Teencode).

<p align="center">
  <img src="pipeline.png" alt="Sơ đồ Luồng Tiền Xử Lý (Data Preprocessing Pipeline)" width="800">
</p>

## 2. Cấu trúc Thư mục Chi tiết
Hệ thống thư mục được tổ chức theo tiêu chuẩn của các dự án Data Science và Machine Learning:

```text
DoAn2/
├── data/                    # Khu vực toàn quyền của Dữ liệu (Không chứa code)
│   ├── raw/                 # 👉 Chứa dữ liệu thô nguyên thủy vừa cào từ Tiki về (.csv)
│   ├── interim/             # 👉 Dữ liệu trung gian (đã chia cắt Train/Test nhưng chưa lọc rác)
│   └── processed/           # 👉 Dữ liệu hoàn chỉnh (sạch 100%, sẵn sàng đưa vào Train)
│
├── src/                     # Source Code Lõi của hệ thống (Viết kiểu OOP chuẩn mực)
│   ├── analysis/            # 👉 Script rà soát lỗi rỗng, đếm số từ, tìm teencode (Scanner)
│   └── preprocessing/       # 👉 Nơi chứa chuỗi thuật toán Làm Sạch Dữ Liệu (Pipeline)
│       ├── unicode_norm.py  # 👉 Chuẩn hóa dấu thanh Tiếng Việt thống nhất
│       ├── emoji_norm.py    # 👉 Dịch biểu tượng cảm xúc (Emoji) ra chữ thuần
│       ├── noise_cleaner.py # 👉 Xóa bỏ rác (Link URL, Số điện thoại, Email, Ký tự HTML)
│       ├── vocab_norm.py    # 👉 Sửa lỗi đánh máy cơ bản, tự động phiên dịch Teencode
│       ├── formatters.py    # 👉 Chuyển chữ thường, cắt bỏ hàng ngàn khoảng trắng thừa
│       ├── quality_filter.py# 👉 Lọc bỏ review quá ngắn, xóa dòng trùng lặp (Drop Duplicates)
│       ├── map_utils.py     # 👉 Tiện ích hỗ trợ đọc thư viện ánh xạ quy tắc
│       ├── pipeline.py      # 👉 Đóng gói tất cả các script trên thành 1 ống nối liền mạch
│       ├── split_dataset.py # 👉 Kịch bản chia data 80% (Train) và 20% (Test) an toàn không Leak
│       └── cli.py           # 👉 File giao tiếp Terminal để user đứng ở ngoài gõ lệnh chay
│
├── notebooks/               # Bếp nấu thực nghiệm (Nghiên cứu/Train mô hình trên Kaggle/Colab)
│   ├── 01_before_after...   # 👉 File chứng minh tính hiệu quả của Pipeline làm sạch
│   ├── 02_eda_detailed...   # 👉 Khám phá dữ liệu, vẽ biểu đồ phân phối từ vựng
│   ├── 03_phobert_balance.. # 👉 Xử lý mất cân bằng nhãn bằng Focal Loss
│   └── 04_absa_roberta...   # 👉 Train mô hình cốt lõi PhoBERT ra checkpoint
│
├── experiments/             # Nơi chứa kết quả chạy thử nghiệm
│   └── reports/             # 👉 Giữ các file JSON quét lỗi (Scan) của Data Scanner 
│
├── web_crapping/            # Web Crawler (Scripts để cào bình luận từ hệ thống Tiki)
│
├── dashboard.py             # App giao diện Web (Streamlit) để vẽ biểu đồ đánh giá độ rác của dữ liệu
├── requirements.txt         # Khai báo các thư viện phụ thuộc của dự án
└── README.md                # Tài liệu hướng dẫn sử dụng
```

## 3. Thư viện cần cài đặt
Yêu cầu Python 3.8 trở lên. Vui lòng cài đặt toàn bộ thư viện thông qua file `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Thư viện lõi bao gồm: Pandas, Scikit-learn, Emoji, Streamlit, Plotly, PyTorch và Transformers).*

## 4. Cách chạy Code
Đứng ở thư mục rễ (root) của dự án `DoAn2/` và chạy lần lượt trong Terminal:

- **Bước 1: Làm sạch và chuẩn bị dữ liệu (Train/Test Split)**
  ```bash
  python -m src.preprocessing.split_dataset
  ```

- **Bước 2: Quét lỗi dữ liệu lập báo cáo thống kê (Scanner)**
  ```bash
  python -m src.analysis.scan_cli
  ```

- **Bước 3: Xem báo cáo phân tích bằng Biểu đồ Web**
  ```bash
  streamlit run dashboard.py
  ```

- **Bước 4: Huấn luyện AI**
  (Sau khi dữ liệu đã sạch, truy cập thư mục `notebooks/` và chạy tiếp các file `01_` -> `04_` trên Jupyter Notebook để tiến hành xây dựng mô hình PhoBERT).

## 5. Đội ngũ thực hiện
- **Nguyễn Văn Tấn Phát**
- **Nguyễn Hoàng Lộc**
