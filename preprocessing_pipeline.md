# Quy Trình Xử Lý Dữ Liệu Review

Tài liệu này mô tả pipeline tiền xử lý dữ liệu review của dự án, bao gồm
quản lý dữ liệu, làm sạch văn bản, trích xuất đặc trưng và lọc chất lượng.
Mục tiêu là chuẩn hóa dữ liệu đầu vào nhưng vẫn giữ lại các tín hiệu cảm xúc
quan trọng cho bài toán sentiment analysis.

## Cấu Trúc Dự Án

```text
DoAn2/
├── scrapers/                   # [GIAI ĐOẠN 0: DATA COLLECTION]
│   ├── tiki_crawler.py         # Crawl review Tiki
│
├── data/                       # [GIAI ĐOẠN 1: QUẢN LÝ DỮ LIỆU ĐA PHIÊN BẢN]
│   ├── raw/                    # Dữ liệu gốc thu thập từ scraping
│   ├── interim/                # Dữ liệu gộp nguồn, xử lý format sơ bộ
│   ├── processed/              # Dữ liệu sạch sau khi qua 7 khâu cleaning
│   └── balanced/               # [THỰC NGHIỆM - ĐẦU RA CỦA BALANCING]
│       ├── method_smote/       # Kết quả sau khi chạy SMOTE (oversampling)
│       ├── method_undersampling/ # Kết quả sau khi chạy random undersampling
│       └── method_oversampling/ # Kết quả sau khi chạy random oversampling
├── src/                        # [GIAI ĐOẠN 2: LOGIC NGHIÊN CỨU - CORE]
│   ├── preprocessing/          # Module 7 khâu làm sạch chuỗi cơ bản
│   │   ├── __init__.py         # Quản lý luồng pipeline chung
│   │   ├── unicode_norm.py     # Khâu 1: Chuẩn hóa NFC/Unicode
│   │   ├── noise_cleaner.py    # Khâu 2: Xử lý URL, email, markup
│   │   ├── emoji_norm.py       # Khâu 3: Map emoji -> text token
│   │   ├── vocab_norm.py       # Khâu 4: Teencode & ký tự lặp
│   │   ├── formatters.py       # Khâu 5: Định dạng cuối (spacing, trim)
│   │   └── quality_filter.py   # Khâu 7: Lọc dòng rác (len < 10, duplicate)
│   ├── balancing/              # [FOLDER MỚI - CHUYÊN THỰC NGHIỆM CÂN BẰNG]
│   │   ├── __init__.py         # Quản lý các phương pháp balancing
│   │   ├── oversamplers.py     # Logic SMOTE, ADASYN, random oversampler
│   │   ├── undersamplers.py    # Logic random undersampler, NearMiss
│   │   └── weights_calc.py     # Logic tính class weights cho loss
│   ├── analysis/               # Module check/scanner (data checker)
│   │   └── data_scanner.py     # Thống kê noise/label/phân bổ before-after
│   └── features/               # Khâu 6: Trích xuất đặc trưng bổ trợ
│       └── stats_extractor.py  # Đếm emoji, capitalized words, symbols...
├── notebooks/                  # [GIAI ĐOẠN 3: THỰC NGHIỆM & PHÂN TÍCH]
│   ├── 01_EDA_Before_After.ipynb    # Phân tích raw & so sánh với cleaned
│   ├── 02_Preprocessing_Run.ipynb   # Chạy 7 khâu & lưu vào data/processed/
│   ├── 03_Balancing_Expt_Study.ipynb
│   ├── 04_Model_PhoBERT.ipynb       # PhoBERT tokenization & training
│   ├── 05_Model_LSTM_RNNs.ipynb     # Word2Vec + deep learning
│   └── 06_..............
├── experiments/                # [GIAI ĐOẠN 4: KẾT QUẢ NGHIÊN CỨU]
│   ├── model_checkpoints/      # Lưu file model (.pth, .bin, .pkl)
│   ├── metrics_summary.csv     # Bảng tổng hợp KPI (F1, Accuracy) của model
│   └── reports/                # Biểu đồ confusion matrix, ROC-AUC, ...
├── requirements.txt
├── preprocessing_pipeline.md
└── README.md
```

## Mục Tiêu

- Giảm noise nhưng không làm mất tín hiệu sentiment.
- Giữ lại emoji, dấu câu cảm xúc, chữ hoa và các dạng nhấn mạnh hợp lệ.
- Chuẩn hóa các mẫu thông tin không mang nội dung như URL, email, số điện thoại.
- Tạo dữ liệu sạch, tái lập được và dễ so sánh giữa các cấu hình.
- Hỗ trợ luôn các thực nghiệm balancing và đánh giá before-after trên cùng
  một pipeline dữ liệu.

## Đặc Thù Dữ Liệu Review Tiếng Việt

Dữ liệu review thường có các đặc điểm cần xử lý khác văn bản chuẩn:

- Ngôn ngữ pha trộn giữa tiếng Việt, tiếng Anh và teencode.
- Dấu câu như `!!!`, `???`, `...` có thể mang tín hiệu cảm xúc.
- Emoji là tín hiệu quan trọng, không nên xóa máy móc.
- Có nhiều biến thể do gõ sai, kéo dài ký tự, viết hoa bất thường hoặc lặp từ.

## Nguyên Tắc Xử Lý

### Bắt buộc

- Chuẩn hóa Unicode về NFC để tránh cùng một ký tự nhưng khác codepoint.
- Giữ emoji và chuyển sang token text có nghĩa.
- Chuẩn hóa ký tự lặp để giảm nhiễu nhưng vẫn giữ mức nhấn mạnh.
- Lọc các dòng quá ngắn, đặc biệt là comment dưới 10 ký tự.

### Cần thận trọng

- Mở rộng teencode chỉ khi từ điển đủ tin cậy.
- Xóa khoảng trắng thừa và chuẩn hóa dấu câu.

### Không nên làm máy móc

- Không xóa stopwords một cách cứng nhắc, vì `không`, `chưa`, `vẫn` rất quan trọng.
- Không stemming/lemmatization nếu không có lý do rõ ràng cho tiếng Việt.
- Không lowercase toàn bộ nếu chữ hoa đang mang ý nghĩa nhấn mạnh.

## Kiểm Tra Trước Tiền Xử Lý

Trước khi làm sạch, nên scan dữ liệu để xác định noise và rủi ro:

- Số dòng, số cột và tên cột.
- Tỷ lệ giá trị rỗng, `null`, hoặc chuỗi chỉ có khoảng trắng.
- Phân bố độ dài văn bản.
- Lỗi encoding, mojibake, ký tự vô hình, broken unicode.
- Dòng chỉ toàn số hoặc chỉ toàn ký hiệu.
- Dòng trùng lặp nguyên văn và trùng sau khi chuẩn hóa.
- Emoji, dấu câu lặp, HTML/markup, URL, email, số điện thoại.
- Teencode, viết tắt, từ domain như `shop`, `ship`, `cod`, `size`, `màu`.

Mục đích của bước này:

- Chọn cấu hình xử lý phù hợp.
- Tránh viết rule quá mạnh ở những chỗ không cần.
- Phân biệt noise cần làm sạch với tín hiệu cần giữ lại.

## Tổng Quan Luồng Xử Lý

```text
raw review
-> chuẩn hóa ký tự và mã hóa
-> loại noise dạng markup/định danh
-> chuẩn hóa emoji thành token
-> chuẩn hóa từ vựng và độ dài ký tự
-> chuẩn hóa định dạng cuối cùng
-> content_clean
-> bổ sung đặc trưng
-> quality filter
-> processed dataset
```

## Đầu Vào

- Dữ liệu gốc là tập review dạng bản ghi.
- Cột văn bản chính được dùng làm nội dung cần xử lý.
- Nếu có nhiều cột văn bản, hệ thống chọn cột phù hợp theo cấu hình chạy.

## Đầu Ra

Sau khi xử lý, dữ liệu thường có các nhóm cột sau:

- `content_raw`: văn bản gốc để đối chiếu và debug.
- `content_clean`: văn bản sau khi làm sạch.
- `content_emoji_norm`: văn bản sau khi chuẩn hóa emoji.
- Các cột đặc trưng phụ: emoji, chữ hoa, ký tự lặp, thống kê liên quan.
- Dataset sau lọc chất lượng: đã loại các dòng quá ngắn, quá noise, chỉ số,
  hoặc trùng lặp.

## Các Khâu Xử Lý

### 1. Chuẩn Hóa Ký Tự Và Mã Hóa

Khâu này xử lý các vấn đề ở mức byte/Unicode:

- Chuẩn hóa Unicode về trạng thái ổn định.
- Sửa một số lỗi mã hóa thường gặp.
- Loại bỏ ký tự thừa, ký tự vô hình, control character.

Lợi ích:

- Giảm lỗi tokenizer.
- Tránh hiện tượng văn bản nhìn giống nhau nhưng khác codepoint.
- Giúp các bước sau hoạt động ổn định hơn.

### 2. Loại Noise Dạng Markup / Định Danh

Khâu này xử lý các phần không mang ý nghĩa nội dung:

- HTML hoặc markup.
- URL.
- Email.
- Số điện thoại.

Nguyên tắc:

- Markup được loại bỏ.
- Thông tin định danh thường được thay bằng placeholder.
- Placeholder giúp giữ cấu trúc câu và cho model biết đó là một loại noise.

Ví dụ:

```text
https://example.com  ->  __url__
a@b.com              ->  __email__
0901234567           ->  __phone__
```

### 3. Chuẩn Hóa Emoji

Emoji không bị xóa. Thay vào đó, emoji được chuyển thành token text có nghĩa.

Mục đích:

- Giữ lại tín hiệu cảm xúc.
- Giúp model downstream hiểu được ý nghĩa của emoji.
- Có thể đếm số lượng, tần suất và nhóm cảm xúc emoji.

Ví dụ:

```text
😡  ->  khong_hai_long
😍  ->  rat_thich
😭  ->  buon_that_vong
👍  ->  tot
```

Nếu emoji không nằm trong bảng ánh xạ, hệ thống vẫn cần cơ chế sinh token đại
diện để tránh mất thông tin hoàn toàn.

### 4. Chuẩn Hóa Từ Vựng Và Nhịp Điệu Của Văn Bản

Khâu này xử lý các hiệu ứng có ý nghĩa sentiment:

- Mở rộng viết tắt và teencode có độ tin cậy cao.
- Giảm ký tự lặp quá dài nhưng vẫn giữ cảm giác nhấn mạnh.

Ví dụ:

```text
ko    -> khong
dc    -> duoc
r     -> roi
quaaa -> quaa
!!!   -> !!
```

Nguyên tắc:

- Ưu tiên an toàn hơn là tham.
- Các từ domain hoặc từ mơ hồ sẽ được giữ lại nếu không chắc chắn.

### 5. Chuẩn Hóa Định Dạng Cuối Cùng

Đây là khâu đưa văn bản về format dễ đọc, dễ so sánh và ổn định hơn:

- Gom khoảng trắng thừa.
- Xóa khoảng trắng đầu và cuối dòng.
- Chuẩn hóa lại dấu câu.
- Viết hoa chữ cái đầu tiên sau dấu câu nếu cần format đầu ra dễ đọc.

Mục đích:

- Làm sạch đầu ra cuối.
- Giữ câu văn dễ đọc.
- Phù hợp cho debug và review bằng mắt người.

Ví dụ:

```text
xin chao. toi rat vui!!!  ->  Xin chao. Toi rat vui!!
```

## Bộ Đặc Trưng Phụ

Ngoài văn bản sạch, hệ thống còn sinh thêm một số đặc trưng để phục vụ nghiên cứu:

- Đếm emoji theo nhóm cảm xúc.
- Số lượng chữ hoa.
- Số từ viết hoa đầu.
- Số đoạn ký tự lặp.

Các đặc trưng này giúp so sánh:

- Có/không có emoji.
- Xử lý an toàn hay xử lý mạnh tay.
- Sự khác biệt giữa các cấu hình tiền xử lý.

## Lọc Chất Lượng

Sau khi làm sạch, dữ liệu được lọc bởi các quy tắc chất lượng:

- Dòng rỗng.
- Dòng có độ dài dưới 10 ký tự.
- Dòng chỉ toàn số.
- Dòng chỉ toàn ký hiệu.
- Dòng chỉ còn placeholder.
- Dòng trùng lặp.

Mục đích của bước này là:

- Tránh đưa noise vào model.
- Loại bỏ comment quá ngắn, không đủ thông tin.
- Loại bỏ các dòng không mang tín hiệu học hữu ích.
- Giữ quy tắc lọc đơn giản và dễ kiểm soát.

## Cấu Hình Xử Lý

Dự án có thể hỗ trợ nhiều cấu hình để so sánh nhanh:

### Cấu hình cơ bản

- Chuẩn hóa ký tự và mã hóa.
- Loại noise định danh.
- Chuẩn hóa emoji.
- Chuẩn hóa định dạng cuối cùng.

Đây là cấu hình cân bằng, phù hợp để chạy mặc định.

### Cấu hình tối giản

- Chuẩn hóa ký tự và mã hóa.
- Loại noise định danh.
- Chuẩn hóa emoji.

Phù hợp khi cần ablation, muốn xem tác động của việc bỏ bước format cuối.

### Cấu hình mạnh tay

- Chuẩn hóa ký tự và mã hóa.
- Loại noise định danh.
- Chuẩn hóa emoji.
- Mở rộng viết tắt.
- Co ký tự lặp.
- Chuẩn hóa định dạng cuối cùng.

Phù hợp khi muốn dữ liệu sạch nhất có thể để thử nghiệm hoặc huấn luyện
baseline.

## Cách Debug

Nếu muốn debug một mẫu cụ thể:

- Chạy từng khâu trên cùng một câu.
- So sánh trước/sau.
- Xác định khâu nào làm thay đổi mạnh nhất.
- Lưu lại mẫu lỗi để phân tích.

Nhóm debug tốt nhất thường là:

- Chuẩn hóa ký tự.
- Loại noise định danh.
- Emoji.
- Viết tắt và ký tự lặp.
- Định dạng cuối.

## Cách Thực Nghiệm

Để thử nhiều cách xử lý mà không phải sửa code liên tục:

- Đổi cấu hình.
- Bỏ một khâu.
- Thêm một khâu.
- Đổi thứ tự các khâu có liên quan.

Khuyến nghị:

1. So sánh cấu hình cơ bản với cấu hình tối giản.
2. So sánh cấu hình cơ bản với cấu hình mạnh tay.
3. Luôn lưu stat, dữ liệu trung gian, và cấu hình của mỗi run.

## Lưu Ý Khi Đánh Giá

- Không nên đánh giá chỉ bằng số dòng còn lại.
- Cần xem cả chất lượng văn bản sau làm sạch.
- Bổ sung quá mạnh có thể làm mất tín hiệu sentiment.
- Bổ sung quá yếu có thể để noise còn sót lại.
- Nên chia train/test trước khi chạy pipeline để tránh leakage.
- Tập test chỉ nên dùng cho benchmark cuối cùng, không dùng để tinh chỉnh quy
  tắc tiền xử lý.

## Tóm Tắt

Pipeline này được thiết kế theo hướng:

- Ít file hơn.
- Ít điểm cần sửa hơn.
- Dễ debug hơn.
- Dễ ablation hơn.
- Dễ chạy nhiều cấu hình hơn.

Mục tiêu cuối cùng là tạo ra dữ liệu sạch nhưng vẫn giữ được tín hiệu cảm xúc
quan trọng cho bài toán review sentiment.
