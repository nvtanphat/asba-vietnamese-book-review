# Báo cáo dataset và tiền xử lý

Tài liệu này mô tả dataset Tiki book review đang dùng trong project, các vấn đề chất lượng dữ liệu đã phát hiện, và những bước xử lý mà pipeline hiện tại áp dụng.

## 1. Tóm tắt nhanh

- File raw chính hiện tại: `data/raw/tiki-book-review.json`
- Dataset raw có `13,412` dòng và `15` cột
- Cột văn bản chính: `content`
- Nhãn sentiment: `sentiment_llm`
- Mapping nhãn sentiment:
  - `0 = negative`
  - `1 = neutral`
  - `2 = positive`
- Project hiện dùng JSON làm nguồn chính, CSV chỉ còn là fallback kỹ thuật

## 2. Cấu trúc dataset

Các cột chính của file raw:

| Cột | Ý nghĩa |
| --- | --- |
| `review_id` | Mã định danh review |
| `product_id` | Mã sản phẩm |
| `product_name` | Tên sản phẩm / tên sách |
| `category` | Danh mục |
| `review_title` | Tiêu đề review |
| `content` | Nội dung review |
| `rating` | Điểm sao 1-5 |
| `created_at` | Thời gian tạo review |
| `sentiment_llm` | Nhãn sentiment 3 lớp |
| `as_content` | Nhãn aspect về nội dung |
| `as_physical` | Nhãn aspect về chất lượng vật lý / sách |
| `as_price` | Nhãn aspect về giá |
| `as_packaging` | Nhãn aspect về đóng gói |
| `as_delivery` | Nhãn aspect về giao hàng |
| `as_service` | Nhãn aspect về dịch vụ |

Ghi chú:

- Các cột `as_*` là nhãn aspect-sentiment kiểu 3 lớp.
- Nhiều dòng không có nhãn aspect ở tất cả các cột, đây là đặc điểm phổ biến của bài toán ABSA, không hẳn là lỗi.
- `product_id` và `created_at` là metadata, không được giữ nguyên trong bộ dữ liệu sạch cho huấn luyện.

## 3. Các vấn đề của dataset

### 3.1. Mất cân bằng lớp sentiment

Phân phối sentiment trong file raw:

| Nhãn | Ý nghĩa | Số lượng | Tỷ lệ |
| --- | --- | ---: | ---: |
| `0` | negative | `7,028` | `52.4%` |
| `2` | positive | `4,196` | `31.29%` |
| `1` | neutral | `2,187` | `16.31%` |

Tỷ lệ giữa lớp lớn nhất và nhỏ nhất xấp xỉ `3.21x`, nên bài toán sentiment có thiên lệch lớp khá rõ.

### 3.2. Nhãn aspect rất thưa

Trên raw train split, các cột aspect bị thiếu rất nhiều:

| Cột | Missing | Tỷ lệ |
| --- | ---: | ---: |
| `as_service` | `10,393` | `96.88%` |
| `as_price` | `9,747` | `90.86%` |
| `as_packaging` | `8,810` | `82.12%` |
| `as_delivery` | `8,030` | `74.85%` |
| `as_content` | `7,501` | `69.92%` |
| `as_physical` | `6,601` | `61.53%` |

Điều này phản ánh đúng tính chất ABSA: một review thường chỉ nhắc đến một vài khía cạnh, không phải tất cả.

### 3.3. Noise trong text

Scanner hiện tại phát hiện nhiều dạng noise trong văn bản:

- lặp dấu câu
- ký tự kéo dài
- URL
- chuỗi chỉ có ký hiệu
- chuỗi chỉ có chữ số

Số liệu trên file raw full:

| Loại noise | Số dòng |
| --- | ---: |
| `punct_repeat` | `827` |
| `elongated` | `148` |
| `digit_only` | `1` |
| `url` | `4` |
| `symbol_only` | `1` |

Trên raw train split:

| Loại noise | Số dòng |
| --- | ---: |
| `punct_repeat` | `660` |
| `elongated` | `115` |
| `url` | `2` |
| `symbol_only` | `1` |

### 3.4. Lỗi encoding và ký tự rác

Raw full dataset có:

- `276` dòng có vấn đề encoding, chiếm `2.06%`
- `20` dòng ngắn hơn ngưỡng tối thiểu, chiếm `0.15%`
- `18` text bị trùng sau khi normalize, chiếm `0.13%`
- `0` exact duplicate

Raw train split có:

- `212` dòng có vấn đề encoding, chiếm `1.98%`
- `14` dòng ngắn hơn ngưỡng tối thiểu
- `11` normalized duplicate texts
- `0` exact duplicate

### 3.5. Vấn đề vocab / teencode / viết tắt

Scanner còn flag nhiều token "bất thường" hoặc giống teencode / viết tắt / accentless tokens. Đây không phải luôn là lỗi, nhưng là một lý do để project cần bước chuẩn hoá vocab.

Trên raw train clean scan:

- `8,916` dòng có suspicious vocab, chiếm `83.36%`
- Các token thường gặp gồm các dạng viết tắt ngắn, token không dấu, token tắt kiểu chat, và một số misspelling

Ghi chú:

- Tỉ lệ này cao vì review tiếng Việt trên sàn thương mại điện tử thường có nhiều từ rút gọn, viết tắt, không dấu và biến thể chính tả.
- Mục tiêu của pipeline là chuẩn hoá bớt, không phải loại bỏ toàn bộ kiểu ngôn ngữ tự nhiên này.

### 3.6. Dữ liệu schema không hoàn toàn sạch

Raw full dataset còn có một số dấu hiệu lệch schema:

- `sentiment_llm` có `1` giá trị thiếu trong file raw
- `rating` có một giá trị bất thường bị lẫn thành câu văn thay vì số sao

Đây là lý do pipeline phải có bước lọc label hợp lệ và kiểm tra dữ liệu trước khi train.

## 4. Dataset theo từng artefact

| Artefact | Số dòng | Số cột | Mục đích |
| --- | ---: | ---: | --- |
| `data/raw/tiki-book-review.json` | `13,412` | `15` | Nguồn raw chính |
| `data/interim/raw_train/train.json` | `10,728` | `15` | Raw train để scan / dashboard |
| `data/interim/raw_test/test.json` | `2,683` | `15` | Raw test để đối chiếu |
| `data/interim/train/train.json` | `10,696` | `14` | Train đã clean cho modeling |
| `data/interim/test/test.json` | `2,675` | `14` | Test đã clean cho modeling |
| `data/processed/train_clean.json` | `10,696` | `14` | Bản processed train cho downstream use |
| `data/processed/test_clean.json` | `2,675` | `14` | Bản processed test cho downstream use |

Ghi chú:

- `data/interim/train/train.json` và `data/processed/train_clean.json` đang cùng nội dung, được ghi ra hai nơi để giữ tương thích với các notebook / script cũ.
- Tương tự, `data/interim/test/test.json` và `data/processed/test_clean.json` cũng cùng nội dung.

## 5. Project đã xử lý những gì

### 5.1. Quy trình tổng quát

Luồng hiện tại của project:

1. Đọc `data/raw/tiki-book-review.json` là nguồn chính
2. Lọc các dòng có `sentiment_llm` không hợp lệ
3. Chia 80/20 theo `sentiment_llm` với `random_state=42`
4. Lưu split raw vào:
   - `data/interim/raw_train/train.json`
   - `data/interim/raw_test/test.json`
5. Chạy pipeline clean trên toàn bộ data
6. Chia tiếp 80/20 trên dữ liệu đã clean
7. Lưu output clean vào:
   - `data/interim/train/train.json`
   - `data/interim/test/test.json`
   - `data/processed/train_clean.json`
   - `data/processed/test_clean.json`

### 5.2. Chuẩn hoá unicode và encoding

Module: `src/preprocessing/unicode_norm.py`

Xử lý:

- sửa lỗi encoding bằng `ftfy.fix_text` nếu có cài đặt
- normalize về NFC
- xoá control / format chars không cần thiết
- giữ lại các ký tự đặc biệt hợp lý như `\n`, `\r`, `\t`, `ZWJ`, `ZWNJ`

Mục tiêu:

- giảm mojibake
- tránh text bị hỏng bởi encoding khác nhau

### 5.3. Làm sạch noise cơ bản

Module: `src/preprocessing/noise_cleaner.py`

Xử lý:

- bỏ HTML bằng `BeautifulSoup`
- thay URL bằng `__url__`
- thay email bằng `__email__`
- thay số điện thoại bằng `__phone__`

Mục tiêu:

- giữ tín hiệu ngôn ngữ
- giảm nhiễu do link / contact info

### 5.4. Chuẩn hoá emoji

Module: `src/preprocessing/emoji_norm.py`

Xử lý:

- chuyển emoji sang dạng text bằng `demojize`
- map alias qua `emoji_map.json`

Mục tiêu:

- giữ lại nghĩa cảm xúc của emoji thay vì bỏ mất hoàn toàn

### 5.5. Chuẩn hoá vocab / teencode / kéo dài ký tự

Module: `src/preprocessing/vocab_norm.py`

Xử lý:

- load mapping từ `vocab_map.json`
- thay các viết tắt phổ biến bằng dạng chuẩn hơn
- co giãn ký tự kéo dài như `đẹppppp` về dạng gọn hơn
- giữ lại token và khoảng trắng hợp lý

Mục tiêu:

- giảm biến thể chính tả
- giúp tokenizer downstream nhìn thấy text ổn định hơn

### 5.6. Chuẩn hoá format

Module: `src/preprocessing/formatters.py`

Xử lý:

- rút gọn lặp dấu câu quá dài về 2 ký tự
- xoá zero-width chars
- gom nhiều khoảng trắng thành 1 khoảng trắng

Mục tiêu:

- giảm noise thị giác
- giúp text nhất quán hơn trước khi tokenize

### 5.7. Lọc chất lượng dữ liệu

Module: `src/preprocessing/quality_filter.py`

Xử lý:

- loại text quá ngắn
- loại dòng chỉ có số hoặc chỉ có ký hiệu
- loại `null`, `none`, `nan`, `#NAME?`
- loại duplicate sau khi normalize

Ngưỡng hiện tại:

- `SHORT_TEXT_MIN_CHARS = 10`

### 5.8. Giữ lại cột cần thiết

Trong quá trình clean, project chỉ giữ các cột phục vụ modeling:

- `review_id`
- `rating`
- `review_title`
- `product_name`
- `category`
- `content`
- `content_raw`
- `sentiment_llm`
- `as_content`
- `as_physical`
- `as_price`
- `as_packaging`
- `as_delivery`
- `as_service`

`content_raw` được thêm vào để lưu bản gốc trước khi clean.

## 6. Kết quả sau xử lý

### 6.1. Raw analysis split

File scan hiện đang dùng cho dashboard và notebook EDA:

- `experiments/reports/train_scan.json`

Nguồn scan:

- `data/interim/raw_train/train.json`

Thống kê chính:

- `10,728` dòng
- `15` cột
- `14` dòng ngắn hơn ngưỡng tối thiểu
- `212` dòng có encoding issue
- `768` dòng có noise
- `11` normalized duplicate texts

### 6.2. Clean train split

File scan của train đã clean:

- `experiments/reports/train_clean_scan.json`

Nguồn:

- `data/interim/train/train.json`
- `data/processed/train_clean.json`

Thống kê chính:

- `10,696` dòng
- `14` cột
- `0` dòng ngắn hơn ngưỡng tối thiểu
- `23` dòng còn dấu hiệu encoding
- `647` dòng còn `punct_repeat`
- `0` normalized duplicate texts

## 7. Đánh giá chi tiết xử lý hiện tại

### 7.1. Hiệu quả xử lý

So với raw train, pipeline hiện tại đã cải thiện rõ ở các lỗi bề mặt:

- encoding issue giảm từ `212` xuống `23` dòng
- normalized duplicate texts giảm từ `11` xuống `0`
- elongated noise giảm từ `115` xuống `2`
- short / rác text được loại bỏ gần hết

Điều này cho thấy pipeline đang làm tốt phần làm sạch text cơ bản, đủ để phục vụ train ABSA và EDA.

### 7.2. Điểm mạnh

- Tách riêng raw analysis split và clean train split, nên vừa nhìn được dữ liệu gốc vừa có dữ liệu sạch để train.
- Giữ `content_raw` giúp trace lại trước/sau xử lý.
- Pipeline được chia module rõ ràng:
  - unicode / encoding
  - noise
  - emoji
  - vocab / teencode
  - format
  - quality filter
- Các bước xử lý đều deterministic, dễ tái lập khi chạy lại với JSON mới.
- Không dựa vào nhãn để clean text, nên giảm nguy cơ “học lẫn” thông tin nhãn vào bước tiền xử lý.

### 7.3. Điểm còn hạn chế

- `punct_repeat` vẫn còn khá nhiều sau clean (`647` dòng ở clean train). Đây không hẳn là lỗi, vì pipeline hiện chỉ rút gọn dấu câu chứ không xoá hoàn toàn.
- `suspicious_vocab` vẫn rất cao ở clean train (`8,916` dòng, khoảng `83.36%`). Phần này phần lớn phản ánh đặc trưng ngôn ngữ review tiếng Việt:
  - viết tắt
  - chat slang
  - không dấu
  - token ngắn
  - biến thể chính tả

  Nói cách khác, đây vừa là tín hiệu cần chuẩn hoá, vừa là nhiễu “tự nhiên” của dữ liệu.
- Missing aspect labels vẫn là vấn đề lớn nhất của dataset. Cleaning text không giải quyết được chỗ này, vì bản thân dữ liệu gốc đã thưa nhãn.
- File raw vẫn còn một số dòng lệch schema, ví dụ giá trị bất thường trong `rating`. Pipeline hiện lọc các label hợp lệ, nhưng vẫn nên có kiểm tra schema chặt hơn ở bước ingest.
- Clean split hiện được tạo bằng cách clean toàn bộ dữ liệu rồi mới chia. Với các bước clean hiện tại là deterministic nên vẫn dùng được, nhưng nếu muốn benchmark chặt hơn thì nên cố định chiến lược split và validation schema rõ ràng ngay từ raw.

### 7.4. Đánh giá tổng quan

Mức xử lý hiện tại là:

- tốt cho nghiên cứu và baseline
- đủ sạch để train lại model
- chưa phải mức “chuẩn production” nếu chưa thêm validation schema, logging chất lượng, và kiểm tra nhất quán nhãn sâu hơn

Nói ngắn gọn:

- phần text cleaning: ổn
- phần quality control: khá
- phần schema / label consistency: còn cần siết thêm
- phần ABSA label sparsity: là vấn đề của dataset gốc, không phải chỉ do preprocessing

## 8. Lưu ý khi đọc dataset này

- Dataset này là bài toán ABSA, nên thiếu nhãn ở các cột `as_*` là bình thường.
- Nếu chỉ nhìn raw full data, nhiều missing value có thể gây cảm giác dataset rất bẩn; thực tế phần lớn missing nằm ở aspect labels.
- Nếu cần train lại model, nên dùng bộ clean trong `data/processed/` hoặc `data/interim/train/` thay vì raw split.
- Nếu cần phân tích chất lượng dữ liệu, dùng `data/interim/raw_train/train.json` để thấy rõ dữ liệu trước xử lý.

## 9. Kết luận

Dataset hiện tại có 3 vấn đề lớn:

1. Nhãn aspect rất thưa
2. Sentiment bị lệch lớp
3. Text có noise, encoding artifacts và một phần vocab không chuẩn

Pipeline của project đã xử lý các vấn đề này bằng:

- chuẩn hoá unicode / encoding
- lọc noise và text rác
- chuẩn hoá emoji
- chuẩn hoá vocab / viết tắt
- chuẩn hoá format
- lọc row chất lượng thấp
- split raw để phân tích và split clean để train

Nếu muốn đọc nhanh hơn, hãy mở:

- `data/raw/tiki-book-review.json`
- `data/interim/raw_train/train.json`
- `data/processed/train_clean.json`
- `experiments/reports/train_scan.json`
- `experiments/reports/train_clean_scan.json`
