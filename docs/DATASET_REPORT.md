# Báo cáo dataset và tiền xử lý cho train split

Báo cáo này được viết trực tiếp từ `experiments/reports/train_scan.json`, tức báo cáo quét chất lượng sinh ra từ `data\interim\raw_train\train.json`.

> Lưu ý:
> - Các số liệu dưới đây chỉ áp dụng cho **raw train split** 10,728 dòng, không phải toàn bộ dữ liệu raw.
> - Ở các phần `encoding` và `noise`, một dòng có thể bị gắn nhiều nhãn cùng lúc, nên tổng số đếm theo hạng mục có thể lớn hơn số dòng được flag.

## 1. Tóm tắt nhanh

| Chỉ số | Giá trị |
| --- | ---: |
| Nguồn scan | `data\interim\raw_train\train.json` |
| File báo cáo | `experiments/reports/train_scan.json` |
| Thời điểm tạo | `2026-03-23 02:27:28 UTC` |
| Số dòng | `10,728` |
| Số cột | `15` |
| Cột văn bản chính | `content` |
| Cột hoàn toàn đầy đủ | `8 / 15` |
| Dòng có ít nhất một missing | `10,723` (`99.95%`) |
| Dòng hoàn toàn đầy đủ | `5` (`0.05%`) |
| Dòng có issue encoding | `212` (`1.98%`) |
| Dòng có noise | `768` (`7.16%`) |
| Dòng có emoji | `338` (`3.15%`) |
| Dòng có suspicious vocab | `9,000` (`83.89%`) |
| Normalized duplicate texts | `11` (`0.10%`) |

## 2. Cấu trúc dữ liệu

### 2.1. Nhóm cột theo kiểu suy đoán

| Kiểu suy đoán | Số cột | Cột |
| --- | ---: | --- |
| `numeric_like` | `5` | `review_id`, `rating`, `product_id`, `created_at`, `sentiment_llm` |
| `text_like` | `4` | `review_title`, `content`, `product_name`, `category` |
| `mixed` | `6` | `as_content`, `as_physical`, `as_price`, `as_packaging`, `as_delivery`, `as_service` |

### 2.2. Vai trò từng cột

| Cột | Vai trò |
| --- | --- |
| `review_id` | Mã định danh của review |
| `rating` | Điểm sao 1-5 |
| `review_title` | Tiêu đề review |
| `content` | Nội dung review, là cột văn bản chính |
| `product_id` | Mã sản phẩm |
| `product_name` | Tên sách / tên sản phẩm |
| `category` | Danh mục sản phẩm |
| `created_at` | Thời điểm tạo review |
| `sentiment_llm` | Nhãn sentiment 3 lớp |
| `as_content` | Nhãn aspect về nội dung |
| `as_physical` | Nhãn aspect về chất lượng vật lý |
| `as_price` | Nhãn aspect về giá |
| `as_packaging` | Nhãn aspect về đóng gói |
| `as_delivery` | Nhãn aspect về giao hàng |
| `as_service` | Nhãn aspect về dịch vụ |

**Quy ước nhãn sentiment**:

- `0 = tiêu cực`
- `1 = trung lập`
- `2 = tích cực`

Với các cột `as_*`, dữ liệu thô lưu dưới dạng nhãn aspect có giá trị số khi có gán nhãn, còn phần chưa được gán để trống. Điều này là bình thường trong bài toán ABSA vì mỗi review chỉ nhắc đến một số khía cạnh nhất định.

## 3. Phân tích chất lượng dữ liệu

### 3.1. Missing values

Missing trong raw train split tập trung gần như hoàn toàn ở nhóm nhãn aspect. `content` chỉ thiếu 6 dòng, còn lại dữ liệu văn bản chính và metadata gần như đầy đủ.

| Cột | Missing | Tỷ lệ missing |
| --- | ---: | ---: |
| `as_service` | `10,393` | `96.88%` |
| `as_price` | `9,747` | `90.86%` |
| `as_packaging` | `8,810` | `82.12%` |ập 
| `as_delivery` | `8,030` | `74.85%` |
| `as_content` | `7,501` | `69.92%` |
| `as_physical` | `6,601` | `61.53%` |
| `content` | `6` | `0.06%` |

Nhận xét:

- Chỉ có `5` dòng hoàn toàn không thiếu giá trị nào.
- Có `7 / 15` cột xuất hiện missing, nhưng missing này chủ yếu là **missing có cấu trúc** của ABSA, không phải lỗi ngẫu nhiên.
- `content` gần như đầy đủ, nên là cột đáng tin cậy để làm đầu vào mô hình.

### 3.2. Độ dài văn bản

| Chỉ số | Giá trị |
| --- | ---: |
| Count | `10,728` |
| Min | `0` |
| P25 | `41` |
| Median | `74` |
| Mean | `123.19` |
| P75 | `145` |
| Max | `3,185` |
| Độ lệch chuẩn | `157.75` |

Phân bố theo bucket:

| Bucket độ dài | Số dòng | Tỷ lệ |
| --- | ---: | ---: |
| `<10` | `14` | `0.13%` |
| `10-19` | `438` | `4.08%` |
| `20-49` | `3,049` | `28.42%` |
| `50-99` | `3,091` | `28.80%` |
| `>=100` | `4,136` | `38.56%` |

Nhận xét:

- Phần lớn review nằm trong vùng `20-145` ký tự, tức là dữ liệu khá thực tế và không quá ngắn.
- Có một đuôi dài rất mạnh, với max `3,185` ký tự.
- Chỉ `14` dòng ngắn hơn 10 ký tự, nên việc lọc text quá ngắn chỉ ảnh hưởng rất nhỏ.

### 3.3. Encoding và Unicode

| Loại issue | Số dòng |
| --- | ---: |
| `fixable_encoding` | `182` |
| `mojibake_hint` | `28` |
| `zero_width` | `7` |

Nhận xét:

- `212` dòng có ít nhất một dấu hiệu encoding issue.
- Tỷ lệ này là `1.98%`, không quá lớn nhưng đủ để ảnh hưởng tokenizer nếu không chuẩn hóa.
- Các nhãn issue có thể chồng lấn, nên tổng số theo loại lớn hơn số dòng bị flag.
- `zero_width` cho thấy có ký tự ẩn cần xóa.
- `mojibake_hint` và `fixable_encoding` cho thấy dữ liệu vẫn còn dấu vết lỗi mã hóa cũ, nên cần bước chuẩn hóa Unicode trước khi train.

### 3.4. Noise trong text

| Loại noise | Số dòng |
| --- | ---: |
| `punct_repeat` | `660` |
| `elongated` | `115` |
| `url` | `2` |
| `symbol_only` | `1` |

Nhận xét:

- `768` dòng có ít nhất một mẫu noise, chiếm `7.16%`.
- `punct_repeat` là kiểu noise phổ biến nhất, thường là dấu câu lặp quá nhiều.
- `elongated` cho thấy vẫn có từ bị kéo dài để nhấn cảm xúc, kiểu như cách viết chat.
- Không phát hiện `email`, `phone`, `html`, `markdown_link` hay `digit_only` trong scan này.

### 3.5. Emoji

| Thống kê | Giá trị |
| --- | ---: |
| Dòng có emoji | `338` |
| Tỷ lệ dòng có emoji | `3.15%` |
| Tổng emoji | `615` |
| Emoji duy nhất | `109` |

Top emoji xuất hiện nhiều nhất:

| Emoji | Count |
| --- | ---: |
| `❤️` | `60` |
| `🥲` | `53` |
| `😭` | `38` |
| `👍` | `36` |
| `✋` | `33` |
| `🥰` | `30` |
| `❤` | `18` |
| `😍` | `17` |
| `⭐` | `14` |
| `😔` | `12` |

Nhận xét:

- Emoji không xuất hiện ở đa số review, nhưng vẫn đủ nhiều để tạo tín hiệu cảm xúc.
- Tập emoji thiên về cảm xúc mạnh, bao gồm cả tích cực lẫn tiêu cực.
- Trung bình mỗi dòng có emoji chứa khoảng `1.82` emoji.

### 3.6. Từ vựng và biến thể chữ viết

| Nhóm token | Số token duy nhất | Ví dụ điển hình |
| --- | ---: | --- |
| `teencode_like_tokens` | `178` | `k`, `dc`, `h`, `sp`, `mn`, `vs`, `mk` |
| `elongated_tokens` | `504` | `bookcare`, `shipper`, `bookmark`, `app`, `Good` |
| `accentless_ascii_tokens` | `1,665` | `giao`, `mua`, `dung`, `hay`, `cho` |
| `mixed_alnum_tokens` | `66` | `mp3`, `x2`, `A4` |
| `possible_misspellings` | `1,959` | heuristic over-flag, cần kiểm tra thủ công |

Thống kê bổ sung:

- Tổng token: `291,962`
- Token duy nhất: `8,218`
- Dòng đáng ngờ theo heuristic vocab: `9,000` (`83.89%`)

Nhận xét:

- Đây là mục có tỷ lệ cao nhất, nhưng không nên hiểu là dữ liệu “xấu”.
- Một phần lớn token bị flag là từ không dấu, viết tắt chat hoặc biến thể tự nhiên của tiếng Việt.
- Heuristic `possible_misspellings` khá rộng, nên nên dùng như tín hiệu chuẩn hóa chứ không phải danh sách lỗi chắc chắn.

### 3.7. Trùng lặp

| Chỉ số | Giá trị |
| --- | ---: |
| Exact duplicate rows | `0` |
| Normalized duplicate texts | `11` |
| Tỷ lệ normalized duplicates | `0.10%` |

Top văn bản trùng:

| Văn bản | Count |
| --- | ---: |
| `#NAME?` | `7` |
| `""` (chuỗi rỗng) | `6` |

Nhận xét:

- Không có dòng nào trùng nguyên văn toàn bộ bản ghi.
- Các bản trùng sau chuẩn hóa chủ yếu là placeholder hoặc giá trị rỗng, nên nên loại bỏ trước khi train.

### 3.8. Phân tích nhãn

#### `sentiment_llm`

| Cột | Missing | Non-missing | Phân bố giá trị | Imbalance |
| --- | ---: | ---: | --- | ---: |
| `sentiment_llm` | `0` | `10,728` | `0`: `5,622`; `2`: `3,357`; `1`: `1,749` | `3.2144` |

Nhận xét:

- Phân phối sentiment lệch về lớp tiêu cực.
- Đây là bài toán 3 lớp nhưng không cân bằng mạnh theo kiểu “bằng nhau”.
- Nếu train mô hình, nên cân nhắc class weight hoặc stratified split theo sentiment.

#### `rating`

| Cột | Missing | Non-missing | Phân bố giá trị | Imbalance |
| --- | ---: | ---: | --- | ---: |
| `rating` | `0` | `10,728` | `5`: `3,206`; `3`: `2,406`; `2`: `1,960`; `1`: `1,951`; `4`: `1,205` | `2.6606` |

Nhận xét:

- Rating 5 sao chiếm nhiều nhất.
- Rating 4 sao là lớp ít nhất trong scan này.
- Phân bố rating không đồng đều nhưng vẫn giữ được độ phủ cả 5 mức.

#### Các cột aspect

Quy ước giá trị:

- `0 = tiêu cực`
- `1 = trung lập`
- `2 = tích cực`
- `null = chưa gán nhãn / không được nhắc đến`

| Cột | Missing | Non-missing | Phân bố giá trị | Imbalance |
| --- | ---: | ---: | --- | ---: |
| `as_content` | `7,501` | `3,227` | `2`: `1,956`; `0`: `976`; `1`: `295` | `6.6305` |
| `as_physical` | `6,601` | `4,127` | `0`: `1,950`; `2`: `1,707`; `1`: `470` | `4.1489` |
| `as_price` | `9,747` | `981` | `1`: `598`; `2`: `262`; `0`: `121` | `4.9421` |
| `as_packaging` | `8,810` | `1,918` | `2`: `1,005`; `0`: `828`; `1`: `85` | `11.8235` |
| `as_delivery` | `8,030` | `2,698` | `2`: `1,720`; `1`: `506`; `0`: `472` | `3.6441` |
| `as_service` | `10,393` | `335` | `2`: `176`; `1`: `91`; `0`: `68` | `2.5882` |

Nhận xét:

- Các cột aspect rất thưa nhãn, đúng bản chất ABSA.
- `as_service` và `as_price` là hai cột thưa nhất.
- `as_packaging` có mức mất cân bằng cao nhất trong nhóm aspect có nhãn.
- Khi huấn luyện, nên xử lý `null` như nhãn “không có aspect / absent” thay vì cố impute giá trị số.

## 4. Khuyến nghị xử lý trước khi train

1. Loại bỏ hoặc kiểm tra kỹ các dòng `#NAME?` và chuỗi rỗng.
2. Chuẩn hóa Unicode, zero-width chars và mojibake trước khi tokenize.
3. Giữ bước xử lý repeated punctuation và elongated words, vì đây là noise phổ biến.
4. Với `as_*`, dùng cơ chế absent/masked label thay vì coi missing là dữ liệu lỗi.
5. Cân bằng `sentiment_llm` bằng class weight hoặc split có stratify.
6. Nếu dùng `content` làm input chính, nên giữ `review_title` và `product_name` như tín hiệu phụ vì chúng đã khá sạch.

## 5. Kết luận

Raw train split này có chất lượng khá tốt ở phần metadata và text chính, nhưng vẫn có ba vấn đề nổi bật:

- missing aspect labels rất thưa
- sentiment bị lệch lớp
- text còn tồn tại encoding artifacts, noise và biến thể từ vựng không chuẩn hóa

Điểm mạnh của dataset là:

- `content` gần như đầy đủ
- không có exact duplicate row
- dữ liệu đủ phong phú để phục vụ ABSA và sentiment analysis

Tổng thể, đây là dataset phù hợp để huấn luyện mô hình, miễn là pipeline tiền xử lý xử lý đúng các vấn đề missing aspect, chuẩn hóa Unicode và cân bằng nhãn.
