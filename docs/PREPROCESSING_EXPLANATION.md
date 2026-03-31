# Hướng dẫn Module Tiền xử lý Văn bản (Text Preprocessing)

Tài liệu này giải thích chi tiết các module trong thư mục `src/preprocessing`, được thiết kế để chuẩn hóa và làm sạch văn bản cho dự án Phân tích cảm xúc (Sentiment Analysis).

---

## 1. `map_utils.py` - Quản lý Bản đồ Cấu hình
Module này cung cấp các công cụ để tải dữ liệu từ các file JSON (như bản đồ emoji).

- **`_read_map`**: Đọc file JSON và chuyển thành dictionary. Sử dụng `@lru_cache` để ghi nhớ dữ liệu, tránh đọc đi đọc lại từ ổ đĩa (tăng tốc độ xử lý).
- **`load_json_map`**: Tải bản đồ từ file và cho phép gộp thêm các giá trị mặc định.

## 2. `emoji_norm.py` - Chuẩn hóa Emoji
Chuyển đổi các biểu tượng cảm xúc (emoji) sang tên gọi tiếng Việt tương ứng.

- **`_normalize_alias`**: Chuyển mã emoji tiếng Anh (ví dụ: `:grinning_face:`) về dạng chuẩn (chữ thường, không ký tự đặc biệt).
- **`demojize_text`**: Hàm chính. Nó chuyển emoji thực tế thành mã tiếng Anh, sau đó tra cứu trong `EMOJI_MAP` để lấy tên tiếng Việt. Ngoài ra, nó cũng tự động loại bỏ các biến thể màu da (skin tone).
- **`normalize_series`**: Áp dụng cho cột dữ liệu Pandas.

## 3. `formatters.py` - Chuẩn hóa Định dạng
Làm sạch các lỗi trình bày và ký tự rác.

- **`SPACE_RE`**: Thay thế các loại khoảng trắng thừa (tab, newline, nhiều dấu cách) bằng duy nhất 1 dấu cách.
- **`PUNCT_RE`**: Giới hạn các dấu câu lặp lại quá nhiều (ví dụ: `!!!` -> `!!`) để giảm nhiễu teencode.
- **`ZERO_WIDTH_RE`**: Loại bỏ các ký tự ẩn (không nhìn thấy bằng mắt thường) có thể gây lỗi khi máy xử lý.

## 4. `noise_cleaner.py` - Lọc Nhiễu Văn bản
Loại bỏ HTML và thay thế các thông tin nhạy cảm/không cần thiết bằng token chuyên dụng.

- **`strip_html`**: Sử dụng `BeautifulSoup` để trích xuất văn bản thuần túy, loại bỏ hoàn toàn các thẻ HTML.
- **`normalize_noise`**: Thay thế **URL**, **Email**, và **Số điện thoại** bằng các token tương ứng (`__url__`, `__email__`, `__phone__`). Việc này giúp mô hình hiểu cấu trúc văn bản mà không bị phân tâm bởi các thông tin định danh cụ thể.

---

## Cách sử dụng tổng quát

Bạn có thể kết hợp các module này để làm sạch dữ liệu trong DataFrame:

```python
import pandas as pd
from src.preprocessing import noise_cleaner, emoji_norm, formatters

def full_clean(text):
    text = noise_cleaner.normalize_noise(text)
    text = emoji_norm.demojize_text(text)
    text = formatters.normalize_format(text)
    return text

df['clean_text'] = df['raw_text'].apply(full_clean)
```
