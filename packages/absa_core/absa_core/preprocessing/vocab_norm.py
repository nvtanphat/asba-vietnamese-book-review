from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .map_utils import load_json_map

# Biểu thức chính quy để tìm các ký tự bị lặp lại từ 3 lần trở lên (ví dụ: 'ngonnnn')
ELONGATED_RE = re.compile(r"(.)\1{2,}")
# Tách văn bản thành các token: từ ngữ (\w+), ký hiệu ([^\w\s]+) hoặc khoảng trắng (\s+)
TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)
# Danh sách các nguyên âm tiếng Việt để xử lý kéo dài âm tiết đúng quy tắc
VIETNAMESE_VOWELS = set("aeiouyăâêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")
SAFE_SINGLE_CHAR_KEYS = {"k", "j", "r", "z", "đ"}


def _build_safe_vocab_map() -> dict[str, str]:
    """Load and filter vocabulary map to reduce noisy/ambiguous substitutions."""
    raw_map = load_json_map("vocab_map.json")
    filtered: dict[str, str] = {}

    for key, value in raw_map.items():
        src = str(key).strip().lower()
        dst = str(value).strip().lower()
        if not src or not dst:
            continue
        # Drop identity mappings (e.g. "app" -> "app"), they do not normalize anything.
        if src == dst:
            continue
        # Single-character substitutions are very risky in Vietnamese.
        # Keep only a small approved set that is common and semantically stable.
        if len(src) == 1 and src not in SAFE_SINGLE_CHAR_KEYS:
            continue
        filtered[src] = dst

    return filtered


# Tải bảng tra cứu từ vựng (viết tắt, tiếng lóng -> từ chuẩn)
VOCAB_MAP = _build_safe_vocab_map()


def _to_text(value: Any) -> str | None:
    """Chuyển đổi giá trị sang chuỗi, trả về None nếu là giá trị rỗng hoặc NaN."""
    if value is None or pd.isna(value):
        return None
    return str(value)


def _collapse_elongation(token: str) -> str:
    """Thu gọn các ký tự kéo dài (ví dụ: 'hayyyy' -> 'hayy')."""
    def replace(match: re.Match[str]) -> str:
        char = match.group(1)
        # Nếu ký tự lặp nằm ở cuối từ và là phụ âm, chỉ giữ lại 1 ký tự (ví dụ: 'ngonnn' -> 'ngon')
        if match.end() == len(token) and char.lower() not in VIETNAMESE_VOWELS:
            return char
        # Nếu là nguyên âm hoặc ở giữa từ, giữ lại 2 ký tự để duy trì sắc thái nhấn mạnh
        return char * 2

    return ELONGATED_RE.sub(replace, token)


def normalize_vocab(value: Any) -> str | None:
    """Chuẩn hóa từ vựng: Xử lý từ viết tắt và thu gọn các từ bị viết kéo dài."""
    text = _to_text(value)
    if text is None:
        return None

    parts: list[str] = []
    # Phân tách văn bản thành từng phần nhỏ (token)
    for chunk in TOKEN_RE.findall(text):
        if chunk.isspace(): # Giữ lại khoảng trắng đơn giản
            parts.append(" ")
            continue
        
        # Nếu token là một từ (chữ cái, chữ số, hoặc có dấu gạch dưới)
        if chunk.isalnum() or "_" in chunk:
            lower = chunk.lower()
            # Ưu tiên thay thế nếu từ đó nằm trong bản đồ từ vựng (vocab_map.json)
            if lower in VOCAB_MAP:
                parts.append(VOCAB_MAP[lower])
            else:
                # Nếu không, thực hiện thu gọn các ký tự lặp lại (nếu có)
                parts.append(_collapse_elongation(chunk))
            continue
        
        # Giữ nguyên các ký tự đặc biệt khác (chấm, phẩy, chấm hỏi,...)
        parts.append(chunk)

    # Kết hợp lại các phần và dọn dẹp khoảng trắng thừa
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def normalize_series(series: pd.Series) -> pd.Series:
    """Áp dụng chuẩn hóa từ vựng cho toàn bộ một cột dữ liệu trong Pandas Series."""
    return series.map(normalize_vocab)
