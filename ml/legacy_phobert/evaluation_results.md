# Kết quả đánh giá Mô hình PhoBERT ABSA (Tập Test)

Dưới đây là kết quả đánh giá chi tiết của mô hình PhoBERT ABSA (`hoangloc112/ABSA-TIKI-BOOK` / `data/models/ABSA-TIKI-BOOK`) trên tập dữ liệu kiểm thử `data/processed/test_clean.json` (gồm 1,993 reviews).

---

## 1. Đánh giá Cảm xúc Tổng quan (Overall Sentiment)

Độ chính xác tổng thể (Accuracy): **85%**
Macro Average F1-score: **0.82**

| Nhãn | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Tiêu cực (Negative)** | 0.93 | 0.85 | 0.89 | 987 |
| **Trung lập (Neutral)** | 0.58 | 0.76 | 0.65 | 345 |
| **Tích cực (Positive)** | 0.94 | 0.91 | 0.93 | 661 |

---

## 2. Đánh giá 6 Khía cạnh (6 Aspects - Chỉ tính khi khía cạnh xuất hiện)

Macro Average F1-score: **0.77**

| Nhãn | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Tiêu cực (Negative)** | 0.90 | 0.84 | 0.87 | 1,645 |
| **Trung lập (Neutral)** | 0.46 | 0.56 | 0.51 | 373 |
| **Tích cực (Positive)** | 0.94 | 0.91 | 0.92 | 1,376 |

---

## 3. F1-score Chi tiết của Từng Khía cạnh (Aspect-specific F1)

Các khía cạnh được đánh giá bằng chỉ số Macro F1-score trên các mẫu có xuất hiện khía cạnh đó (Present-only):

* 📝 **Nội dung sách (`as_content`):** `0.7910`
* 📖 **Hình thức vật lý (`as_physical`):** `0.7801`
* 💰 **Giá cả (`as_price`):** `0.6693`
* 📦 **Đóng gói (`as_packaging`):** `0.6777`
* 🚚 **Giao hàng (`as_delivery`):** `0.7250`
* 🎧 **Dịch vụ/Tư vấn (`as_service`):** `0.6633`

---

## 4. Các chỉ số Notebook-style

* **F1 Sentiment (Macro):** `0.8229`
* **F1 Aspect All:** `0.5502`
* **F1 Aspect Present:** `0.7673`
* **F1 Aspect Neutral Present:** `0.5054`
* **F1 Combined (Trung bình F1 Sentiment & F1 Aspect Present):** `0.7951`
* **Accuracy (Toàn bộ nhãn):** `0.6723`

---

## 5. Cấu hình Ngưỡng Hiệu chuẩn (Calibrated Thresholds)

Sử dụng cấu hình từ [thresholds.json](file:///d:/vietcv/absa-multi-agent-crm/packages/absa_core/absa_core/models/thresholds.json):

```json
{
  "thresholds": {
    "as_content": 0.1,
    "as_physical": 0.25,
    "as_price": 0.15,
    "as_packaging": 0.6,
    "as_delivery": 0.25,
    "as_service": 0.05
  }
}
```
