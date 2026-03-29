# 📊 Quy trình & Quy tắc Mô tả Biểu đồ (Project ABSA)

Dựa trên nghiên cứu từ AIO Conquer, mọi biểu đồ trong dự án cần được mô tả theo cấu trúc 4 phần để đảm bảo tính chuyên nghiệp và giá trị phân tích.

---

## 🏗️ 1. Cấu trúc Mô tả (The 4-Pillar Template)

Mỗi mô tả biểu đồ nên bao gồm:

1.  **Loại & Mục đích (Type & Purpose):**
    *   Tên biểu đồ (Bar, Boxplot, Heatmap...).
    *   Biểu đồ này giải quyết câu hỏi gì? (Ví dụ: "So sánh phân phối độ dài bình luận giữa các nhãn cảm xúc").
2.  **Chi tiết Kỹ thuật (Technical Detail):**
    *   Trục X, trục Y đại diện cho điều gì?
    *   Màu sắc (Hue) thể hiện biến số nào? Có sử dụng dải tin cậy (Confidence Interval - cho Line Chart) hay không?
3.  **Khám phá Trọng tâm (Key Insights):**
    *   *Quan trọng nhất:* Không chỉ liệt kê số liệu, hãy tìm ra những điểm "phi trực giác" (Ví dụ: "Dữ liệu không phân phối chuẩn mà bị lệch phải rõ rệt").
    *   Nhận diện Outliers (nếu có - dùng quy tắc IQR từ Boxplot).
    *   Nhận diện xu hướng (Tính tuyến tính, tương quan âm/dương).
4.  **Kết luận/Hành động (Actionable Insight):**
    *   Dữ liệu này gợi ý điều gì cho bước Tiền xử lý hoặc Huấn luyện mô hình? (Ví dụ: "Cần loại bỏ các bình luận quá dài > 500 từ vì chúng là outliers").

---

## 🎨 2. Quy tắc Thẩm mỹ & Nội dung (Guiding Principles)

*   **Nguyên tắc Tối giản (Data-Ink Ratio):** Hạn chế mô tả các yếu tố phụ (đường lưới, khung viền). Tập trung vào thông tin cốt lõi.
*   **Chiến lược Màu sắc:** 
    *   *Qualitative:* Dùng cho nhãn (Tích cực/Tiêu cực).
    *   *Sequential:* Dùng cho tần suất hoặc mức độ.
    *   *Diverging:* Dùng cho sự thay đổi (từ -1 đến 1).
*   **Kiểm chứng (Anscombe's Law):** Luôn dùng biểu đồ để kiểm chứng lại các chỉ số thống kê (Mean, Std). Hình ảnh mới là bằng chứng cuối cùng.

---

## 📑 3. Ví dụ mẫu (Vietnamese ABSA Context)

> **Hình 1: Boxplot phân phối độ dài từ (Word Count) theo Aspect**
>
> *   **Mục đích:** Kiểm tra xem các khía cạnh (Service, Price, Quality) có sự khác biệt về độ chi tiết trong bình luận hay không.
> *   **Kỹ thuật:** Trục X là các Aspects, Trục Y là số lượng từ. Màu sắc phân biệt theo Sentiment.
> *   **Insight:** Khía cạnh 'Price' thường có độ dài ngắn hơn đáng kể so với 'Service'. Phát hiện nhiều Outliers ở khía cạnh 'Quality' với các bình luận cực dài (>200 từ).
> *   **Hành động:** Thiết lập ngưỡng `max_length` là 150 để tối ưu hóa bộ nhớ cho mô hình BERT sau này.

---

## 🚀 4. Workflow 5 bước (Khi viết Code)
1. **Chuẩn bị:** Import Pandas/Seaborn & Load dữ liệu.
2. **Khởi tạo:** `fig, ax = plt.subplots(figsize=(...))`.
3. **Vẽ:** Chọn Chart phù hợp (Heatmap cho Correlation, Scatter cho Relationship).
4. **Styling:** Set Title, Label, Legend rõ ràng.
5. **Output:** `plt.tight_layout()` và Save/Show.
