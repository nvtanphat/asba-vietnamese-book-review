# HƯỚNG DẪN DÁN NHÃN KHÍA CẠNH (ABSA)

**DỰ ÁN:** PHÂN LOẠI CẢM XÚC REVIEW SÁCH TIKI

## 1. DANH MỤC KHÍA CẠNH (7 ASPECTS)

*   `as_content`: Nội dung, cốt truyện, kiến thức, văn phong, chất lượng dịch thuật, lỗi chính tả, biên tập.
*   `as_physical`: Chất lượng giấy (vàng/trắng, dày/mỏng), bìa, gáy sách (bung keo), mực in, quà kèm của NXB (bookmark, postcard...).
*   `as_price`: Giá tiền, săn sale, đắt/rẻ, tính kinh tế/đáng tiền.
*   `as_packaging`: Hộp carton, bọc chống sốc (bubble wrap), tình trạng đóng gói khi nhận (móp méo do gói ẩu).
*   `as_delivery`: Tốc độ giao hàng, thái độ shipper, dịch vụ vận chuyển.
*   `as_service`: Tư vấn, đổi trả, thái độ của Shop/Tiki, quà tặng thêm riêng của shop (ngoài quà của NXB).
*   `as_general`: Đánh giá tổng quan chung chung ("Tuyệt", "Ổn", "Thất vọng", "Nên mua"), không chỉ rõ khía cạnh nào.

## 2. QUY ƯỚC NHÃN CẢM XÚC (POLARITY)

*   `0` (**Negative**): Phàn nàn, tiêu cực, lỗi sản phẩm/dịch vụ.
*   `1` (**Neutral / Conflict**):
    *   **Neutral**: Đánh giá bình thường ("Giao hàng đúng hạn", "Giấy hơi vàng nhưng đọc được").
    *   **Conflict**: Vừa khen vừa chê trong cùng 1 khía cạnh ("Sách hay nhưng dịch hơi sượng").
*   `2` (**Positive**): Hài lòng, khen ngợi, khuyến khích.
*   `null`: Review KHÔNG nhắc tới khía cạnh này.

## 3. QUY TẮC VÀNG KHI DÁN NHÃN

*   **Ưu tiên "Dịch thuật" vào Content:** Dịch tệ = Sản phẩm lỗi (`as_content` = 0).
*   **Ưu tiên "Bookmark/Quà NXB" vào Physical:** Thuộc về hình thức vật lý của bộ sách.
*   **Định nghĩa Đóng gói (Packaging):** Chỉ tập trung vào cách "bảo vệ" sách (Hộp, xốp). Móp góc do gói ẩu -> `as_packaging` = 0.
*   **Định nghĩa General:** Dùng cho các câu cảm thán chung hoặc tóm tắt cảm xúc cuối review.

## 4. VÍ DỤ CHUẨN (TOTAL MAPPING)

**Câu:** "Sách nội dung hay nhưng dịch hơi sượng (1), giấy ngà vàng đọc dịu mắt (2), giao hàng nhanh (2), mỗi tội đóng gói sơ sài làm móp góc (0)."

**Mapping:**
*   `as_content`: 1
*   `as_physical`: 2
*   `as_price`: null
*   `as_packaging`: 0
*   `as_delivery`: 2
*   `as_service`: null
*   `as_general`: null

## 5. MỤC TIÊU PHÂN LOẠI

Giúp các mô hình học máy học sâu học được các vector đặc trưng tách biệt: Trí tuệ (Content), Vật lý (Physical), Logistics (Packaging/Delivery) và Kinh tế (Price).
