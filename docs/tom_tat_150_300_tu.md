# Tóm tắt dự án ABSA (150-300 từ)

**Mục tiêu**  
Nghiên cứu xây dựng hệ thống phân tích cảm xúc đa khía cạnh cho review sách, gồm hai tác vụ: phân loại cảm xúc tổng thể 3 lớp và dự đoán cảm xúc theo 6 khía cạnh (`content`, `physical`, `price`, `packaging`, `delivery`, `service`). Cách tiếp cận này cho phép đánh giá đồng thời mức độ hài lòng chung và nguyên nhân cảm xúc theo từng thuộc tính dịch vụ.

**Dataset**  
Dữ liệu gốc có 13.411 mẫu, chia train/val/test lần lượt 9.392/2.009/2.010. Sau tiền xử lý, dữ liệu clean còn 13.374 mẫu (9.360/2.009/2.005). Mất cân bằng nhãn theo khía cạnh khá rõ: ở train, `service` chỉ có 302/9.360 mẫu hiện diện (3,23%), `price` là 860/9.360 (9,19%), trong khi `physical` đạt 3.662/9.360 (39,12%).

**Phương pháp tiền xử lý chính**  
Pipeline gồm chuẩn hóa Unicode, làm sạch nhiễu (URL, lỗi mã hóa, ký tự ẩn), chuẩn hóa emoji và từ vựng teencode, chuẩn hóa định dạng câu, rồi lọc mẫu quá ngắn và trùng lặp theo normalized text. Chất lượng dữ liệu cải thiện định lượng: lỗi encoding ở train giảm từ 198 xuống 0, số trùng lặp chuẩn hóa giảm từ 17 xuống 2, và toàn bộ mẫu dưới 10 ký tự được loại bỏ.

**Kết quả nổi bật**  
Baseline Logistic Regression trên test đạt `F1_combined = 0.7797` (`F1_sentiment = 0.7828`, `F1_aspect_all = 0.7046`, `F1_aspect_present = 0.7765`). Trong nhóm BiLSTM, cấu hình dùng biểu diễn PhoBERT đạt tốt nhất với `F1_Final = 0.7300` (`F1 Overall = 0.8465`, `F1 Aspects = 0.6136`). Mô hình PhoBERT cân bằng nhãn theo 2-stage đạt `eval_f1_combined = 0.7657` trước calibrate; sau calibrate threshold theo tiêu chí present-only, chỉ số tăng lên `0.8154`. ViT5 fine-tune bằng LoRA đạt `test_f1_final = 0.8098` (`test_f1_sentiment = 0.8405`, `test_f1_aspect_present = 0.7790`).
