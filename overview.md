# 1. Tổng quan và giới thiệu bài toán

## 1.1. Bối cảnh

Sự phát triển của thương mại điện tử làm gia tăng đáng kể khối lượng đánh giá do người dùng tạo ra sau khi mua hàng. Trong lĩnh vực sách, các nhận xét này không chỉ phản ánh mức độ hài lòng tổng thể của người đọc mà còn chứa nhiều thông tin chi tiết về các khía cạnh cụ thể như nội dung, hình thức, giá cả, đóng gói, giao hàng và dịch vụ. Việc khai thác tự động các phản hồi này có ý nghĩa quan trọng đối với cả người mua lẫn nhà bán hàng, vì nó giúp rút ra tín hiệu chất lượng sản phẩm một cách nhanh chóng và có hệ thống.

Tuy nhiên, dữ liệu đánh giá tiếng Việt trên các sàn thương mại điện tử thường có đặc trưng ngôn ngữ không chuẩn, ngắn gọn, nhiều lỗi chính tả, ký hiệu cảm xúc, teencode và nhiễu văn bản. Những đặc điểm này làm cho bài toán phân tích cảm xúc trở nên khó khăn hơn so với văn bản sạch và dài. Vì vậy, việc xây dựng một hệ thống phân tích cảm xúc đa khía cạnh cho review sách tiếng Việt là cần thiết cả về mặt học thuật lẫn ứng dụng thực tiễn.

## 1.2. Phát biểu bài toán

Bài toán trong nghiên cứu này được phát biểu như sau: với đầu vào là một bình luận hoặc đánh giá tiếng Việt về sách, hệ thống cần xác định:

1. Cảm xúc tổng thể của toàn bộ review, thuộc một trong ba nhãn `tiêu cực`, `trung lập` hoặc `tích cực`.
2. Các khía cạnh được đề cập trong review.
3. Cảm xúc tương ứng cho từng khía cạnh xuất hiện.

Nói cách khác, đây là một bài toán phân tích cảm xúc đa khía cạnh, trong đó mô hình không chỉ dự đoán sentiment ở mức toàn cục mà còn phải chỉ ra sentiment gắn với từng thuộc tính cụ thể của trải nghiệm người dùng. So với phân loại cảm xúc thông thường, bài toán này có độ phức tạp cao hơn vì một review có thể chứa nhiều khía cạnh với các thái độ khác nhau, ví dụ nội dung tích cực nhưng giao hàng tiêu cực.

## 1.3. Mục tiêu nghiên cứu

Mục tiêu của đề tài là xây dựng một quy trình xử lý và mô hình hóa hoàn chỉnh cho dữ liệu review sách tiếng Việt, bao gồm:

1. Thu thập và tổ chức dữ liệu đánh giá sách từ nguồn thực tế.
2. Tiền xử lý văn bản để giảm nhiễu, chuẩn hóa biểu diễn và tăng chất lượng dữ liệu đầu vào.
3. Xây dựng mô hình có khả năng dự đoán cảm xúc tổng thể và cảm xúc theo từng khía cạnh.
4. Cung cấp giao diện trực quan để kiểm tra dữ liệu đầu vào, đánh giá chất lượng tập dữ liệu và thử nghiệm dự đoán trên từng review.

Trên phương diện ứng dụng, hệ thống hướng tới việc hỗ trợ phân tích phản hồi khách hàng một cách tự động, từ đó phục vụ các hoạt động giám sát chất lượng sản phẩm, phát hiện vấn đề nổi bật và cải thiện trải nghiệm người dùng.

## 1.4. Đối tượng và phạm vi

Đối tượng của bài toán là các review tiếng Việt về sách trên sàn thương mại điện tử. Phạm vi bài toán tập trung vào các khía cạnh phổ biến trong trải nghiệm mua sách, bao gồm:

1. Nội dung
2. Hình thức
3. Giá cả
4. Đóng gói
5. Giao hàng
6. Dịch vụ

Hệ thống không nhằm giải quyết bài toán hiểu ngôn ngữ tổng quát, mà tập trung vào phân tích cảm xúc trong miền dữ liệu cụ thể là review sản phẩm sách. Do đặc thù dữ liệu thực tế, phần tiền xử lý đóng vai trò quan trọng trong toàn bộ quy trình, nhằm xử lý các hiện tượng như chuẩn hóa Unicode, làm sạch noise, chuẩn hóa emoji, và thống nhất các biến thể từ vựng không chuẩn.

## 1.5. Ý nghĩa của bài toán

Việc giải quyết bài toán này có ba ý nghĩa chính. Thứ nhất, nó cung cấp một công cụ tự động để tổng hợp ý kiến người dùng ở quy mô lớn. Thứ hai, nó cho phép xác định rõ khía cạnh nào đang gây tích cực hoặc tiêu cực trong trải nghiệm mua hàng, thay vì chỉ nhìn vào cảm xúc tổng thể. Thứ ba, đây là một bài toán có tính thực nghiệm cao đối với xử lý ngôn ngữ tự nhiên tiếng Việt, đặc biệt trong bối cảnh dữ liệu nhiễu và nhiều biến thể biểu đạt phi chuẩn.

Từ những yêu cầu trên, có thể thấy phân tích cảm xúc đa khía cạnh là một hướng tiếp cận phù hợp để khai thác sâu hơn giá trị thông tin ẩn trong các review sách tiếng Việt.

## 2. Dataset và quy trình xây dựng dữ liệu

## 2.1. Nguồn dữ liệu

Bộ dữ liệu được xây dựng từ các đánh giá sản phẩm sách thu thập trên Tiki, tập trung vào các sản phẩm thuộc nhóm sách tiếng Việt và sách tiếng Anh. Trong mã nguồn, quá trình crawl được triển khai tại `web_crapping/crawler.py`, sử dụng API của nền tảng để thu thập review theo từng sản phẩm, từng trang và từng mức đánh giá.

Mỗi bản ghi thô ban đầu bao gồm các trường chính:

1. `review_id`: mã định danh của review.
2. `rating`: số sao do người dùng đánh giá.
3. `review_title`: tiêu đề nhận xét.
4. `content`: nội dung review.
5. `product_id`: mã sản phẩm.
6. `product_name`: tên sản phẩm.
7. `category`: nhóm danh mục sách.
8. `created_at`: thời điểm tạo review.

Về mặt chiến lược thu thập, script crawl được thiết kế theo hướng ưu tiên bao phủ đủ các mức đánh giá, đồng thời tìm kiếm thêm các mẫu khó hơn như review tiêu cực hoặc review ngắn nhưng có giá trị ngữ nghĩa. Cách tiếp cận này giúp bộ dữ liệu không bị lệch hoàn toàn về các phản hồi tích cực vốn thường chiếm ưu thế trên sàn thương mại điện tử.

Sau khi thu thập và hợp nhất, file dữ liệu chính được lưu tại `data/raw/tiki-book-review.json`. Từ file này, quy trình tiền xử lý và tách tập sẽ tạo ra các bản sao trung gian và bản đã làm sạch để phục vụ huấn luyện, đánh giá và phân tích.

## 2.2. Quy mô và đặc trưng tổng quát của dữ liệu

Theo thống kê trên file dữ liệu gốc, bộ dữ liệu gồm 13.412 bản ghi, tương ứng 13.412 `review_id` duy nhất và 2.010 sản phẩm khác nhau. Mỗi dòng dữ liệu đại diện cho một review, tức là đơn vị quan sát trong bài toán là bình luận của người dùng chứ không phải sản phẩm.

Một số đặc trưng mô tả đáng chú ý:

1. Độ dài nội dung review biến thiên lớn, từ các phản hồi rất ngắn đến các nhận xét dài hơn một vài nghìn ký tự.
2. Nội dung review tiếng Việt thực tế có nhiều hiện tượng không chuẩn như viết tắt, lỗi chính tả, thiếu dấu, lẫn ký hiệu cảm xúc và các cụm từ khẩu ngữ.
3. Dữ liệu có cấu trúc nhãn đa nhiệm, trong đó vừa có nhãn cảm xúc tổng thể, vừa có nhãn cảm xúc theo từng khía cạnh.

Thống kê độ dài nội dung cho thấy:

1. Độ dài trung bình của `content` xấp xỉ 123.66 ký tự.
2. Trung vị khoảng 74 ký tự.
3. 25% mẫu có độ dài dưới 41 ký tự.
4. 75% mẫu có độ dài dưới 145 ký tự.
5. Có 20 mẫu ngắn hơn 10 ký tự và 542 mẫu ngắn hơn 20 ký tự.

Các con số này cho thấy dữ liệu thực tế có phân bố ngắn-dài khá lệch, nên bước làm sạch và lọc chất lượng đóng vai trò quan trọng trong toàn bộ pipeline.

## 2.3. Cấu trúc nhãn

Bộ dữ liệu sử dụng hai tầng nhãn chính:

1. Nhãn cảm xúc tổng thể, lưu trong cột `sentiment_llm`.
2. Nhãn cảm xúc theo khía cạnh, gồm 6 khía cạnh:
   - `as_content`
   - `as_physical`
   - `as_price`
   - `as_packaging`
   - `as_delivery`
   - `as_service`

Theo tài liệu hướng dẫn gán nhãn ABSA của dự án, bộ khía cạnh gốc được mô tả ở mức nghiệp vụ còn bao gồm `as_general` cho nhận xét tổng quan chung chung. Tuy nhiên, trong schema dữ liệu hiện tại, cảm xúc tổng thể của review được biểu diễn bằng `sentiment_llm`, còn 6 cột khía cạnh bên trên được dùng để lưu trạng thái cảm xúc cho các khía cạnh xuất hiện trong nội dung.

Theo quy ước của bộ dữ liệu:

1. `0` tương ứng với `tiêu cực`.
2. `1` tương ứng với `trung lập` hoặc `xung đột` trong cùng một khía cạnh.
3. `2` tương ứng với `tích cực`.
4. Giá trị `null` ở một khía cạnh có nghĩa là khía cạnh đó không được nhắc đến trong review.

Tài liệu gán nhãn cũng nêu rõ các quy tắc ưu tiên trong một số trường hợp đặc biệt:

1. Nếu review phê phán phần dịch thuật, nội dung nên được gán vào `as_content`.
2. Nếu review nói về bookmark, quà tặng từ nhà xuất bản hoặc đặc điểm vật lý của bộ sách, nội dung nên gán vào `as_physical`.
3. `as_packaging` chỉ dùng cho cách đóng gói và bảo vệ sản phẩm trong quá trình giao hàng, chẳng hạn hộp carton, xốp, bong bóng chống sốc, hoặc tình trạng móp méo do đóng gói kém.
4. `as_general` được dùng cho các nhận xét chung chung như “tuyệt”, “ổn”, “thất vọng”, “nên mua”, khi câu đánh giá không chỉ rõ khía cạnh cụ thể nào.

Một ví dụ mapping điển hình từ tài liệu hướng dẫn là câu: “Sách nội dung hay nhưng dịch hơi sượng, giấy ngà vàng đọc dịu mắt, giao hàng nhanh, mỗi tội đóng gói sơ sài làm móp góc.” Khi đó:

1. `as_content = 1`
2. `as_physical = 2`
3. `as_price = null`
4. `as_packaging = 0`
5. `as_delivery = 2`
6. `as_service = null`
7. `as_general = null`

Phân bố nhãn tổng thể cho 13.411 mẫu hợp lệ là:

1. `tiêu cực`: 7.015 mẫu.
2. `trung lập`: 2.170 mẫu.
3. `tích cực`: 4.226 mẫu.

Điều này cho thấy bộ dữ liệu có xu hướng nghiêng về lớp tiêu cực và tích cực nhiều hơn lớp trung lập. Vì vậy, khi xây dựng mô hình cần quan tâm tới khả năng mất cân bằng lớp và ảnh hưởng của nó lên đánh giá thực nghiệm.

Với nhãn khía cạnh, dữ liệu thể hiện tính thưa cao, vì không phải review nào cũng đề cập đến mọi khía cạnh. Số mẫu có nhãn khác `null` của từng khía cạnh lần lượt là:

1. `as_content`: 4.241 mẫu.
2. `as_physical`: 5.259 mẫu.
3. `as_price`: 1.227 mẫu.
4. `as_packaging`: 2.404 mẫu.
5. `as_delivery`: 3.402 mẫu.
6. `as_service`: 427 mẫu.

Từ đó có thể thấy, `nội dung` và `hình thức` là hai khía cạnh được nhắc đến nhiều nhất, trong khi `dịch vụ` xuất hiện ít hơn đáng kể. Đặc điểm này phản ánh đúng hành vi phản hồi thực tế của người dùng: họ thường tập trung vào những yếu tố trực tiếp như chất lượng cuốn sách, hình thức in ấn, đóng gói và giao hàng.

## 2.4. Đặc trưng dữ liệu thô và vấn đề chất lượng

Dữ liệu thu thập từ môi trường thực tế thường chứa nhiều nhiễu, và bộ dữ liệu này không phải ngoại lệ. Các vấn đề thường gặp gồm:

1. Review quá ngắn, không đủ ngữ nghĩa.
2. Dòng chỉ chứa số, ký hiệu hoặc biểu tượng.
3. Dấu câu lặp, kéo dài ký tự, emoji và ký tự đặc biệt.
4. Lỗi mã hóa, lỗi Unicode hoặc văn bản bị biến dạng do quá trình crawl và lưu trữ.
5. Trùng lặp nội dung giữa các review.

Do đó, dữ liệu thô không được đưa trực tiếp vào mô hình mà phải đi qua một chuỗi làm sạch và chuẩn hóa. Trong quá trình kiểm tra, bộ dữ liệu còn ghi nhận một số bản ghi bất thường ở metadata, vì vậy bước chuẩn hóa nhãn và lọc bản ghi hợp lệ là bắt buộc trước khi chia tập.

## 2.5. Quy trình làm dữ liệu

Quy trình xử lý dữ liệu được thiết kế theo hướng tuần tự, từ dữ liệu thô đến dữ liệu huấn luyện cuối cùng. Có thể tóm tắt như sau:

### Bước 1. Thu thập dữ liệu

Review được crawl từ Tiki và lưu thành file gốc. Mỗi bản ghi ban đầu giữ lại nội dung review và các thông tin liên quan đến sản phẩm, thời gian và điểm đánh giá.

### Bước 2. Lọc bản ghi hợp lệ

Trong bước tách dữ liệu, chỉ các bản ghi có `sentiment_llm` hợp lệ trong tập `{0, 1, 2}` mới được giữ lại. Điều này đảm bảo nhãn đầu ra nhất quán cho bài toán phân loại cảm xúc 3 lớp.

### Bước 3. Chuẩn hóa văn bản

Phần tiền xử lý trong `src/preprocessing/pipeline.py` thực hiện các bước:

1. Chuẩn hóa Unicode.
2. Làm sạch noise văn bản.
3. Chuẩn hóa emoji.
4. Chuẩn hóa từ vựng và biến thể viết không chuẩn.
5. Chuẩn hóa định dạng cuối.
6. Chuyển về chữ thường nếu cần.

Đây là bước quan trọng nhất để giảm phương sai biểu diễn đầu vào, đặc biệt với tiếng Việt trên dữ liệu mạng xã hội hoặc dữ liệu thương mại điện tử.

### Bước 4. Lọc chất lượng

Trong `src/preprocessing/quality_filter.py`, dữ liệu được loại bỏ nếu:

1. Nội dung rỗng hoặc gần như rỗng.
2. Chỉ chứa giá trị vô nghĩa như `null`, `none`, `nan`.
3. Chỉ chứa chữ số hoặc chỉ chứa ký hiệu.
4. Có độ dài ngắn hơn ngưỡng tối thiểu `SHORT_TEXT_MIN_CHARS = 10`.
5. Bị trùng sau khi chuẩn hóa.

Nhờ vậy, dữ liệu giữ lại có chất lượng tốt hơn và ít gây nhiễu cho mô hình.

### Bước 5. Chia tập train/validation/test

Dữ liệu được chia theo tỉ lệ 70/15/15 cho `train`, `validation` và `test`. Điểm đáng chú ý là việc chia tập không thực hiện trên từng dòng độc lập mà trên các nhóm nội dung đã chuẩn hóa, nhằm giảm rò rỉ dữ liệu do trùng hoặc gần trùng giữa các tập.

Trong `src/preprocessing/split_dataset.py`, quy trình chia tập được thực hiện như sau:

1. Chuẩn hóa nội dung để tạo khóa nhóm.
2. Gom các review có nội dung tương đương vào cùng một nhóm.
3. Dùng nhãn chiếm đa số trong nhóm để hỗ trợ stratified split khi có thể.
4. Tách nhóm thành train, validation và test.
5. Ghi ra cả bản raw trung gian và bản đã clean.

Các file đầu ra chính gồm:

1. `data/interim/raw_train/train.json`
2. `data/interim/raw_val/val.json`
3. `data/interim/raw_test/test.json`
4. `data/processed/train_clean.json`
5. `data/processed/val_clean.json`
6. `data/processed/test_clean.json`

### Bước 6. Giữ lại các cột cần thiết

Sau khi làm sạch, bộ dữ liệu huấn luyện chỉ giữ những cột phục vụ cho mô hình và phân tích:

1. `review_id`
2. `content`
3. `sentiment_llm`
4. `as_content`
5. `as_physical`
6. `as_price`
7. `as_packaging`
8. `as_delivery`
9. `as_service`

Thiết kế này giúp giảm nhiễu từ metadata không cần thiết và làm cho dữ liệu đầu vào nhất quán hơn cho toàn bộ pipeline.

### Sơ đồ tiền xử lý

```mermaid
flowchart TD
    A[Dữ liệu thô<br/>data/raw/tiki-book-review.json] --> B[Lọc nhãn hợp lệ<br/>sentiment_llm ∈ {0,1,2}]
    B --> C[Tạo khóa nhóm từ nội dung đã chuẩn hóa<br/>phục vụ chống trùng]
    C --> D[Chia theo nhóm nội dung<br/>giảm trùng và leakage]
    D --> E[Split Train / Val / Test<br/>70 / 15 / 15]
    E --> F[Chuẩn hóa Unicode]
    F --> G[Làm sạch noise]
    G --> H[Chuẩn hóa emoji]
    H --> I[Chuẩn hóa từ vựng / teencode]
    I --> J[Chuẩn hóa định dạng]
    J --> K[Lowercase]
    K --> L[Lọc chất lượng<br/>rỗng, ngắn, số, ký hiệu, trùng]
    L --> M[Dữ liệu processed<br/>train_clean / val_clean / test_clean]
```

## 2.6. Ý nghĩa của quy trình dữ liệu

Quy trình xây dựng dữ liệu không chỉ đơn thuần là làm sạch văn bản, mà còn nhằm đảm bảo tính tin cậy của thực nghiệm. Việc chuẩn hóa Unicode, loại bỏ review vô nghĩa, xử lý trùng lặp và chia tập theo nhóm nội dung giúp:

1. Giảm rò rỉ dữ liệu giữa các tập.
2. Hạn chế tác động của văn bản nhiễu lên mô hình.
3. Tăng độ ổn định của kết quả đánh giá.
4. Giữ được tính phản ánh thực tế của dữ liệu review tiếng Việt.

Từ góc nhìn thực nghiệm, đây là nền tảng quan trọng để xây dựng mô hình phân tích cảm xúc đa khía cạnh có khả năng tổng quát hóa tốt hơn trên dữ liệu đời thực.

## 2.7. Thống kê tóm tắt

Bảng dưới đây tóm tắt các chỉ số chính của bộ dữ liệu và các tập sau khi xử lý:

| Hạng mục | Số lượng |
| --- | ---: |
| Tổng số review thô | 13.412 |
| Số `review_id` duy nhất | 13.412 |
| Số sản phẩm khác nhau | 2.010 |
| Train thô | 9.392 |
| Validation thô | 2.009 |
| Test thô | 2.010 |
| Train đã xử lý | 9.360 |
| Validation đã xử lý | 2.009 |
| Test đã xử lý | 2.005 |

Phân bố nhãn cảm xúc tổng thể trên dữ liệu gốc:

| Nhãn | Số mẫu |
| --- | ---: |
| `0` - Tiêu cực | 7.015 |
| `1` - Trung lập | 2.170 |
| `2` - Tích cực | 4.226 |

Mức độ xuất hiện của từng khía cạnh trong dữ liệu:

| Khía cạnh | Số mẫu có nhãn |
| --- | ---: |
| `as_content` | 4.241 |
| `as_physical` | 5.259 |
| `as_price` | 1.227 |
| `as_packaging` | 2.404 |
| `as_delivery` | 3.402 |
| `as_service` | 427 |

Nhìn chung, `as_physical` và `as_content` là hai khía cạnh xuất hiện nhiều nhất, trong khi `as_service` có mật độ thấp hơn đáng kể. Điều này phù hợp với bản chất review sách, nơi người dùng thường ưu tiên nhận xét về nội dung và hình thức sản phẩm hơn là dịch vụ đi kèm.

## 2.8. Ví dụ minh họa

Để làm rõ cách biểu diễn dữ liệu, có thể xét các ví dụ điển hình sau:

### Ví dụ 1. Review chỉ mang tính tổng quan

`Cực kì hài lòng`

- `sentiment_llm = 2`
- Các nhãn khía cạnh đều `null`

Trường hợp này phù hợp với quy tắc `as_general` trong tài liệu gán nhãn, nhưng trong schema hiện tại vẫn được lưu qua nhãn sentiment tổng thể và không gán cụ thể vào một khía cạnh nào.

### Ví dụ 2. Review có một khía cạnh rõ ràng

`Sách về tay đẹp, không tì vết.`

- `sentiment_llm = 2`
- `as_physical = 2`
- Các khía cạnh còn lại `null`

Đây là ví dụ cho thấy review tập trung vào chất lượng vật lý của cuốn sách, cụ thể là tình trạng sản phẩm khi nhận hàng.

### Ví dụ 3. Review có nhiều khía cạnh và có xung đột

`Nội dung sách hay, nhưng trình bày bố cục chưa hợp lí lắm nên hơi khó hiểu chút, chất lượng sách hơi tệ vì mặc dù mình đọc rất cẩn thận nhưng vẫn bị bung keo sứt một số trang.`

- `sentiment_llm = 1`
- `as_content = 1`
- `as_physical = 0`

Ví dụ này minh họa rõ tính chất đa khía cạnh của bài toán: cùng một review có thể đồng thời chứa nhận xét tích cực về nội dung và tiêu cực về hình thức. Vì vậy, mô hình cần học được cả cảm xúc tổng thể lẫn cảm xúc theo từng khía cạnh.

### Ví dụ 4. Review có dấu hiệu nhiễu ngôn ngữ

`Mua cỡ 200 hoi :v chất lượng bìa okey mà spine uk hơi cứng.`

- Sau tiền xử lý, văn bản được chuẩn hóa về dạng nhất quán hơn.
- Nhãn khía cạnh chủ yếu rơi vào `as_physical`.

Ví dụ này cho thấy dữ liệu thực tế chứa nhiều biến thể ngôn ngữ không chuẩn, nên bước chuẩn hóa văn bản là cần thiết để giảm sai khác biểu diễn giữa các mẫu.

## 3. Thực nghiệm và kết quả

## 3.1. Mục tiêu thực nghiệm

Phần thực nghiệm trong notebook `notebooks/03_baseline_logistic_sentiment_aspect.ipynb` được thiết kế như một baseline cổ điển cho bài toán ABSA. Mục tiêu không phải là đạt kết quả tối đa bằng mọi giá, mà là xây dựng một mốc tham chiếu rõ ràng để so sánh với các mô hình sâu hơn ở các notebook sau.

Baseline này trả lời ba câu hỏi chính:

1. TF-IDF kết hợp Logistic Regression có đủ mạnh để giải quyết sentiment đa khía cạnh trên review sách tiếng Việt hay không.
2. Các biến thể giảm chiều như `chi2`, `SVD`, hay `L1 selection` có giúp cải thiện chất lượng hoặc tốc độ không.
3. Cơ chế tách hai bài toán con - phát hiện khía cạnh xuất hiện và dự đoán sentiment của khía cạnh đó - có hoạt động ổn định trên dữ liệu thực tế hay không.

## 3.2. Thiết lập mô hình

### Biểu diễn đặc trưng

Notebook sử dụng tổ hợp đặc trưng cổ điển gồm:

1. `word TF-IDF` với `ngram_range=(1, 2)`.
2. `char TF-IDF` với `analyzer='char_wb'` và `ngram_range=(3, 5)`.
3. `meta features` thủ công gồm:
   - số ký tự,
   - số từ,
   - số dấu chấm than,
   - số dấu hỏi,
   - tỷ lệ chữ hoa.

Việc kết hợp word-level, char-level và metadata giúp mô hình bắt được cả tín hiệu ngữ nghĩa lẫn các mẫu hình bề mặt thường gặp trong review tiếng Việt, đặc biệt là cách viết ngắn, lặp ký tự hoặc nhấn mạnh cảm xúc.

### Mô hình phân loại

Mô hình trung tâm là `Logistic Regression` dùng solver `saga`, với:

1. `C = 8.0`
2. `min_df = 4`
3. `max_df = 0.95`
4. `char_max_df = 0.95`
5. `class_weight = balanced`

Ngoài bản gốc, notebook còn thử các biến thể giảm chiều:

1. `none`
2. `chi2_20k`
3. `svd_512`
4. `chi2_20k_svd_512`
5. `l1_select_median`

Mục đích của các biến thể này là kiểm tra trade-off giữa chất lượng và chi phí tính toán khi làm việc với không gian đặc trưng lớn từ TF-IDF.

### Thiết lập ABSA

Notebook không chỉ dự đoán sentiment tổng thể mà còn triển khai một pipeline hai tầng cho từng khía cạnh:

1. Mô hình `presence_model` để dự đoán khía cạnh có được nhắc đến hay không.
2. Mô hình `sent_model` để dự đoán polarity của khía cạnh khi khía cạnh đó xuất hiện.

Nếu mô hình presence trả về xác suất thấp hơn ngưỡng tối ưu trên validation, hệ thống gán nhãn `ABSENT_LABEL`; ngược lại, nó dùng sentiment dự đoán cho khía cạnh đó.

Cách thiết kế này phù hợp với bản chất dữ liệu ABSA thưa, vì nhiều review không nhắc tới tất cả khía cạnh và việc ép mô hình luôn dự đoán sentiment cho mọi khía cạnh sẽ làm giảm độ chính xác.

## 3.3. Thiết kế đánh giá

Notebook đánh giá theo ba mức:

1. Sentiment tổng thể.
2. Aspect sentiment trên các khía cạnh có nhắc đến.
3. Aspect sentiment trên toàn bộ nhãn, bao gồm cả trạng thái không xuất hiện.

Các chỉ số chính gồm:

1. `macro F1` cho sentiment tổng thể.
2. `macro F1` cho khía cạnh xuất hiện.
3. `macro F1` cho toàn bộ aspect labels.
4. `F1` riêng cho từng aspect.
5. `F1` riêng cho lớp trung lập trong từng aspect.
6. `accuracy` bổ sung cho các nhìn nhìn tổng quát.

Đặc biệt, notebook dùng một chỉ số tổng hợp là:

`f1_combined = 0.5 * f1_sentiment + 0.5 * f1_aspect_present`

Chỉ số này cân bằng giữa hai nhiệm vụ quan trọng nhất: dự đoán sentiment toàn cục và dự đoán sentiment cho các khía cạnh có mặt trong review.

## 3.4. Kết quả validation

Sau khi chạy 5 cấu hình, kết quả validation được sắp xếp theo `val_f1_combined`. Bảng sau tóm tắt kết quả chính:

| Xếp hạng | Cấu hình | `val_f1_combined` | `val_f1_sentiment` | `val_f1_aspect_present` | `val_f1_aspect_neutral_present` | `fit_time_sec` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `best_lr_saga_c8_min4__none` | 0.7644 | 0.7693 | 0.7595 | 0.5924 | 874.15 |
| 2 | `best_lr_saga_c8_min4__l1_select_median` | 0.7644 | 0.7693 | 0.7595 | 0.5924 | 2317.61 |
| 3 | `best_lr_saga_c8_min4__chi2_20k` | 0.7623 | 0.7666 | 0.7581 | 0.5967 | 613.30 |
| 4 | `best_lr_saga_c8_min4__chi2_20k_svd_512` | 0.7514 | 0.7592 | 0.7437 | 0.5848 | 825.90 |
| 5 | `best_lr_saga_c8_min4__svd_512` | 0.7512 | 0.7560 | 0.7465 | 0.5869 | 980.32 |

Nhận xét từ validation:

1. Cấu hình tốt nhất là `none`, tức không giảm chiều bổ sung.
2. `l1_select_median` cho điểm tương đương nhưng thời gian huấn luyện dài hơn đáng kể.
3. `chi2_20k` khá gần cấu hình tốt nhất và là anchor để so sánh các phiên bản sau.
4. Các cấu hình có `SVD` cho chất lượng thấp hơn, đặc biệt ở `f1_aspect_all`.

Điều này cho thấy với baseline tuyến tính trên review tiếng Việt, không gian đặc trưng TF-IDF gốc vẫn mang nhiều tín hiệu hữu ích; giảm chiều không phải lúc nào cũng đem lại cải thiện.

## 3.5. Kết quả trên tập test

Sau khi chọn cấu hình tốt nhất theo validation, notebook đánh giá đúng một lần trên tập test. Kết quả chính:

| Chỉ số | Giá trị |
| --- | ---: |
| `f1_sentiment` | 0.7828 |
| `f1_aspect_present` | 0.7765 |
| `f1_aspect_all` | 0.7046 |
| `f1_aspect_neutral_present` | 0.6368 |
| `f1_combined` | 0.7797 |
| `acc_sentiment` | 0.8344 |
| `acc_aspect_present` | 0.7997 |
| `acc_aspect_all` | 0.8682 |

Kết quả chi tiết cho sentiment tổng thể trên test:

| Lớp | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.91 | 0.89 | 0.90 |
| Neutral | 0.57 | 0.62 | 0.59 |
| Positive | 0.86 | 0.85 | 0.85 |

Quan sát này cho thấy mô hình nhận diện tốt hai lớp biên `negative` và `positive`, trong khi lớp `neutral` vẫn là lớp khó nhất, phù hợp với đặc tính mơ hồ của sentiment trung lập trong dữ liệu thực tế.

Kết quả `present only` cho 6 khía cạnh trên test:

| Lớp | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.82 | 0.79 | 0.80 |
| Neutral | 0.67 | 0.61 | 0.64 |
| Positive | 0.91 | 0.86 | 0.89 |

Trong các khía cạnh, `as_delivery` có `F1` cao nhất ở mức `0.7860`, tiếp theo là `as_content` và `as_physical`. `as_service` và `as_packaging` thấp hơn, phản ánh độ thưa dữ liệu và tính đa dạng của cách diễn đạt.

## 3.6. Phân tích theo từng khía cạnh

Chỉ số `F1` trên test cho từng khía cạnh:

| Khía cạnh | F1 |
| --- | ---: |
| `as_delivery` | 0.7860 |
| `as_content` | 0.7240 |
| `as_physical` | 0.7179 |
| `as_price` | 0.6427 |
| `as_packaging` | 0.6113 |
| `as_service` | 0.6046 |

Chỉ số `F1` cho lớp trung lập trên từng khía cạnh:

| Khía cạnh | F1 trung lập |
| --- | ---: |
| `as_price` | 0.8359 |
| `as_service` | 0.7368 |
| `as_delivery` | 0.7143 |
| `as_content` | 0.4330 |
| `as_physical` | 0.4157 |
| `as_packaging` | 0.1818 |

Từ bảng trên có thể rút ra một số nhận xét:

1. `as_delivery` là khía cạnh dễ học nhất vì thường đi kèm tín hiệu ngôn ngữ rõ ràng như “giao nhanh”, “trễ”, “ship”.
2. `as_price` có F1 khá tốt cho lớp trung lập nhưng mẫu dữ liệu ít hơn nhiều so với các khía cạnh khác.
3. `as_packaging` khó nhất ở lớp trung lập, do cách người dùng mô tả thường rất ngắn và ít cấu trúc.
4. `as_content` và `as_physical` là hai khía cạnh quan trọng nhất về mặt nghiệp vụ, đồng thời cũng có đủ dữ liệu để mô hình học tương đối tốt.

## 3.7. Phân tích lỗi

Notebook có bước kiểm tra dự đoán trên một số mẫu test ngẫu nhiên. Kết quả cho thấy mô hình có thể:

1. Dự đoán đúng sentiment tổng thể trong nhiều trường hợp.
2. Bắt được các tín hiệu rõ ràng như giao hàng nhanh, nội dung hay, sách lỗi, bung keo.
3. Tuy nhiên, vẫn còn nhầm giữa sentiment tổng thể và sentiment của từng khía cạnh khi review chứa nhiều ý trái chiều.

Ví dụ, có trường hợp review được dự đoán là tiêu cực ở mức tổng thể nhưng lại gán sai cho khía cạnh giao hàng do review chứa nhiều tín hiệu ngữ nghĩa đồng thời. Điều này phản ánh một giới hạn tự nhiên của baseline tuyến tính: nó chủ yếu khai thác pattern bề mặt và khó mô hình hóa đầy đủ quan hệ ngữ cảnh dài, đặc biệt khi một câu chứa nhiều khía cạnh xung đột.

## 3.8. Hạn chế của baseline

Baseline Logistic Regression cho kết quả tốt, nhưng vẫn có các hạn chế rõ ràng:

1. Không xử lý ngữ cảnh sâu như mô hình ngôn ngữ lớn hoặc mô hình transformer.
2. Nhạy với cách biểu đạt khác biệt giữa các review.
3. Chất lượng trung lập thấp hơn hai lớp còn lại.
4. Các khía cạnh hiếm như `as_service` và `as_packaging` còn thiếu dữ liệu để đạt hiệu năng cao.
5. Một số cấu hình giảm chiều làm tăng thời gian tính toán mà không cải thiện chất lượng đáng kể.

Vì vậy, kết quả của notebook này nên được xem như một mốc baseline mạnh, hữu ích để so sánh với các mô hình sâu hơn trong các notebook tiếp theo như BiLSTM hoặc PhoBERT.

## 3.9. Lý thuyết giảm chiều dữ liệu

Trong bài toán xử lý văn bản, số chiều đặc trưng thường rất lớn vì mỗi từ, mỗi n-gram hoặc mỗi mẫu ký tự có thể trở thành một trục riêng trong không gian biểu diễn. Khi dùng TF-IDF, số đặc trưng có thể lên đến hàng chục nghìn hoặc hơn, làm cho mô hình:

1. Tốn nhiều bộ nhớ hơn.
2. Huấn luyện chậm hơn.
3. Dễ bị nhiễu bởi các đặc trưng hiếm hoặc ít ý nghĩa.
4. Khó tổng quát hóa nếu không gian đặc trưng quá thưa và quá lớn.

Giảm chiều là quá trình chọn lọc hoặc biến đổi bộ đặc trưng ban đầu sang một không gian nhỏ hơn nhưng vẫn giữ lại phần thông tin quan trọng nhất. Mục tiêu không phải là xóa bớt dữ liệu một cách máy móc, mà là loại bỏ phần dư thừa và giữ lại tín hiệu hữu ích cho mô hình.

### Bản chất của giảm chiều

Có thể hiểu giảm chiều theo hai hướng:

1. Chọn lọc đặc trưng: giữ lại một phần đặc trưng gốc tốt nhất, bỏ bớt phần còn lại.
2. Biến đổi đặc trưng: tạo ra các trục mới từ tổ hợp của đặc trưng gốc.

Nói ngắn gọn:

1. `chi2` là kiểu chọn đặc trưng.
2. `SVD` là kiểu biến đổi đặc trưng.
3. `L1 selection` là kiểu chọn đặc trưng tự động bằng mô hình.

Nếu một review được biểu diễn bởi 50.000 từ khác nhau nhưng thực tế chỉ vài trăm từ là thật sự có ích cho sentiment, giảm chiều giúp mô hình tập trung vào vài trăm tín hiệu quan trọng đó thay vì phải xử lý toàn bộ không gian khổng lồ.

### Vì sao văn bản cần giảm chiều

Dữ liệu văn bản khác dữ liệu số thông thường ở chỗ:

1. Không gian đặc trưng rất thưa.
2. Nhiều từ xuất hiện rất ít.
3. Nhiều từ khác nhau nhưng cùng mang một ý nghĩa gần giống nhau.
4. Một số đặc trưng chỉ xuất hiện do ngẫu nhiên, không giúp phân loại.

Ví dụ:

1. Các từ như “hay”, “tốt”, “ổn”, “đẹp” thường cùng nghiêng về tích cực.
2. Các từ như “lỗi”, “bung”, “trễ”, “sai” thường nghiêng về tiêu cực.
3. Nếu giữ quá nhiều từ hiếm hoặc nhiễu, mô hình phải học trên những tín hiệu không ổn định.

Vì vậy, giảm chiều giúp:

1. Tăng tốc huấn luyện.
2. Giảm bộ nhớ.
3. Giảm nhiễu.
4. Có thể cải thiện khả năng tổng quát hóa.

### Khi nào nên giảm chiều

Giảm chiều thường hữu ích khi:

1. Số đặc trưng quá lớn so với số mẫu.
2. Dữ liệu thưa và có nhiều đặc trưng ít xuất hiện.
3. Mô hình tuyến tính cần tăng tốc hoặc giảm bộ nhớ.
4. Muốn thử nén đặc trưng để xem chất lượng có cải thiện không.

Tuy nhiên, với bài toán văn bản, giảm chiều không phải lúc nào cũng tốt hơn. Nếu biến đổi làm mất các từ khóa quan trọng, mô hình có thể giảm chất lượng. Vì vậy, notebook baseline này thử nhiều cách giảm chiều rồi so sánh trực tiếp trên validation thay vì giả định trước rằng giảm chiều chắc chắn có lợi.

### 1. Chọn đặc trưng theo `chi2`

`chi2` là cách chọn ra các đặc trưng có liên hệ mạnh nhất với nhãn mục tiêu.

Ý tưởng trực quan:

1. Với mỗi đặc trưng, đo xem nó xuất hiện có khác biệt đáng kể giữa các lớp hay không.
2. Đặc trưng nào giúp phân biệt các lớp tốt hơn sẽ được giữ lại.
3. Đặc trưng nào gần như không liên quan đến nhãn sẽ bị loại.

Nói dễ hiểu hơn:

1. Nếu một từ thường xuyên xuất hiện trong review tiêu cực nhưng ít xuất hiện ở lớp khác, từ đó có giá trị phân biệt cao.
2. Nếu một từ xuất hiện gần như đều ở mọi lớp, nó ít giúp mô hình quyết định lớp nào.
3. `chi2` cố gắng giữ các từ có dấu vết phân loại rõ hơn.

Trong notebook, `chi2_20k` nghĩa là giữ lại khoảng 20.000 đặc trưng tốt nhất theo kiểm định `chi-square`. Cách này phù hợp với dữ liệu văn bản vì:

1. Không làm biến đổi ý nghĩa đặc trưng gốc.
2. Giữ nguyên không gian sparse.
3. Có thể giảm nhiễu mà vẫn bảo toàn các từ khóa quan trọng.

Ưu điểm của `chi2`:

1. Dễ hiểu.
2. Không làm mất diễn giải của đặc trưng gốc.
3. Hợp với TF-IDF và phân loại tuyến tính.

Nhược điểm của `chi2`:

1. Chỉ giữ đặc trưng, không tạo đặc trưng mới.
2. Có thể bỏ sót các tổ hợp đặc trưng hữu ích nếu từng từ riêng lẻ chưa đủ mạnh.
3. Kết quả phụ thuộc vào số lượng `k` được chọn.

### 2. Giảm chiều bằng `SVD`

`SVD` là cách biến đổi dữ liệu từ không gian nhiều chiều sang một không gian nhỏ hơn bằng cách tìm các thành phần chính ẩn trong ma trận đặc trưng.

Hiểu đơn giản:

1. TF-IDF ban đầu có thể rất lớn và thưa.
2. `SVD` nén ma trận này thành một số chiều nhỏ hơn, ví dụ `512`.
3. Các chiều mới không còn là từ cụ thể nữa mà là tổ hợp tuyến tính của nhiều đặc trưng gốc.

Nói cách khác, `SVD` không chọn từng từ riêng lẻ, mà học ra các trục ẩn đại diện cho những hướng biến thiên quan trọng trong dữ liệu. Những trục này có thể gom nhiều từ gần nghĩa hoặc cùng ngữ cảnh vào chung một không gian nhỏ hơn.

Ưu điểm của `SVD`:

1. Giảm mạnh số chiều.
2. Giúp mô hình chạy nhanh hơn trong một số trường hợp.
3. Có thể làm mượt nhiễu bằng cách gom các tín hiệu tương đồng vào cùng một trục ẩn.
4. Có ích khi muốn nén dữ liệu thưa thành dạng gọn hơn.

Nhược điểm:

1. Mất khả năng diễn giải trực tiếp theo từ khóa.
2. Có thể làm mờ các tín hiệu rất cụ thể, vốn quan trọng trong sentiment.
3. Có thể làm giảm hiệu năng nếu tín hiệu phân loại nằm ở các từ khóa rất rõ.

Với bài toán review sách, nhiều tín hiệu quan trọng lại nằm ở các từ khóa rõ ràng như “bung keo”, “giao nhanh”, “sách hay”, nên `SVD` đôi khi làm giảm chất lượng thay vì cải thiện.

Có thể nhớ ngắn gọn:

1. `chi2` là lọc từ tốt.
2. `SVD` là nén sang trục mới.

### 3. Chọn đặc trưng bằng `L1 selection`

`L1 selection` dựa trên Logistic Regression có chuẩn hóa `L1`. Cơ chế của `L1` là đẩy nhiều trọng số về đúng bằng 0.

Ý nghĩa:

1. Đặc trưng có ích sẽ giữ trọng số khác 0.
2. Đặc trưng yếu hoặc dư thừa sẽ bị triệt tiêu.
3. Tập đặc trưng còn lại trở nên gọn hơn.

Hiểu trực giác:

1. Mô hình tự học xem cột nào quan trọng.
2. Cột nào ít đóng góp sẽ bị triệt tiêu vì trọng số về 0.
3. Đây là cách vừa huấn luyện mô hình vừa chọn đặc trưng cùng lúc.

Cách này vừa là một dạng học mô hình, vừa là một dạng chọn lọc đặc trưng. Nó thường phù hợp khi:

1. Muốn mô hình tự học đặc trưng quan trọng.
2. Muốn giảm bớt số chiều mà không cần thiết kế thủ công.

Ưu điểm của `L1 selection`:

1. Có khả năng tự chọn đặc trưng.
2. Giữ được diễn giải ở mức vừa phải vì vẫn còn đặc trưng gốc.
3. Có thể tạo bộ đặc trưng gọn hơn.

Nhược điểm của `L1 selection`:

1. Huấn luyện chậm hơn.
2. Có thể không ổn định nếu dữ liệu nhiễu hoặc tương quan mạnh giữa các đặc trưng.
3. Có thể tạo cảm giác gọn hơn nhưng chưa chắc tốt hơn về mặt điểm số.

Trong baseline này, `l1_select_median` cho kết quả validation gần như tương đương cấu hình không giảm chiều, nhưng thời gian huấn luyện dài hơn đáng kể. Điều đó cho thấy với dữ liệu này, lợi ích của việc chọn lọc bằng `L1` chưa đủ lớn để bù chi phí tính toán.

### Kết luận ngắn về giảm chiều

Với dữ liệu review sách tiếng Việt, giảm chiều nên được xem như một công cụ thử nghiệm chứ không phải mặc định bắt buộc. Nếu không gian TF-IDF ban đầu đã mang đủ thông tin, việc giữ nguyên đặc trưng gốc đôi khi cho kết quả tốt hơn. Đây cũng chính là lý do baseline trong notebook giữ cấu hình `none` làm kết quả tốt nhất trên validation.

Tóm tắt để học nhanh:

1. `chi2`: chọn ra đặc trưng liên quan nhất đến nhãn.
2. `SVD`: nén đặc trưng sang không gian mới nhỏ hơn.
3. `L1`: tự triệt tiêu đặc trưng yếu bằng trọng số 0.
4. Giảm chiều hữu ích khi dữ liệu quá rộng, nhưng không phải lúc nào cũng thắng được giữ nguyên.

### Diễn giải dài hơn để dễ hiểu

Nếu nhìn từ góc độ thực hành, giảm chiều giống như việc bạn đang đọc một cuốn sách rất dày và phải tìm ra những trang thật sự quan trọng để ôn thi. Không phải trang nào cũng đáng đọc kỹ như nhau. Có trang chứa kiến thức cốt lõi, có trang chỉ lặp lại ý trước đó, và có trang hầu như không giúp gì cho mục tiêu của bạn. Giảm chiều trong machine learning cũng vậy: nó cố gắng giữ lại phần “trang quan trọng” và loại bớt phần lặp, phần nhiễu, phần ít giá trị.

Trong dữ liệu văn bản, đặc biệt là TF-IDF, mỗi từ hoặc n-gram giống như một mảnh thông tin riêng. Nhưng không phải mảnh nào cũng hữu ích. Có những từ xuất hiện quá thường xuyên như “sách”, “mình”, “rất”, “có”, “là”. Các từ này đúng là có mặt trong dữ liệu, nhưng lại ít giúp phân biệt sentiment. Ngược lại, những từ như “bung keo”, “giao nhanh”, “hơi trễ”, “đóng gói kém”, “nội dung hay” mới là những tín hiệu thực sự có ích cho bài toán. Giảm chiều là cách để mô hình không phải nhìn quá nhiều “nhiễu nền” mà tập trung vào tín hiệu quan trọng đó.

Với `chi2`, ta tưởng tượng như đang làm một bài kiểm tra thống kê để xem từ nào “có liên quan” nhất đến nhãn. Nếu một từ hay xuất hiện ở review tiêu cực, nhưng hiếm khi xuất hiện ở review tích cực, thì từ đó có khả năng cao là một đặc trưng tốt. Cách này giữ nguyên từ gốc, nên vẫn dễ diễn giải. Khi mô hình báo rằng một số từ như “lỗi”, “trễ”, “bung keo” có trọng số lớn, ta vẫn hiểu được vì sao nó dự đoán như vậy. Đây là lý do `chi2` thường được xem là cách giảm chiều an toàn trong bài toán văn bản.

Tuy nhiên, `chi2` cũng có giới hạn. Nó chỉ biết chọn từ tốt hơn trong số các từ đang có, chứ không tạo ra kiến trúc biểu diễn mới. Nếu dữ liệu chứa nhiều từ khác nhau nhưng cùng chỉ một ý niệm chung, `chi2` sẽ vẫn phải làm việc với danh sách từ gốc khá lớn. Nói cách khác, `chi2` giống như việc lọc sách ra thành một danh sách những trang đáng đọc nhất, nhưng bạn vẫn phải đọc từng trang riêng rẽ.

`SVD` thì khác. Nó không giữ nguyên từng từ mà nén cả ma trận đặc trưng thành các trục ẩn. Bạn có thể hiểu nó như việc gộp nhiều trang sách thành vài chương tổng hợp. Một chương có thể đại diện cho nhiều từ cùng mang ý nghĩa tích cực, một chương khác đại diện cho các tín hiệu về giao hàng, một chương khác nữa gợi ý về chất lượng in ấn hay đóng gói. Cách này rất mạnh khi dữ liệu có nhiều đặc trưng chồng chéo hoặc khi muốn giảm mạnh kích thước đầu vào. Nhưng đổi lại, bạn mất khả năng truy ngược trực tiếp từ trục ẩn về từng từ cụ thể.

Vì vậy `SVD` có một điểm yếu khá rõ với bài toán sentiment. Tín hiệu cảm xúc thường nằm ở các từ khóa rất trực diện. Nếu nén quá mạnh, nhiều từ khóa quan trọng có thể bị hòa lẫn vào nhau. Ví dụ, “bung keo” và “hơi cứng” không hẳn cùng một kiểu tín hiệu. Nếu nén không khéo, mô hình sẽ khó phân biệt giữa vấn đề chất lượng giấy, vấn đề đóng gáy, hay vấn đề đóng gói. Do đó, dù `SVD` rất hấp dẫn về mặt lý thuyết, trên dữ liệu này nó chưa chắc tốt hơn giữ nguyên đặc trưng gốc.

`L1 selection` lại là một cách tiếp cận kiểu khác. Ở đây mô hình Logistic Regression tự đẩy các trọng số không quan trọng về 0. Điều đó có nghĩa là mô hình đang tự nói: “những đặc trưng này không cần thiết, bỏ đi cũng được”. Cách này khá tự nhiên vì nó vừa học mô hình vừa chọn đặc trưng. Nhưng nó không hề miễn phí. Nếu dữ liệu nhiễu nhiều hoặc các đặc trưng tương quan mạnh với nhau, việc học bằng `L1` có thể chậm hơn và không ổn định bằng cách giữ nguyên hoặc chỉ lọc nhẹ bằng `chi2`.

Nhìn vào kết quả thực nghiệm trong notebook, cấu hình không giảm chiều (`none`) cho `val_f1_combined` tốt nhất. Điều này rất đáng nhớ vì nó cho thấy một nguyên tắc quan trọng trong xử lý văn bản: không phải cứ giảm chiều là tốt hơn. Nếu bộ đặc trưng TF-IDF ban đầu đã được xây dựng khá hợp lý, nó có thể đã đủ gọn để mô hình tuyến tính học tốt ngay trên đó. Khi đó, giảm chiều chỉ làm mất bớt tín hiệu mà không đem lại lợi ích tương xứng.

Nếu cần ghi nhớ một câu thật ngắn, có thể nhớ thế này:

1. `chi2` lọc ra từ tốt nhất.
2. `SVD` nén dữ liệu sang trục ẩn.
3. `L1` để mô hình tự bỏ bớt đặc trưng yếu.
4. Dữ liệu văn bản không phải lúc nào cũng cần giảm chiều mạnh.

## 4. Thực nghiệm mô hình BiLSTM

## 4.1. Mục tiêu thực nghiệm

Notebook `notebooks/04_absa_bilstm.ipynb` mở rộng từ baseline tuyến tính sang mô hình học sâu với kiến trúc BiLSTM kết hợp attention. Mục tiêu của phần thực nghiệm này là kiểm tra xem:

1. Mô hình tuần tự hai chiều có học tốt hơn baseline TF-IDF + Logistic Regression hay không.
2. Embedding nào phù hợp hơn cho dữ liệu review sách tiếng Việt: Word2Vec, FastText hay PhoBERT.
3. Kiến trúc ABSA hai tầng với các head riêng cho sentiment, aspect presence và aspect sentiment có cải thiện hiệu quả trên dữ liệu thưa hay không.

Khác với notebook baseline, phần này đi sâu hơn vào biểu diễn chuỗi và ngữ cảnh. Điều đó quan trọng vì review thực tế thường không chỉ có từ khóa đơn lẻ mà còn có cấu trúc câu, liên kết giữa các mệnh đề và các tín hiệu phụ thuộc ngữ cảnh.

## 4.2. Thiết lập dữ liệu

Notebook sử dụng các file đã làm sạch:

1. `train_clean.json`
2. `val_clean.json`
3. `test_clean.json`

Dữ liệu đầu vào được ghép từ `review_title` và `content` thành `text_full`, sau đó tách từ bằng `ViTokenizer` để tạo `text_seg`. Cách này giúp mô hình học trên token tiếng Việt ổn định hơn, đặc biệt với các cụm từ nhiều âm tiết.

Các nhãn được giữ lại gồm:

1. `sentiment_llm`
2. `as_content`
3. `as_physical`
4. `as_price`
5. `as_packaging`
6. `as_delivery`
7. `as_service`

Các giá trị `null` ở khía cạnh được quy về `ABSENT_CLASS = 3` để biểu diễn trạng thái “không nhắc đến”.

## 4.3. Cấu hình huấn luyện

Các siêu tham số chính của notebook:

1. `MAX_LENGTH = 160`
2. `HIDDEN_DIM = 256`
3. `NUM_LAYERS = 2`
4. `DROPOUT = 0.2`
5. `EPOCHS = 8`
6. `LEARNING_RATE = 1e-3`

Kích thước batch:

1. `BATCH_SIZE_TRADITIONAL = 128` cho Word2Vec và FastText.
2. `BATCH_SIZE_PHOBERT_TRAIN = 32` cho huấn luyện PhoBERT.
3. `BATCH_SIZE_PHOBERT_EVAL = 64` cho đánh giá.

Notebook cũng dùng chiến lược trọng số động:

1. Trọng số lớp cho sentiment được lấy từ `compute_class_weight(..., balanced)` rồi lấy căn bậc hai.
2. Trọng số presence cho từng aspect cũng được tính tương tự.
3. Điều này giúp mô hình bớt lệch về các lớp chiếm ưu thế.

## 4.4. Tiền xử lý cho mô hình BiLSTM

Pipeline của notebook không dùng văn bản raw trực tiếp mà chuẩn bị lại thành:

1. `title`
2. `body`
3. `text_full = title + body`
4. `text_seg = ViTokenizer.tokenize(text_full.lower())`

Lý do cần bước này:

1. Ghép title và body giúp không bỏ sót tín hiệu sentiment ở tiêu đề.
2. Lowercase làm dữ liệu ổn định hơn.
3. Word segmentation giúp tách đúng ranh giới từ tiếng Việt.

Trong thực nghiệm, dữ liệu sau tách từ được dùng làm đầu vào cho cả ba biến thể embedding.

## 4.5. Kiến trúc mô hình

Notebook xây dựng một kiến trúc nền chung `BaseABSABiLSTM`, sau đó thay phần embedding đầu vào theo từng biến thể.

### Khối xử lý chuỗi

Luồng xử lý chính gồm:

1. `Embedding`
2. `SpatialDropout1D`
3. `BiLSTM`
4. `MultiheadAttention`
5. `Attention pooling`
6. `Max pooling`
7. `Average pooling`
8. Ghép ba vector ngữ cảnh lại với nhau

Ý nghĩa của từng phần:

1. `BiLSTM` học ngữ cảnh hai chiều, tức là vừa nhìn trái sang phải vừa nhìn phải sang trái.
2. `MultiheadAttention` giúp mô hình tập trung vào các token quan trọng hơn trong chuỗi.
3. `attention pooling`, `max pooling` và `average pooling` gom thông tin theo nhiều góc nhìn khác nhau.
4. Việc ghép ba kiểu pooling giúp mô hình giữ được cả tín hiệu nổi bật lẫn tín hiệu tổng quát.

### Các head đầu ra

Từ vector ngữ cảnh chung, mô hình tách thành ba nhánh:

1. `sent_classifier` cho sentiment tổng thể.
2. `pres_classifier` cho sự xuất hiện của từng aspect.
3. `asp_classifier` cho sentiment của aspect khi aspect đó xuất hiện.

Thiết kế này phản ánh đúng cấu trúc ABSA: không chỉ “review tốt hay xấu” mà còn phải biết “khía cạnh nào được nhắc đến” và “khía cạnh đó mang cảm xúc gì”.

## 4.6. Ba biến thể embedding

Notebook thử ba kiểu embedding:

1. `BiLSTM + Word2Vec`
2. `BiLSTM + FastText`
3. `BiLSTM + PhoBERT`

### Word2Vec

Word2Vec học embedding từ chính tập train. Ưu điểm là nhẹ, dễ chạy và không phụ thuộc mô hình ngoài. Nhược điểm là embedding chỉ học từ corpus hiện có, nên nếu corpus không đủ lớn thì khó bao phủ tốt các biến thể ngôn ngữ.

### FastText

FastText bổ sung thông tin n-gram ký tự trong từ, nên thường có lợi với tiếng Việt vì dữ liệu thực tế có nhiều biến thể chính tả, teencode và cách viết không chuẩn. Tuy nhiên, trong notebook này FastText vẫn chưa vượt Word2Vec.

### PhoBERT

PhoBERT là embedding mạnh nhất trong ba biến thể vì nó mang kiến thức ngôn ngữ tiền huấn luyện từ trước. Khi đưa PhoBERT vào BiLSTM, mô hình vừa có biểu diễn ngữ nghĩa giàu hơn vừa vẫn giữ được lớp xử lý tuần tự của BiLSTM.

## 4.7. Hàm mất mát và tối ưu

Notebook không dùng một loss duy nhất cho toàn bộ bài toán mà tách thành ba thành phần:

1. `loss_sent` cho sentiment.
2. `loss_pres` cho presence của aspect.
3. `loss_asp` cho sentiment của aspect.

Các thành phần loss được kết hợp bằng `AutomaticWeightedLoss`, tức là mô hình tự học trọng số tương đối cho từng nhiệm vụ phụ.

Điểm quan trọng là:

1. `FocalLoss` được dùng cho sentiment để giảm ảnh hưởng của lớp dễ và tập trung vào mẫu khó.
2. `CrossEntropyLoss` được dùng cho presence và aspect sentiment.
3. `label_smoothing` được áp dụng cho aspect sentiment để giảm overconfidence.

Chiến lược này hợp lý với bài toán ABSA vì đây là bài toán đa nhiệm và mất cân bằng rõ rệt.

## 4.8. Kết quả so sánh ba embedding

Kết quả tổng kết trên tập test cho ba biến thể:

| Mô hình | F1 Sentiment | F1 Aspects | F1 Final |
| --- | ---: | ---: | ---: |
| BiLSTM + Word2Vec | 0.7920 | 0.6050 | 0.6985 |
| BiLSTM + FastText | 0.7772 | 0.6019 | 0.6895 |
| BiLSTM + PhoBERT | 0.8341 | 0.6227 | 0.7284 |

Nhận xét:

1. PhoBERT cho kết quả tốt nhất ở cả sentiment lẫn điểm tổng hợp.
2. Word2Vec đứng thứ hai.
3. FastText không vượt Word2Vec trong thiết lập này, dù về mặt lý thuyết FastText thường có lợi với dữ liệu nhiễu.

Điều này cho thấy sức mạnh của biểu diễn tiền huấn luyện là rất quan trọng. Khi embedding đã mang kiến thức ngôn ngữ tốt hơn, BiLSTM phía sau có nhiều tín hiệu để học hơn.

## 4.9. Kết quả chi tiết của PhoBERT

Trong phần đánh giá chi tiết trên test, mô hình BiLSTM + PhoBERT đạt:

| Chỉ số | Giá trị |
| --- | ---: |
| `eval_f1_sentiment` | 0.8341 |
| `eval_f1_aspect_present` | 0.8303 |
| `eval_f1_aspect_all` | 0.8107 |
| `eval_f1_combined` | 0.8322 |
| `eval_accuracy` | 0.8813 |

Kết quả sentiment tổng thể:

| Lớp | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.93 | 0.97 | 0.95 |
| Neutral | 0.64 | 0.71 | 0.67 |
| Positive | 0.95 | 0.83 | 0.89 |

Kết quả cho 6 aspect ở chế độ `present only`:

| Lớp | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.91 | 0.82 | 0.87 |
| Neutral | 0.81 | 0.63 | 0.71 |
| Positive | 0.94 | 0.89 | 0.91 |

Đây là mức cải thiện rõ rệt so với baseline Logistic Regression, đặc biệt ở sentiment tổng thể và độ cân bằng giữa các lớp.

## 4.10. Phân tích theo từng aspect

Chỉ số F1 theo từng khía cạnh trên test:

| Khía cạnh | F1 |
| --- | ---: |
| `as_content` | 0.7586 |
| `as_physical` | 0.7904 |
| `as_price` | 0.7944 |
| `as_packaging` | 0.7827 |
| `as_delivery` | 0.8301 |
| `as_service` | 0.7082 |

Nhận xét:

1. `as_delivery` tiếp tục là khía cạnh dễ học nhất.
2. `as_price`, `as_physical` và `as_packaging` đều đạt mức cao hơn so với baseline tuyến tính.
3. `as_service` vẫn là khía cạnh khó nhất vì số mẫu ít.

So với notebook baseline, đây là cải thiện khá rõ ở nhiều aspect, cho thấy BiLSTM + PhoBERT học được ngữ cảnh tốt hơn.

## 4.11. Phân tích lỗi

Notebook lưu lại một file `error_analysis.csv` để kiểm tra các mẫu dự đoán sai. Các lỗi điển hình gồm:

1. Đoán đúng sentiment tổng thể nhưng bỏ sót một aspect cụ thể.
2. Nhận diện sai aspect xuất hiện do câu có nhiều ý đan xen.
3. Nhầm giữa neutral và positive khi câu mang tính nhận xét nửa khen nửa chê.

Ví dụ, một review có thể khen giao hàng nhưng chê hình thức và nội dung. Khi đó mô hình phải xử lý đồng thời nhiều tín hiệu trái ngược. Đây là bài toán khó hơn nhiều so với phân loại một nhãn chung duy nhất.

## 4.12. Hạn chế của mô hình

Dù tốt hơn baseline, mô hình BiLSTM vẫn có các giới hạn:

1. Huấn luyện nặng hơn và chậm hơn đáng kể so với mô hình tuyến tính.
2. Phụ thuộc vào chất lượng segmentation và embedding.
3. Vẫn còn yếu ở các khía cạnh ít mẫu như `as_service`.
4. Việc tinh chỉnh threshold cho presence vẫn có ảnh hưởng đáng kể đến kết quả cuối cùng.

Tuy vậy, đây là một bước tiến rõ rệt vì mô hình đã tận dụng được ngữ cảnh chuỗi và kiến thức tiền huấn luyện, nên phù hợp hơn cho bài toán ABSA thực tế.

## 5. Thực nghiệm với các bộ tách từ

## 5.1. Mục tiêu thực nghiệm

Notebook `notebooks/04.1-bilstm-wordsegmenters.ipynb` được xây dựng để trả lời một câu hỏi rất thực tế: với tiếng Việt, nên tách từ bằng công cụ nào trước khi đưa vào PhoBERT + BiLSTM?

Bài toán này quan trọng vì segmentation ảnh hưởng trực tiếp đến:

1. Ranh giới token đầu vào.
2. Chất lượng biểu diễn ngữ nghĩa.
3. Khả năng nhận diện các cụm từ nhiều âm tiết.
4. Độ ổn định của mô hình khi xử lý câu review nhiễu hoặc viết không chuẩn.

Notebook so sánh bốn lựa chọn:

1. Không tách từ (`No_Segmenter`)
2. `Pyvi`
3. `Underthesea`
4. `VnCoreNLP`

## 5.2. Các bộ tách từ được so sánh

### Không tách từ

Đây là baseline đơn giản nhất: giữ nguyên văn bản sau lowercase, không áp dụng word segmentation. Ưu điểm là nhanh và ít phụ thuộc công cụ ngoài. Nhược điểm là PhoBERT và BiLSTM phải tự xử lý ranh giới từ từ chuỗi raw, trong khi tiếng Việt có nhiều từ nhiều âm tiết nên ranh giới không rõ nếu không tách.

### Pyvi

Pyvi là một lựa chọn nhẹ, dễ dùng và thường ổn định trên nhiều đoạn văn ngắn. Nó phù hợp khi cần cân bằng giữa tốc độ và độ chính xác.

### Underthesea

Underthesea cho cách tách từ khá phổ biến trong xử lý tiếng Việt. Nó thường dùng tốt cho các bài toán NLP cổ điển và dễ tích hợp.

### VnCoreNLP

VnCoreNLP là bộ công cụ mạnh và khá phổ biến trong tiếng Việt. Điểm mạnh của nó là có thể cho kết quả tách từ ổn định trên nhiều kiểu câu khác nhau, nhưng việc chạy mô hình này thường nặng hơn và phụ thuộc môi trường Java.

## 5.3. Thiết lập mô hình

Notebook này vẫn dùng kiến trúc chính là:

1. PhoBERT làm encoder.
2. BiLSTM phía sau để xử lý chuỗi biểu diễn.
3. Multi-head attention và pooling để gom ngữ cảnh.
4. Ba nhánh đầu ra cho sentiment, aspect presence và aspect sentiment.

Khác biệt duy nhất giữa các thí nghiệm là bộ tách từ được áp dụng trước khi token hóa PhoBERT. Điều đó giúp cô lập được tác động của segmentation lên kết quả.

## 5.4. Cấu hình huấn luyện

Thiết lập chính:

1. `MAX_LENGTH = 160`
2. `HIDDEN_DIM = 256`
3. `NUM_LAYERS = 2`
4. `DROPOUT = 0.2`
5. `EPOCHS = 8`
6. `BATCH_SIZE = 128`
7. `LEARNING_RATE = 1e-3`
8. `WEIGHT_DECAY = 1e-4`

Notebook cũng dùng:

1. `FocalLoss` cho sentiment.
2. `AutomaticWeightedLoss` cho đa nhiệm.
3. Trọng số lớp động để giảm mất cân bằng.

## 5.5. Kiến trúc mô hình

Kiến trúc PhoBERT + BiLSTM trong notebook này về cơ bản giống notebook trước, nhưng được đóng gói lại thành một pipeline dễ so sánh giữa các bộ tách từ:

1. Segmenter xử lý văn bản.
2. PhoBERT tạo embedding theo ngữ cảnh.
3. BiLSTM tiếp tục khai thác thứ tự chuỗi.
4. Attention và pooling rút gọn thành vector ngữ cảnh.
5. Ba nhánh đầu ra xử lý sentiment và aspect.

Điểm đáng chú ý là, khi segmentation thay đổi, token đầu vào của PhoBERT thay đổi, nên toàn bộ biểu diễn phía sau cũng thay đổi theo.

## 5.6. Cách đánh giá

Notebook đánh giá trên ba mức:

1. Sentiment tổng thể.
2. Mean 6 aspects ở chế độ present only.
3. `F1_Final` là trung bình của hai giá trị trên.

Chỉ số này giúp nhìn rõ hơn tác động của từng bộ tách từ lên cả sentiment lẫn aspect.

## 5.7. Kết quả so sánh

Notebook cho ra bảng so sánh sau:

| Segmenter | F1 Overall Sentiment | F1 Mean Aspects | F1 Final |
| --- | ---: | ---: | ---: |
| No_Segmenter | 0.8295 | 0.5996 | 0.7146 |
| Pyvi | 0.8473 | 0.5895 | 0.7184 |
| Underthesea | 0.8468 | 0.5939 | 0.7204 |
| VnCoreNLP | 0.8449 | 0.5868 | 0.7158 |

Kết quả cho thấy:

1. `Underthesea` đạt `F1 Final` cao nhất và được chọn làm segmenter chiến thắng trong phần đánh giá test.
2. `Pyvi` có `F1 Overall Sentiment` cao nhất nhưng điểm aspect trung bình thấp hơn `Underthesea`.
3. `No_Segmenter` vẫn khá mạnh ở sentiment tổng thể, nhưng yếu hơn khi xét cả aspect.
4. `VnCoreNLP` ổn định nhưng không vượt được hai cấu hình dẫn đầu trong thiết lập này.

Điều này phản ánh một thực tế thường gặp trong tiếng Việt: chọn tách từ không chỉ ảnh hưởng đến một chỉ số duy nhất, mà tác động khác nhau lên sentiment tổng thể và sentiment theo khía cạnh. Vì vậy, cách chọn tốt nhất là nhìn vào chỉ số tổng hợp thay vì chỉ nhìn riêng sentiment hoặc riêng aspect.

Điều quan trọng rút ra từ thiết kế notebook là:

1. Mục tiêu chính không chỉ là chọn segmenter tốt nhất, mà còn hiểu segmentation ảnh hưởng ra sao đến mô hình.
2. Với tiếng Việt, segmentation là bước có tác động thật sự lớn, không thể xem nhẹ.
3. Segmenter tốt nhất không chỉ dựa vào điểm số, mà còn cần xét đến tốc độ, độ ổn định và khả năng triển khai.

## 5.8. Ý nghĩa thực nghiệm

Thí nghiệm này có giá trị vì nó giúp tách riêng ảnh hưởng của word segmentation khỏi kiến trúc mô hình. Nếu BiLSTM hay PhoBERT cho kết quả tốt nhưng segmentation kém, toàn bộ pipeline vẫn có thể bị giảm chất lượng.

Từ góc nhìn học thuật, đây là một bước kiểm tra cần thiết đối với NLP tiếng Việt. Từ góc nhìn ứng dụng, nó giúp chọn ra tiền xử lý phù hợp trước khi triển khai mô hình trên dữ liệu thực tế.

## 6. Thực nghiệm cân bằng lớp với PhoBERT

## 6.1. Mục tiêu thực nghiệm

Notebook `notebooks/05_phobert_balance_experiment.ipynb` là bước tiếp theo sau BiLSTM, nhưng tập trung vào một câu hỏi khác: làm thế nào để cải thiện hiệu năng PhoBERT khi dữ liệu mất cân bằng và các aspect rất thưa?

Notebook này không chỉ fine-tune `vinai/phobert-base-v2`, mà còn thử nhiều chiến lược cân bằng lớp và điều chỉnh loss để xử lý:

1. Lớp `neutral` khó học hơn sentiment.
2. Một số aspect như `service`, `price`, `packaging` có rất ít mẫu.
3. Nếu chỉ dùng CE thuần, mô hình dễ thiên lệch về các lớp đông hoặc bỏ sót các mẫu khó.

## 6.2. Ý tưởng chính

Notebook xuất phát từ các quan sát sau:

1. Review trong dataset thường ngắn.
2. `neutral` là lớp khó nhất trong sentiment.
3. Các aspect không đồng đều về mật độ.
4. Kiến trúc hai tầng cho aspect là cần thiết vì aspect có thể xuất hiện hoặc không xuất hiện.

Từ đó, notebook kết hợp các hướng sau:

1. `class-balanced weights`
2. `joint balanced sampler`
3. `improved focal loss`
4. `neutral-aware configuration`
5. `supervised contrastive loss` ở một số experiment

## 6.3. Cấu hình mô hình

Notebook dùng:

1. `MODEL_NAME = vinai/phobert-base-v2`
2. `MAX_LENGTH = 160`
3. `BATCH_SIZE = 16`
4. `EPOCHS = 7`
5. `LEARNING_RATE = 2e-5`

Thiết lập loss quan trọng:

1. `BASE_FOCAL_GAMMA = 2.0`
2. `CLASS_BALANCED_BETA = 0.999`
3. `STAGE1_LOSS_WEIGHT = 0.25`
4. `STAGE2_LOSS_WEIGHT = 0.75`
5. `IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT = 2.5`
6. `IMPROVED_FOCAL_ASPECT_GAMMA_ABSENT = 1.0`

Ở đây, `neutral` được xử lý cẩn thận hơn vì đây là lớp dễ bị mô hình bỏ qua khi chỉ tối ưu tổng thể.

## 6.4. Kiến trúc

Kiến trúc model là PhoBERT theo kiểu 2-stage aspect head:

1. `sentiment_head` dự đoán sentiment tổng thể.
2. `presence_heads` dự đoán aspect có xuất hiện hay không.
3. `aspect_sentiment_heads` dự đoán sentiment cho từng aspect khi aspect đó xuất hiện.

Logits được ghép thành một vector duy nhất để phục vụ train/eval, sau đó tách ra khi tính metrics.

Thiết kế này phù hợp với ABSA vì nó làm rõ hai bài toán con:

1. Phát hiện aspect.
2. Phân loại sentiment của aspect đó.

## 6.5. Chiến lược cân bằng lớp

Notebook thử nhiều chiến lược:

1. `baseline`
2. `clean_class_balanced_ce`
3. `clean_joint_balanced_ce`
4. `clean_joint_balanced_focal`
5. `clean_joint_balanced_neutral_aspect`
6. `clean_joint_balanced_neutral_scl`

Ý nghĩa:

1. `baseline` là CE thuần.
2. `class_balanced_ce` thêm class-balanced weights cho sentiment.
3. `joint_balanced_ce` thêm balanced sampler.
4. `joint_balanced_focal` dùng focal loss cải tiến.
5. `neutral_aspect` tập trung hơn vào lớp neutral của aspect.
6. `neutral_scl` thêm supervised contrastive loss để tăng separability.

## 6.6. Kết quả validation

Bảng so sánh validation của notebook:

| Experiment | Val F1 Sentiment | Val F1 Aspect Present | Val F1 Combined | Val Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `clean_joint_balanced_focal` | 0.8322 | 0.6931 | 0.7627 | 0.8811 |
| `clean_joint_balanced_neutral_aspect` | 0.8214 | 0.6905 | 0.7560 | 0.8703 |
| `clean_joint_balanced_neutral_scl` | 0.8214 | 0.6905 | 0.7560 | 0.8703 |
| `clean_joint_balanced_ce` | 0.8336 | 0.5578 | 0.6957 | 0.8459 |
| `clean_class_balanced_ce` | 0.8360 | 0.5182 | 0.6771 | 0.8473 |
| `baseline` | 0.8418 | 0.4558 | 0.6488 | 0.8403 |

Kết luận chính:

1. `clean_joint_balanced_focal` là experiment tốt nhất theo `val_f1_combined`.
2. Các biến thể chỉ dùng CE nhưng không có sampler tốt hơn về sentiment nhưng yếu về aspect.
3. `joint balanced` quan trọng hơn việc chỉ thêm class weights.
4. Focal loss giúp mô hình cải thiện rõ hơn ở aspect presence và aspect khó.

## 6.7. Kết quả test

Mô hình tốt nhất là `clean_joint_balanced_focal`. Kết quả test raw:

| Chỉ số | Giá trị |
| --- | ---: |
| `eval_f1_sentiment` | 0.8415 |
| `eval_f1_aspect_present` | 0.6898 |
| `eval_f1_aspect_all` | 0.6826 |
| `eval_f1_combined` | 0.7657 |
| `eval_accuracy` | 0.8824 |

Sau khi calibration ngưỡng presence trên validation:

| Chỉ số | Giá trị |
| --- | ---: |
| `eval_f1_sentiment` | 0.8415 |
| `eval_f1_aspect_present` | 0.7893 |
| `eval_f1_aspect_all` | 0.3556 |
| `eval_f1_combined` | 0.8154 |

Điểm đáng chú ý là:

1. Calibration giúp tăng mạnh `f1_aspect_present`.
2. Nhưng `f1_aspect_all` giảm mạnh vì ngưỡng presence được tối ưu cho phần present only.
3. Điều này cho thấy metric phải được đọc đúng ngữ cảnh, không thể nhìn một con số đơn lẻ.

Kết quả classification report trên test với ngưỡng calibrated:

| Lớp sentiment | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.95 | 0.92 | 0.94 |
| Neutral | 0.65 | 0.71 | 0.68 |
| Positive | 0.90 | 0.91 | 0.91 |

Kết quả aspect present only:

| Lớp aspect | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.80 | 0.84 | 0.82 |
| Neutral | 0.73 | 0.57 | 0.64 |
| Positive | 0.90 | 0.92 | 0.91 |

## 6.8. Phân tích theo aspect

Chỉ số F1 theo từng khía cạnh trên test:

| Khía cạnh | F1 |
| --- | ---: |
| `as_content` | 0.5942 |
| `as_physical` | 0.5868 |
| `as_price` | 0.6674 |
| `as_packaging` | 0.5501 |
| `as_delivery` | 0.7572 |
| `as_service` | 0.5808 |

So với BiLSTM + PhoBERT ở notebook trước, mô hình này tập trung mạnh hơn vào cân bằng lớp và calibrate threshold, nên sentiment tổng thể vẫn tốt nhưng aspect-wise không phải lúc nào cũng vượt lên tuyệt đối. Đây là cái giá của việc tối ưu vào presence gate và neutral-aware training.

## 6.9. Phân tích lỗi

Notebook có phần error analysis cho thấy:

1. Mô hình vẫn có thể đoán sai khi review chứa nhiều mệnh đề trái chiều.
2. Lỗi thường tập trung ở `neutral` và các aspect thưa.
3. Nếu threshold presence quá thấp, mô hình dễ “nhìn thấy” quá nhiều aspect.
4. Nếu threshold quá cao, mô hình lại bỏ sót aspect thật.

Một số thống kê lỗi cho thấy sau calibration, số câu lỗi theo present-only giảm mạnh, nhưng lỗi ở aspect toàn bộ tăng theo cách đánh đổi metric. Đây là dấu hiệu rõ của việc threshold cần được đọc cùng với mục tiêu đánh giá, không thể tối ưu chung chung.

## 6.10. Hạn chế và ý nghĩa

Notebook này cho thấy một điểm quan trọng:

1. Cân bằng lớp không chỉ là thêm weight vào loss.
2. Với ABSA, phải cân bằng cả sampling, loss và threshold.
3. Cải thiện một metric có thể làm giảm metric khác.

Về mặt thực nghiệm, đây là bước nâng cấp đáng kể vì mô hình đã xử lý tốt hơn lớp khó và aspect hiếm. Về mặt học thuật, notebook này cho thấy một pipeline ABSA thực tế cần phối hợp nhiều kỹ thuật thay vì chỉ dựa vào một backbone mạnh.

## 3.10. Lý thuyết TF-IDF

TF-IDF là một cách biến văn bản thành số để máy học có thể xử lý. Ý tưởng của nó là:

1. Từ nào xuất hiện nhiều trong một văn bản thì quan trọng hơn đối với văn bản đó.
2. Nhưng từ nào xuất hiện ở quá nhiều văn bản thì lại ít có khả năng giúp phân biệt các lớp.

TF-IDF kết hợp hai thành phần:

1. `TF` - term frequency, tức tần suất xuất hiện của từ trong một văn bản.
2. `IDF` - inverse document frequency, tức mức độ hiếm của từ trên toàn bộ tập dữ liệu.

Trực giác:

1. Từ như “sách”, “rất”, “và” xuất hiện rất nhiều nên thường không mang nhiều sức phân biệt.
2. Từ như “bung keo”, “giao nhanh”, “trễ”, “hài lòng” lại mang tín hiệu rõ hơn cho sentiment hoặc aspect.

Trong notebook, TF-IDF được dùng ở hai mức:

1. `word TF-IDF` để bắt các cụm từ có nghĩa rõ ràng.
2. `char TF-IDF` để bắt các mảnh ký tự, hữu ích khi văn bản có lỗi chính tả, teencode hoặc biến thể không chuẩn.

Nhờ vậy, mô hình có thể tận dụng cả tín hiệu ngôn ngữ chuẩn và tín hiệu bề mặt của dữ liệu review tiếng Việt.

## 3.11. Lý thuyết Logistic Regression

Logistic Regression là mô hình phân loại tuyến tính. Dù tên là “regression”, nó thường được dùng để dự đoán nhãn lớp.

Ý tưởng cốt lõi:

1. Mỗi đặc trưng được gán một trọng số.
2. Mô hình tính tổng có trọng số của các đặc trưng đầu vào.
3. Tổng này được biến đổi thành xác suất thuộc về từng lớp.

Đặc điểm của Logistic Regression:

1. Đơn giản, dễ huấn luyện.
2. Chạy khá nhanh trên dữ liệu thưa.
3. Hoạt động tốt với TF-IDF vì cả hai đều là dạng tuyến tính.
4. Dễ giải thích hơn nhiều mô hình sâu.

Trong bài toán sentiment, nếu một review chứa các từ như “hay”, “đẹp”, “hài lòng”, mô hình có thể đẩy xác suất về lớp tích cực. Ngược lại, các từ như “lỗi”, “trễ”, “bung keo” sẽ kéo xác suất về lớp tiêu cực.

Hạn chế của mô hình này là nó chủ yếu học quan hệ tuyến tính, nên khó bắt được ngữ cảnh phức tạp, phủ định xa, hoặc câu có nhiều khía cạnh trái ngược trong cùng một review.

## 3.12. Lý thuyết ABSA hai tầng

ABSA trong notebook này được triển khai theo tư duy hai tầng:

1. Tầng 1: xác định khía cạnh có xuất hiện hay không.
2. Tầng 2: nếu có xuất hiện thì dự đoán polarity cho khía cạnh đó.

Cách làm này phù hợp vì trong dữ liệu thực tế, phần lớn khía cạnh là `null` đối với một review bất kỳ. Nếu bắt mô hình dự đoán sentiment cho mọi khía cạnh ở mọi câu, nhiều dự đoán sẽ vô nghĩa.

Quy trình vận hành:

1. Mô hình presence ước lượng xác suất khía cạnh được nhắc đến.
2. Nếu xác suất nhỏ hơn ngưỡng, gán `ABSENT_LABEL`.
3. Nếu xác suất đủ lớn, dùng mô hình sentiment của khía cạnh để dự đoán `0`, `1` hoặc `2`.

Ưu điểm:

1. Giảm lỗi khi khía cạnh không xuất hiện.
2. Tách rõ bài toán phát hiện và phân loại cảm xúc.
3. Phù hợp với dữ liệu ABSA thưa và mất cân bằng.

Nhược điểm:

1. Lỗi ở tầng presence sẽ lan sang tầng sentiment.
2. Cần chọn ngưỡng cẩn thận trên validation.

## 3.13. Lý thuyết macro F1

`F1` là thước đo cân bằng giữa `precision` và `recall`.

1. `precision` trả lời câu hỏi: trong các dự đoán dương tính, có bao nhiêu dự đoán đúng.
2. `recall` trả lời câu hỏi: trong các mẫu dương tính thật, mô hình bắt được bao nhiêu.
3. `F1` là trung bình điều hòa của hai đại lượng trên.

Khi dữ liệu mất cân bằng, `accuracy` có thể đánh lừa vì mô hình chỉ cần đoán lớp nhiều nhất là đã cao điểm. `macro F1` khắc phục điều này bằng cách:

1. Tính F1 riêng cho từng lớp.
2. Lấy trung bình đều giữa các lớp.

Vì vậy, `macro F1` phù hợp hơn để đánh giá sentiment 3 lớp và ABSA nhiều khía cạnh, nhất là khi lớp trung lập hoặc các khía cạnh hiếm có số lượng ít hơn.

## 3.14. Lý thuyết ma trận nhầm lẫn

Ma trận nhầm lẫn là bảng cho biết mô hình đã dự đoán nhãn nào khi nhãn thật là gì.

Ý nghĩa:

1. Ô đường chéo chính là các dự đoán đúng.
2. Ô ngoài đường chéo cho biết mô hình hay nhầm lớp nào với lớp nào.

Trong bài toán sentiment:

1. Nếu `neutral` bị nhầm sang `positive`, mô hình có xu hướng quá lạc quan.
2. Nếu `positive` bị nhầm sang `negative`, mô hình có thể quá khắt khe.

Trong bài toán ABSA, confusion matrix còn giúp xem khía cạnh nào dễ bị nhầm nhất, ví dụ `packaging` với `physical`, hoặc `delivery` với `service`.

## 3.15. Lý thuyết class imbalance

Class imbalance là tình huống số lượng mẫu giữa các lớp không đều nhau.

Ví dụ trong dữ liệu này:

1. `sentiment_llm` có lớp tiêu cực nhiều hơn lớp trung lập.
2. Một số khía cạnh như `as_service` rất ít mẫu.

Vấn đề của mất cân bằng:

1. Mô hình có thể thiên về lớp đông.
2. Chỉ số accuracy vẫn cao nhưng F1 của lớp hiếm lại thấp.

Giải pháp thường dùng:

1. `class_weight = balanced`
2. Chọn ngưỡng hợp lý
3. Dùng macro F1 thay vì chỉ accuracy
4. Bổ sung dữ liệu cho lớp hiếm nếu có thể

Trong notebook baseline, `class_weight = balanced` được dùng để giảm thiên lệch về các lớp xuất hiện nhiều hơn.

## 3.16. Lý thuyết class-balanced loss

Class-balanced loss là cách điều chỉnh trọng số loss dựa trên số lượng mẫu thật sự của từng lớp. Ý tưởng không chỉ là nói rằng lớp hiếm quan trọng hơn, mà còn định lượng mức độ quan trọng đó theo tần suất xuất hiện.

Về trực giác:

1. Nếu một lớp có rất nhiều mẫu, mô hình nhìn thấy lớp đó quá thường xuyên trong quá trình huấn luyện.
2. Nếu giữ nguyên CE, gradient từ lớp đông sẽ áp đảo gradient từ lớp hiếm.
3. Class-balanced loss giảm ảnh hưởng của lớp đông và tăng ảnh hưởng tương đối của lớp hiếm.

Điều này đặc biệt hữu ích khi:

1. Một vài nhãn sentiment hoặc aspect xuất hiện ít hơn đáng kể.
2. Dữ liệu có nhiều `null` hoặc `absent`.
3. Ta muốn mô hình học công bằng hơn giữa các lớp thay vì chỉ tối ưu lớp phổ biến.

Trong notebook 05, class-balanced loss được dùng như một biến thể để xem việc gán trọng số theo độ hiếm có giúp PhoBERT học tốt hơn hay không. Kết quả cho thấy trọng số lớp giúp sentiment ổn hơn, nhưng nếu chỉ dùng riêng nó thì chưa đủ để xử lý triệt để phần aspect thưa.

## 3.17. Lý thuyết focal loss

Focal loss được thiết kế để buộc mô hình tập trung vào các mẫu khó thay vì quá dễ dãi với những mẫu đã dự đoán đúng gần như chắc chắn.

Nếu CE chuẩn xử lý mọi mẫu gần như tương đương, focal loss thêm một cơ chế "giảm trọng số" cho các mẫu dễ:

1. Mẫu nào mô hình đã đoán rất đúng thì đóng góp vào loss giảm xuống.
2. Mẫu nào mô hình còn nhầm lẫn thì đóng góp vào loss tăng tương đối.

Trực giác này rất quan trọng trong bài toán mất cân bằng:

1. Lớp đông thường có nhiều mẫu dễ.
2. Nếu cứ tối ưu CE, mô hình nhanh chóng học tốt các mẫu dễ đó.
3. Các mẫu hiếm và khó bị bỏ quên.

Focal loss khắc phục bằng cách làm cho quá trình học "khắt khe" hơn với các mẫu chưa chắc chắn.

Trong notebook 05, `clean_joint_balanced_focal` là experiment tốt nhất theo `val_f1_combined`. Điều này phù hợp với trực giác lý thuyết: focal loss không chỉ cân bằng theo số lượng, mà còn nhấn mạnh đúng các vị trí mà mô hình còn yếu, đặc biệt là aspect presence và các lớp neutral khó phân biệt.

## 3.18. Lý thuyết balanced sampler

Balanced sampler là chiến lược điều khiển cách lấy minibatch trong quá trình huấn luyện.

Thay vì để dữ liệu được đưa vào theo phân bố gốc, sampler sẽ cố gắng:

1. Tăng xác suất xuất hiện của lớp hiếm trong một batch.
2. Giảm tình trạng batch nào cũng đầy các mẫu từ lớp đông.
3. Làm cho tín hiệu gradient ổn định hơn giữa các lớp.

Về mặt thực nghiệm, balanced sampler thường hữu ích hơn chỉ sửa loss, vì loss chỉ tác động sau khi batch đã được tạo ra, còn sampler thay đổi luôn "mẫu mà mô hình nhìn thấy" trong từng bước cập nhật.

Điều này rất hợp với ABSA:

1. Aspect hiếm như `service` hoặc `packaging` cần được nhìn thấy đủ nhiều.
2. Một batch cân bằng hơn giúp mô hình học ranh giới lớp tốt hơn.
3. Khi kết hợp với focal loss, mô hình vừa thấy dữ liệu cân bằng hơn vừa tập trung vào mẫu khó hơn.

Trong notebook 05, các biến thể `joint balanced` cho thấy cải thiện rõ hơn so với CE thuần hoặc chỉ thêm class weight.

## 3.19. Lý thuyết supervised contrastive loss

Supervised contrastive loss là một dạng loss học biểu diễn, trong đó các mẫu cùng lớp được kéo gần nhau còn các mẫu khác lớp bị đẩy ra xa.

Khác với CE chỉ quan tâm "đúng hay sai nhãn", supervised contrastive loss còn quan tâm đến cấu trúc không gian biểu diễn:

1. Mẫu cùng lớp nên tạo thành một cụm chặt hơn.
2. Các lớp khác nhau nên tách biệt hơn.
3. Biểu diễn học được vì thế giàu thông tin hơn cho lớp phân loại phía sau.

Tác dụng của nó đặc biệt rõ khi:

1. Lớp có nhiều biến thể ngôn ngữ.
2. Các lớp gần nhau về nghĩa, ví dụ các aspect như `physical` và `packaging`.
3. Dữ liệu nhiễu khiến boundary giữa lớp bị mờ.

Trong notebook 05, supervised contrastive loss được thử như một thành phần bổ sung cho một số experiment neutral-aware. Mục tiêu là tăng separability của embedding, giúp model ít nhầm các lớp gần nhau hơn.

## 3.20. Lý thuyết điều chỉnh ngưỡng dự đoán

Nhiều bài toán phân loại không chỉ phụ thuộc vào xác suất đầu ra, mà còn phụ thuộc vào ngưỡng quyết định.

Trong ABSA hai tầng:

1. Tầng presence dự đoán aspect có xuất hiện hay không.
2. Nếu xác suất presence vượt ngưỡng thì mới chuyển sang tầng sentiment.

Điều này dẫn đến một vấn đề quan trọng: ngưỡng tốt cho `present-only` chưa chắc tốt cho `all-label`.

Khi ngưỡng quá thấp:

1. Mô hình dễ gán nhầm là có aspect.
2. Recall tăng nhưng precision giảm.

Khi ngưỡng quá cao:

1. Mô hình bỏ sót nhiều aspect thật.
2. Precision tăng nhưng recall giảm.

Vì vậy, calibration ngưỡng trên validation là bước thực tế rất cần thiết. Nó giúp tách riêng phần "mô hình học tốt" và phần "quyết định cuối cùng" để tối ưu đúng mục tiêu đánh giá.

## 3.21. Lý thuyết vì sao các kỹ thuật này đi cùng nhau

Các kỹ thuật ở notebook 05 không nên hiểu như những mẹo rời rạc. Chúng bổ sung cho nhau theo ba tầng:

1. `Balanced sampler` kiểm soát dữ liệu đầu vào mỗi batch.
2. `Class-balanced loss` và `focal loss` điều chỉnh cách mô hình phản ứng với từng mẫu.
3. `Threshold calibration` điều chỉnh cách biến xác suất thành nhãn cuối cùng.

Nếu chỉ dùng một kỹ thuật, mô hình thường chỉ cải thiện một phần:

1. Chỉ dùng weight có thể giúp lớp hiếm nhưng chưa đủ làm mô hình tập trung vào mẫu khó.
2. Chỉ dùng focal loss có thể giúp mẫu khó nhưng vẫn bị lệch vì batch quá mất cân bằng.
3. Chỉ calibrate threshold có thể cải thiện metric cuối nhưng không sửa được biểu diễn học bên trong.

Do đó, notebook 05 chọn cách phối hợp nhiều biện pháp để xử lý đúng bản chất của ABSA: vừa mất cân bằng, vừa nhiều nhãn thưa, vừa có hai tầng dự đoán, vừa phụ thuộc mạnh vào ngưỡng quyết định.

## 7. Thực nghiệm Generative ABSA với ViT5

## 7.1. Mục tiêu thực nghiệm

Notebook `notebooks/06_vit5.ipynb` chuyển bài toán ABSA sang hướng sinh chuỗi văn bản, thay vì chỉ phân loại từng nhãn rời rạc. Cách tiếp cận này thường được gọi là `Generative ABSA` hoặc `Text-to-Text ABSA`.

Ý tưởng cốt lõi là:

1. Đầu vào là văn bản review đã ghép `title` và `content`.
2. Đầu ra là một chuỗi mô tả toàn bộ nhãn cần dự đoán.
3. Mô hình học cách sinh ra chuỗi nhãn đó thay vì trả về vector nhãn cố định.

Cách làm này có hai lợi ích rõ ràng:

1. Nó thống nhất sentiment tổng thể và sentiment theo aspect vào cùng một khung sinh chuỗi.
2. Nó tận dụng tốt khả năng sinh ngôn ngữ của các mô hình T5-style như ViT5.

## 7.2. Lý thuyết Text-to-Text

Trong bài toán phân loại truyền thống, mỗi đầu ra là một lớp rời rạc. Trong bài toán text-to-text, cả đầu vào lẫn đầu ra đều là văn bản.

Với notebook này, chuỗi mục tiêu có dạng:

`Tổng thể: tích cực, Content: tiêu cực, Physical: trung tính, ...`

Điều này có nghĩa là mô hình không còn phải học riêng từng head phân loại, mà học ánh xạ từ câu review sang một chuỗi mô tả có cấu trúc.

Ưu điểm của hướng này:

1. Dễ mở rộng thêm nhãn mới mà không cần sửa kiến trúc head quá nhiều.
2. Có thể khai thác tri thức tiền huấn luyện của mô hình sinh chuỗi.
3. Phù hợp với các bài toán mà đầu ra có cấu trúc ngôn ngữ rõ ràng.

Nhược điểm:

1. Đầu ra phụ thuộc rất mạnh vào đúng định dạng chuỗi.
2. Nếu mô hình sinh sai dấu phẩy, dấu hai chấm hoặc sai từ khóa, quá trình parse sẽ hỏng hoặc mất nhãn.
3. Đánh giá phức tạp hơn vì phải chuyển ngược văn bản sinh ra thành nhãn số.

## 7.3. Cấu hình dữ liệu và nhãn

Notebook dùng các nhãn:

1. `0 = tiêu cực`
2. `1 = trung tính`
3. `2 = tích cực`
4. `3 = absent` cho aspect không xuất hiện

Cấu trúc đầu vào được ghép từ:

1. `review_title`
2. `content`

Phần prompt còn bổ sung một chỉ dẫn rõ hơn về lớp `neutral`, theo hướng coi `neutral` là câu trần thuật khách quan hoặc câu có khen chê cân bằng nhau. Đây là một cải tiến quan trọng vì nhiều mô hình sinh chuỗi thường thiên về `positive` nếu prompt quá mơ hồ.

Phần đầu ra được tạo bằng hàm `format_target_text`, trong đó:

1. Luôn có `Tổng thể`.
2. Chỉ thêm aspect nào thực sự có nhãn.
3. Aspect absent không bị ép xuất hiện trong chuỗi đích.

Cách biểu diễn này rất phù hợp với dữ liệu ABSA thưa, vì không bắt mô hình phải sinh những nhãn không cần thiết.

## 7.4. Cơ sở lý thuyết của ViT5

ViT5 là một biến thể T5 được tiền huấn luyện cho tiếng Việt. Về mặt kiến trúc, đây là mô hình encoder-decoder:

1. Encoder đọc toàn bộ input và tạo biểu diễn ngữ cảnh.
2. Decoder sinh chuỗi output từng token một.

Kiến trúc encoder-decoder phù hợp với Generative ABSA vì bài toán này không chỉ là phân loại mà là tái biểu diễn ý nghĩa của toàn bộ review thành một chuỗi nhãn có cấu trúc.

So với classifier thuần:

1. ViT5 linh hoạt hơn về mặt định dạng đầu ra.
2. Có thể tận dụng tốt pretraining language modeling.
3. Thích hợp khi muốn mô hình học một biểu diễn chung cho nhiều nhãn cùng lúc.

Tuy vậy, vì đây là mô hình sinh chuỗi, nên việc kiểm soát đầu ra khó hơn so với một head softmax cố định.

## 7.5. LoRA và fine-tuning hiệu quả tham số

Notebook không fine-tune toàn bộ ViT5, mà dùng LoRA.

LoRA là kỹ thuật chèn thêm các ma trận hạng thấp vào một số projection layer của transformer, thay vì cập nhật toàn bộ trọng số gốc. Ý nghĩa thực tế:

1. Giảm số tham số phải huấn luyện.
2. Giảm chi phí bộ nhớ.
3. Giúp fine-tuning nhanh và ổn định hơn.

Notebook cấu hình:

1. `MODEL_NAME = VietAI/vit5-base`
2. `r = 16`
3. `lora_alpha = 32`
4. `lora_dropout = 0.05`
5. `target_modules = q, k, v, o, wi_0, wi_1, wo`

Sau khi gắn LoRA, số tham số trainable chỉ còn khoảng `5,013,504` trên tổng `230,964,480`, tức xấp xỉ `2.17%`.

Đây là lợi thế lớn nếu mục tiêu là thử nghiệm nhanh một mô hình sinh chuỗi mạnh nhưng vẫn tiết kiệm tài nguyên.

## 7.6. Trọng số mẫu và custom loss

Notebook không dùng loss mặc định hoàn toàn, mà tự định nghĩa `WeightedSeq2SeqTrainer`.

Ý tưởng của phần này:

1. Tính loss token-level bằng Cross Entropy với `ignore_index=-100`.
2. Gom về loss theo từng sequence.
3. Nhân thêm trọng số mẫu nếu có.
4. Lấy trung bình trên batch.

Trọng số mẫu được suy ra từ phân bố `sentiment_llm` trên train. Mẫu thuộc lớp hiếm sẽ có trọng số cao hơn.

Điều này cho thấy notebook không chỉ chuyển sang mô hình sinh chuỗi, mà còn cố xử lý mất cân bằng ngay trong quá trình tối ưu.

Lý do phải làm vậy là vì generative model thường rất mạnh ở việc sinh chuỗi phổ biến, nhưng nếu dữ liệu lệch lớp thì nó vẫn có thể thiên lệch rõ rệt về các nhãn đông.

## 7.7. Cơ chế đánh giá

Đánh giá của notebook không dựa trên chuỗi sinh ra trực tiếp, mà dựa trên việc parse lại chuỗi đó thành nhãn số.

Quy trình:

1. Decode output của mô hình thành văn bản.
2. Tách chuỗi theo dấu phẩy.
3. Tách từng cặp `key: value`.
4. Ánh xạ lại `tiêu cực / trung tính / tích cực` thành `0 / 1 / 2`.
5. Tính F1 cho sentiment và aspect.

Điều này có nghĩa là chất lượng mô hình phụ thuộc không chỉ vào đúng nội dung, mà còn phụ thuộc vào đúng format.

Nhận xét học thuật quan trọng:

1. Đây là ưu điểm nếu muốn output có cấu trúc dễ đọc.
2. Đây cũng là nhược điểm vì một lỗi cú pháp nhỏ trong chuỗi sinh ra có thể làm mất thông tin khi chấm điểm.

## 7.8. Thiết lập huấn luyện

Notebook dùng các tham số chính:

1. `MAX_INPUT_LENGTH = 256`
2. `MAX_TARGET_LENGTH = 64`
3. `BATCH_SIZE = 8`
4. `EPOCHS = 8`
5. `LEARNING_RATE = 2e-4`

Trainer được cấu hình theo hướng seq2seq chuẩn:

1. Đánh giá theo từng epoch.
2. Lưu checkpoint theo từng epoch.
3. Dùng `predict_with_generate=True`.
4. Chọn mô hình tốt nhất theo `f1_sentiment`.
5. Bật `fp16` khi có CUDA.

Điểm đáng chú ý là learning rate của phần LoRA lớn hơn các notebook fine-tune đầy đủ trước đó, vì chỉ một phần nhỏ tham số được cập nhật.

## 7.9. Kết quả test

Kết quả từ trainer trên tập test:

| Chỉ số | Giá trị |
| --- | ---: |
| `test_loss` | 0.1196 |
| `test_f1_sentiment` | 0.8405 |
| `test_f1_aspect_present` | 0.7790 |
| `test_f1_final` | 0.8098 |

Báo cáo sentiment tổng thể trên test:

| Lớp | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.95 | 0.92 | 0.93 |
| Neutral | 0.66 | 0.70 | 0.68 |
| Positive | 0.90 | 0.92 | 0.91 |

Báo cáo 6 aspect tổng thể:

| Lớp | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.90 | 0.76 | 0.83 |
| Neutral | 0.71 | 0.52 | 0.60 |
| Positive | 0.94 | 0.88 | 0.91 |

F1 theo từng aspect:

| Khía cạnh | F1 |
| --- | ---: |
| `as_content` | 0.78 |
| `as_physical` | 0.70 |
| `as_price` | 0.57 |
| `as_packaging` | 0.57 |
| `as_delivery` | 0.80 |
| `as_service` | 0.46 |

Các con số này cho thấy mô hình làm khá tốt ở sentiment tổng thể và một số aspect phổ biến như `delivery`, nhưng yếu hơn rõ rệt ở các aspect hiếm như `service`.

## 7.10. Nhận xét và hạn chế

Notebook 06 cho thấy Generative ABSA là một hướng khả thi, nhưng không phải hướng dễ thay thế hoàn toàn các classifier trước đó.

Ưu điểm:

1. Mô hình hóa bài toán theo đúng kiểu sinh chuỗi có cấu trúc.
2. Linh hoạt hơn khi cần mở rộng định dạng đầu ra.
3. LoRA giúp fine-tuning rẻ hơn rất nhiều về tham số.

Hạn chế:

1. Phụ thuộc mạnh vào format của chuỗi sinh ra.
2. Không ổn định bằng các mô hình phân loại chuyên biệt nếu đầu ra bị lệch cấu trúc.
3. Các aspect hiếm vẫn là nút thắt, đặc biệt `service`.

Về mặt thực nghiệm, notebook này phù hợp để xem Generative ABSA như một nhánh nghiên cứu bổ sung, không phải lúc nào cũng vượt hẳn classifier hai tầng. Giá trị lớn nhất của nó là chứng minh rằng bài toán ABSA có thể được mô tả lại thành text-to-text một cách nhất quán và có thể huấn luyện hiệu quả bằng LoRA.
