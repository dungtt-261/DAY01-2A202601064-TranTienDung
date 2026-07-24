# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00  
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng.

---

## Block 1 — API Cơ Bản

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?**
> Khi đặt `temperature=0.0`, model thường chọn cách trả lời an toàn, ngắn và gần như không thay đổi nhiều giữa các lần thử. Từ `0.7` đến `1.2`, nội dung phong phú hơn và cách kể hấp dẫn hơn, nhưng độ ổn định giảm; ở `1.8`, một số câu có thể dài dòng hoặc chứa liên tưởng thiếu liên kết. Tôi thấy mức cao nhất bắt đầu không phù hợp với câu hỏi yêu cầu thông tin chính xác.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Với hợp đồng pháp lý, tôi ưu tiên khoảng `0.1` vì đầu ra cần nhất quán, ít biến thể và bám sát cấu trúc được quy định. Với slogan quảng cáo, tôi có thể dùng khoảng `1.0` để model thử nhiều cách chơi chữ và ý tưởng mới. Hai mức khác nhau vì một tác vụ đặt độ tin cậy lên trước, còn tác vụ kia cần mở rộng không gian sáng tạo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ:**
> Workload tạo ra `20.000 × 2 × 500 = 20.000.000` output token mỗi ngày. Theo bảng giá trong bài, GPT-4o có chi phí khoảng `20.000 × 0,010 = 200 USD/ngày`, trong khi GPT-4o-mini khoảng `20.000 × 0,0006 = 12 USD/ngày`, chênh lệch khoảng `188 USD`. Model lớn hợp lý cho phân tích nhiều bước, nội dung có ảnh hưởng lớn hoặc trường hợp cần chất lượng lập luận cao; model nhỏ phù hợp với hỏi đáp phổ biến, trích xuất trường dữ liệu và các request có quy mô lớn.

---

## Block 2 — System Prompt & Token

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng dùng persona khác nhau.

**Hai phản hồi khác nhau như thế nào?**
> Với persona nhà thơ, phần giải thích thiên về hình ảnh ví von, nhịp câu mềm và gần như không dùng từ chuyên môn. Với persona kỹ sư senior, câu trả lời có định nghĩa rõ, thuật ngữ chính xác, cấu trúc theo ý và có thể thêm ví dụ bằng code hoặc dữ liệu. Điều này cho thấy system prompt có thể định hướng vai trò, giọng điệu, đối tượng người đọc, mức kỹ thuật, độ dài và kiểu trình bày. Dù vậy, persona chỉ điều khiển cách phản hồi chứ không tự bảo đảm rằng mọi thông tin đều đúng.

### Câu 2.2 — tiktoken vs đếm từ
**So sánh `count_tokens` với ước lượng `số từ / 0.75`:**
> Tôi thử với một đoạn tiếng Việt khoảng 150 từ. Giả sử tokenizer trả về 184 token, còn phép ước lượng theo số từ cho `150 / 0,75 = 200` token, thì chênh lệch là 16 token, tương đương khoảng 8% so với kết quả ước lượng. Trong ví dụ này, cách đếm thô dự toán cao hơn thực tế, nhưng kết quả có thể đảo chiều với văn bản chứa nhiều dấu câu, từ tiếng Anh, mã nguồn hoặc tên riêng; vì vậy nên đo token trên dữ liệu đại diện của sản phẩm.

---

## Block 3 — Streaming & Độ Bền

### Câu 3.1 — Trải nghiệm người dùng với streaming
> Trợ lý giọng nói và chatbot văn bản đều hưởng lợi từ streaming vì người dùng nhận phần đầu của câu trả lời sớm hơn, làm giảm thời gian chờ cảm nhận. Với giọng nói, lợi ích đặc biệt rõ vì hệ thống có thể bắt đầu tổng hợp âm thanh ngay khi có vài câu đầu. Ngược lại, pipeline dịch tài liệu chạy vào ban đêm không cần hiển thị từng token; nhận kết quả hoàn chỉnh theo lô sẽ dễ lưu trữ, kiểm tra và chạy lại khi gặp lỗi.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
> Delay cố định khiến nhiều client dễ gửi lại request theo cùng một nhịp, làm dịch vụ tiếp tục bị dồn tải. Exponential backoff kéo giãn dần khoảng cách giữa các lần thử, nên số request giảm khi lỗi kéo dài và API có thêm thời gian phục hồi. Jitter bổ sung một độ lệch ngẫu nhiên để các client không thức dậy đồng thời ở những mốc giống nhau, qua đó giảm hiện tượng một đợt retry lớn xuất hiện lặp lại.

---

## Block 4 — Mini-Project

### Câu 4.1 — Thiết kế persona
> System prompt: “Bạn là trợ giảng thực hành AI. Trả lời bằng tiếng Việt, ưu tiên câu ngắn và hướng dẫn theo trình tự có thể thực hiện ngay. Khi có code, hãy đưa ví dụ tối thiểu và giải thích phần quan trọng. Không tự suy đoán dữ kiện còn thiếu; hãy nói rõ điểm chưa chắc chắn và yêu cầu người dùng bổ sung thông tin.” Nếu bỏ yêu cầu “hướng dẫn theo trình tự có thể thực hiện ngay”, câu trả lời có thể chuyển thành mô tả lý thuyết chung thay vì các bước thao tác. Nếu bỏ quy tắc “không tự suy đoán dữ kiện”, trợ lý có khả năng điền giả định và đưa ra hướng xử lý không phù hợp với môi trường thật.

### Câu 4.2 — Hạn chế & cải thiện
> Một tình huống cụ thể là người dùng nêu ở đầu phiên rằng hệ thống phải chạy trên Windows và không được dùng Docker, sau đó trao đổi nhiều lượt về API và giao diện. Khi câu hỏi cuối yêu cầu viết lệnh triển khai, hai ràng buộc ban đầu có thể đã nằm ngoài bốn lượt được giữ và trợ lý lại đề xuất Docker. Có thể khắc phục bằng cách tách “thông tin bền vững của phiên” thành một bản tóm tắt hoặc danh sách ràng buộc, cập nhật sau mỗi lượt và luôn đưa phần này vào context bên cạnh tám message gần nhất.

---
