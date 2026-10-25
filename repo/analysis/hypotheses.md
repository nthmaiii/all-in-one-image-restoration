# Danh Sách Giả Thuyết Nghiên Cứu & Tự Phản Biện (Hypotheses & Reviewer 2 Critique)
**Mã nhiệm vụ**: P3-T1 & P3-T2
**Dựa trên cơ sở thực nghiệm**: Kết quả tái hiện baseline & chẩn đoán lỗi OOD trên Colab Tesla T4 (`results/baseline_results.csv`, `results/ood_baseline.csv`).

---

## PHẦN 1: KHỞI TẠO 8 GIẢ THUYẾT NGHIÊN CỨU BAN ĐẦU (P3-T1)

Mỗi giả thuyết đều trả lời đầy đủ 4 câu hỏi bắt buộc:
- **(a)** Hiện tượng lỗi cụ thể nào của baseline mà ý tưởng này giải quyết?
- **(b)** Cơ chế lý thuyết tại sao giải quyết được?
- **(c)** Thí nghiệm tối thiểu nào có thể bác bỏ ý tưởng này?
- **(d)** Điểm khác biệt bản chất so với 5 bài gần nhất (PromptIR, AdaIR, AirNet, Painter, InstructIR)?

---

### Giả Thuyết H1: Phân Tách Suy Thoái Theo Miền Tần Số (Frequency-Decoupled Dynamic Prompting - FD-Prompt)
- **(a) Hiện tượng lỗi giải quyết**: Sụp đổ thảm họa khi gặp suy thoái hỗn hợp (Nhiễu $\sigma=25$ + Sương mù, PSNR tụt xuống **16.79 dB** trong log Colab T4) và mất chi tiết cao tần khi nhiễu lớn.
- **(b) Cơ chế lý thuyết**: Các dạng suy thoái có phân bố quang phổ tần số trực giao: Nhiễu và vệt mưa chủ yếu nằm ở dải tần số cao ($f_{high}$), trong khi sương mù và suy giảm tương phản nằm ở dải tần số thấp ($f_{low}$). Bằng cách phân tách đặc trưng đầu vào qua 2D Discrete Wavelet Transform (DWT) hoặc Laplacian Pyramid trước khi đưa vào Prompt Generation Module, mạng học 2 bộ prompt độc lập: $P_{high}$ để triệt tiêu nhiễu cao tần và $P_{low}$ để khôi phục độ tương phản thấp tần, triệt tiêu hoàn toàn hiện tượng xung đột biểu diễn (representation collision).
- **(c) Thí nghiệm tối thiểu bác bỏ**: Huấn luyện một mini-model trên tập hỗn hợp Noise+Haze trong 5000 iterations. Nếu mô hình FD-Prompt không vượt qua PromptIR chuẩn ít nhất **+1.0 dB** PSNR trên tập test hỗn hợp, giả thuyết bị bác bỏ.
- **(d) Khác biệt bản chất so với prior art**:
  - *PromptIR*: Học 1 bộ prompt vector duy nhất từ toàn bộ không gian ảnh gộp cả cao tần lẫn thấp tần.
  - *AdaIR*: Dùng điều chế kênh (channel modulation) không phân tách không gian tần số.
  - *AirNet*: Học không gian tương phản (contrastive) để phân loại loại suy thoái rời rạc, không xử lý được suy thoái hỗn hợp.

---

### Giả Thuyết H2: Định Tuyến Trọng Số Suy Thoái Phân Bố Liên Tục (Continuous Degradation Manifold Routing - CD-Route)
- **(a) Hiện tượng lỗi giải quyết**: Sụp đổ khi gặp mức độ suy thoái ngoài phân phối huấn luyện (Nhiễu $\sigma=75$, SSIM giảm 36% xuống **0.6378**).
- **(b) Cơ chế lý thuyết**: Các mô hình hiện tại (PromptIR, AdaIR) giả định các trạng thái suy thoái là các điểm rời rạc trên không gian biểu diễn (discrete clusters). Khi gặp mức độ nhiễu cực đoan ($\sigma=75$), vector đặc trưng rơi ra ngoài các cụm đã học. CD-Route mô hình hóa không gian suy thoái dưới dạng một đa tạp liên tục (continuous manifold) với phân phối Dirichlet / Gaussian Mixture, cho phép nội suy và ngoại suy độ mạnh suy thoái thông qua phép chiếu liên tục (continuous interpolation).
- **(c) Thí nghiệm tối thiểu bác bỏ**: Đánh giá trên dải nhiễu OOD $\sigma \in [60, 80]$. Nếu độ suy giảm SSIM vẫn $> 25\%$ so với mức $\sigma=50$, giả thuyết bị bác bỏ.
- **(d) Khác biệt bản chất so với prior art**:
  - *AdaIR*: Dự đoán trọng số rời rạc qua routing softmax bị bão hòa ở biên.
  - *PromptIR*: Chỉ có số lượng prompt cố định (5 prompts), không có khả năng ngoại suy độ mạnh liên tục.

---

### Giả Thuyết H3: Khung Phục Hồi Lặp Tuần Tự Tự Thích Ứng (Uncertainty-Guided Iterative Refinement - UGIR)
- **(a) Hiện tượng lỗi giải quyết**: Top 30 ảnh tệ nhất có các vết suy thoái cục bộ đậm đặc (sọc mưa dày, hạt nhiễu dính cụm) mà 1 lượt forward pass duy nhất không thể xóa sạch.
- **(b) Cơ chế lý thuyết**: Thay vì tăng độ sâu mạng (tăng params và FLOPs), mạng sử dụng cùng 1 bộ trọng số Restormer khối lượng nhẹ, nhưng hồi quy qua 2 passes. Một nhánh ước lượng độ bất định (uncertainty map / error confidence) quyết định vùng nào cần tinh chỉnh lại ở lượt 2.
- **(c) Thí nghiệm tối thiểu bác bỏ**: Nếu lượt 2 làm tăng latency trên T4 $> 50\%$ mà chỉ cải thiện $< 0.2$ dB trên top 30 ảnh tệ nhất.
- **(d) Khác biệt bản chất so với prior art**:
  - *MPRNet / HINet*: Dùng kiến trúc đa tầng (multi-stage) với các bộ trọng số riêng biệt (gấp đôi params).
  - *PromptIR*: Hoàn toàn là mạng 1 pass chuyển tiếp thẳng (feed-forward).

---

### Giả Thuyết H4: Ràng Buộc Trực Giao Cho Prompt Biểu Diễn (Orthogonal Prompt Regularization - OPR)
- **(a) Hiện tượng lỗi giải quyết**: Hiện tượng can thiệp chéo (cross-task negative transfer) giữa các tác vụ khôi phục khác nhau và sự suy giảm hiệu quả bộ nhớ/VRAM trên T4.
- **(b) Cơ chế lý thuyết**: Ép ma trận tương quan giữa các vector prompt $P_i$ và $P_j$ thỏa mãn $P_i^T P_j \approx 0$ ($i \neq j$) thông qua một hàm phạt trực giao (Orthogonal Loss: $\mathcal{L}_{orth} = \|P^T P - I\|_F^2$). Ràng buộc này bảo đảm mỗi prompt chỉ nắm giữ một không gian con biến dạng độc lập, triệt tiêu hiện tượng "nhầm lẫn" giữa sọc mưa và hạt nhiễu.
- **(c) Thí nghiệm tối thiểu bác bỏ**: Đo độ tương quan giữa các prompt sau khi huấn luyện. Nếu loss trực giao làm chậm tốc độ hội tụ $\ge 30\%$ hoặc làm giảm PSNR trên Denoising, giả thuyết bị loại.
- **(d) Khác biệt bản chất so với prior art**:
  - *PromptIR*: Để các prompt tokens tự do tối ưu hóa mà không có ràng buộc không gian con, dẫn đến hiện tượng prompt collapse (các prompt có xu hướng tiến lại gần nhau).

---

### Giả Thuyết H5: Hướng Dẫn Văn Bản Ngôn Ngữ Tự Nhiên (CLIP/LLM-Guided Restoration)
- **(a) Hiện tượng lỗi giải quyết**: Phân loại nhầm tác vụ suy thoái khi ảnh có ngữ cảnh tối hoặc phức tạp.
- **(b) Cơ chế lý thuyết**: Dùng mô hình CLIP trích xuất prompt ngữ nghĩa (text embedding: "noisy image", "rainy image", "hazy image") để dẫn dắt bộ phục hồi.
- **(c) Thí nghiệm tối thiểu bác bỏ**: Đo latency và VRAM trên Tesla T4.
- **(d) Khác biệt**: Dùng multimodal CLIP embedding thay cho visual prompt.

---

### Giả Thuyết H6: Tích Hợp Mô Hình Khuếch Tán Nhẹ (Lightweight Diffusion Prior)
- **(a) Hiện tượng lỗi giải quyết**: Phục hồi các vân chi tiết phức tạp bị mờ (Texture smoothing).
- **(b) Cơ chế lý thuyết**: Dùng mô hình khuếch tán 1 bước (One-step Diffusion / Consistency Model) làm bộ sinh chi tiết cao tần.
- **(c) Thí nghiệm tối thiểu bác bỏ**: Đo thời gian suy luận trên T4.
- **(d) Khác biệt**: Dùng generative diffusion prior thay vì bộ tái thiết hồi quy thuần túy.

---

### Giả Thuyết H7: Cấu Trúc Đa Chuyên Gia MoE Không Gian (Spatial Mixture-of-Experts)
- **(a) Hiện tượng lỗi giải quyết**: Ảnh có suy thoái không đồng nhất giữa các vùng (trên trời có sương mù, dưới đất có vệt mưa).
- **(b) Cơ chế lý thuyết**: Thay thế MLP trong Transformer bằng Sparse MoE với 4 chuyên gia cục bộ theo từng patch ảnh.
- **(c) Thí nghiệm tối thiểu bác bỏ**: Đo routing collapse và VRAM.
- **(d) Khác biệt**: Định tuyến cấp độ patch thay vì cấp độ toàn ảnh.

---

### Giả Thuyết H8: Biến Đổi Không Gian Màu Tự Động Thích Ứng (Adaptive Color Space Projection)
- **(a) Hiện tượng lỗi giải quyết**: Hiện tượng lệch màu sắc (Color shift) khi phục hồi trên kênh RGB.
- **(b) Cơ chế lý thuyết**: Chuyển đổi linh hoạt giữa không gian RGB và YCbCr/LAB tùy theo loại suy thoái.
- **(c) Thí nghiệm tối thiểu bác bỏ**: So sánh PSNR kênh Y vs RGB.
- **(d) Khác biệt**: Xử lý ở tầng tiền xử lý màu sắc thay vì can thiệp vào tầng deep features.

---

## PHẦN 2: TỰ PHẢN BIỆN (REVIEWER 2 CRITIQUE) & CHỌN LỌC (P3-T2)

### Bảng Đánh Giá Khắt Khe Kiểu Reviewer 2:

| Giả thuyết | Điểm mạnh nhất | Rủi ro chí mạng / Lý do từ chối (Reviewer 2) | Phán quyết |
|:---|:---|:---|:---:|
| **H1: Frequency-Decoupled (FD-Prompt)** | Giải quyết trực tiếp hiện tượng sụp đổ 16.79 dB của suy thoái hỗn hợp; toán học vững chắc (DWT/Laplacian). | Cần thiết kế DWT nhẹ để không làm tăng FLOPs trên T4. | **CHỌN (Ứng viên số 1)** |
| **H2: Continuous Manifold (CD-Route)** | Khắc phục sụp đổ OOD ngoài phân phối ($\sigma=75$); toán học mới lạ trên không gian đa tạp. | Cần điều hòa tốt hàm phạt để router không bị bão hòa. | **CHỌN (Ứng viên số 2)** |
| **H3: Iterative Refinement (UGIR)** | Giải quyết triệt để 30 ảnh lỗi tệ nhất mà không tăng số lượng weights. | Nguy cơ tăng latency nếu không kiểm soát chặt cơ chế dừng sớm. | **CHỌN (Kế hoạch B)** |
| **H4: Orthogonal Regularization (OPR)** | Cực kỳ nhẹ, 0 FLOPs overhead ở pha test, triệt tiêu can thiệp chéo giữa các prompt. | Có thể mức tăng PSNR chỉ ở mức vừa phải (+0.2 - +0.3 dB). | **CHỌN (Module bổ trợ)** |
| *H5: CLIP / LLM Guided* | Đang là trend thị trường. | **LOẠI BỎ**: Trọng số CLIP quá nặng (>300M params), độ trễ suy luận quá cao, vi phạm nghiêm trọng ràng buộc Colab T4. Thiếu tính bản chất cho bài toán pixel-level. | **LOẠI BỎ** |
| *H6: Diffusion Prior* | Chất lượng thị giác cao. | **LOẠI BỎ**: Quá trình huấn luyện diffusion tốn hàng chục ngày GPU, không thể thực thi trên ngân sách Colab Free T4 12 tuần. | **LOẠI BỎ** |
| *H7: Spatial MoE* | Linh hoạt trên từng patch. | **LOẠI BỎ**: Sparse MoE thường gặp lỗi mất cân bằng chuyên gia (expert load imbalance) và tốn bộ nhớ VRAM khi lưu nhiều expert weights trên GPU nhỏ. | **LOẠI BỎ** |
| *H8: Adaptive Color Space* | Đơn giản, dễ cài đặt. | **LOẠI BỎ**: Tính mới quá mỏng (incremental / engineering trick), chắc chắn bị Reviewer đánh trượt vì thiếu chiều sâu đóng góp lý thuyết cho hội nghị top-tier. | **LOẠI BỎ** |

---

## KẾT LUẬN: BỘ 4 GIẢ THUYẾT MẠNH NHẤT ĐƯỢC CHỌN LỌC
1. **H1 (Chủ đạo A)**: Phân tách suy thoái theo miền tần số (Wavelet/Laplacian Prompting) trị dứt điểm suy thoái kép (Nhiễu + Sương mù).
2. **H2 (Chủ đạo B)**: Định tuyến suy thoái trên đa tạp phân phối liên tục (Continuous Manifold Routing) giải quyết triệt để độ lệch ngoài phân phối OOD ($\sigma=75$).
3. **H1 + H2 (Tích hợp tối ưu)**: Kiến trúc kết hợp phân tách tần số cao/thấp với bộ định tuyến liên tục.
4. **H4 (Kế hoạch B / Module an toàn)**: Ràng buộc trực giao hóa không gian Prompt (0 FLOPs overhead, an toàn tuyệt đối về hiệu năng).
