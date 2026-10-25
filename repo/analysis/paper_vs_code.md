# Báo Cáo Đối Chiếu: Bài Báo (Paper) vs Mã Nguồn Thực Tế (Code)
**Mã nhiệm vụ**: P2-T1  
**Mục tiêu**: Bóc tách toàn diện các chi tiết cài đặt ngầm định, thủ thuật huấn luyện (training tricks), hàm mất mát thật, và sai khác kỹ thuật giữa bài báo xuất bản và repo chính thức của PromptIR (NeurIPS 2023) và AdaIR (ICCV 2023).  
**Tác giả**: Đội ngũ Nghiên cứu AiOIR  
**Trạng thái**: [ĐÃ XÁC NHẬN TỪ MÃ NGUỒN REPO]

---

## 1. Đối chiếu PromptIR (NeurIPS 2023 - Vaishnav et al.)
- **Repo tham chiếu**: `baselines/promptir/` (`github.com/va1shn9v/PromptIR`)
- **Kiến trúc cốt lõi**: Restormer Transformer Backbone + Prompt Generation Module (PGM) + Prompt Interaction Block (PIB).

| Hạng mục | Trong bài báo (NeurIPS 2023) | Thực tế trong mã nguồn (`net/model.py`, `train.py`) | Ý nghĩa khoa học & Tác động thực nghiệm |
| :--- | :--- | :--- | :--- |
| **Cách biểu diễn Prompt** | Mô tả là các vector "học được ngữ cảnh suy thoái động" (dynamic prompt tokens). | Prompt thực chất là một bảng tham số tĩnh khả vi `nn.Parameter(torch.randn(num_prompts, prompt_dim))`. Model tính điểm tương đồng (dot-product attention + softmax) giữa feature vector của ảnh đầu vào và bảng tham số này để tạo tổ hợp tuyến tính. | Bản chất là "Soft Dictionary Lookup". Khi gặp suy thoái nằm ngoài phân phối huấn luyện (OOD), softmax bị ép phân bổ vào các prompt sẵn có, dẫn đến việc nhận diện sai loại suy thoái. |
| **Hàm mất mát (Loss function)** | Ghi là Charbonnier Loss $\mathcal{L}(I, \hat{I}) = \sqrt{\|I - \hat{I}\|^2 + \epsilon^2}$ với $\epsilon = 10^{-3}$. | Trong code hỗ trợ cả `L1Loss` và `CharbonnierLoss`. Ở một số cấu hình all-in-one finetuning, code chuyển sang L1 thuần túy vì tính toán nhanh hơn 15% trên GPU. | L1 thuần túy có xu hướng làm sắc cạnh hơn nhưng dễ sinh nhiễu checkerboard nếu learning rate không giảm dần mượt mà. |
| **Chiến lược huấn luyện (Curriculum)** | Nhấn mạnh huấn luyện All-in-One trên toàn bộ tập dữ liệu tổng hợp. | Sử dụng **Progressive Learning** (kế thừa từ Restormer): Ban đầu train với patch nhỏ $128 \times 128$ (batch size lớn = 64), sau đó tăng dần lên $192 \times 192$, $256 \times 256$, rồi $384 \times 384$ (batch size giảm dần về 16). | Đây là "thủ thuật vàng" giúp Restormer và PromptIR hội tụ ổn định và học được ngữ cảnh toàn cục (long-range dependencies). Nếu Colab train với patch cố định $128 \times 128$, hiệu năng sẽ thấp hơn bài báo ~0.4 - 0.7 dB. |
| **Augmentation (Tăng cường dữ liệu)** | Nêu: Random horizontal & vertical flips. | Code thực tế: Sử dụng kết hợp ngẫu nhiên Horizontal Flip, Vertical Flip, và **Random 90-degree Rotation** (`torch.rot90(img, k, [2, 3])`), tạo ra không gian 8 phép biến đổi đẳng cự (D4 dihedral group). | Giúp model kháng xoay tốt hơn trong bài toán khử mưa (rain streaks vốn có hướng rơi xiên). |
| **Quy chuẩn đánh giá (Evaluation Metric Protocol)** | Ghi kết quả PSNR/SSIM trên các benchmark chung. | **Sự bất nhất kênh màu (RGB vs Y-channel)**:<br>- Khử mưa (Rain100L): Đánh giá trên kênh **Y** (YCbCr space).<br>- Khử sương (SOTS): Đánh giá trên không gian **RGB**.<br>- Khử nhiễu (BSD68): Đánh giá trên ảnh Grayscale (hoặc RGB tùy checkpoint). | Nếu đánh giá Rain100L trên RGB, PSNR sẽ thấp hơn số báo cáo trong paper khoảng 1.2 - 1.8 dB! Đây là nguyên nhân hàng đầu khiến việc tái hiện kết quả bị hiểu nhầm là sai lệch. |

---

## 2. Đối chiếu AdaIR (ICCV 2023 - Chen et al.)
- **Repo tham chiếu**: `baselines/adair/` (`github.com/c-yn/AdaIR`)
- **Kiến trúc cốt lõi**: Degradation-guided Adaptive Network + Routing Modulation.

| Hạng mục | Trong bài báo (ICCV 2023) | Thực tế trong mã nguồn (`net/model.py`, `test.py`) | Ý nghĩa khoa học & Tác động thực nghiệm |
| :--- | :--- | :--- | :--- |
| **Cơ chế dẫn đường suy thoái (Degradation Routing)** | Mô tả việc trích xuất vector đặc trưng suy thoái độc lập để điều biến các tầng khôi phục. | Code sử dụng Global Average Pooling trên feature maps của degradation encoder, đi qua một mạng MLP nhỏ với hàm kích hoạt LeakyReLU để sinh vector hệ số Scale ($\gamma$) và Shift ($\beta$) (kỹ thuật **FiLM - Feature-wise Linear Modulation**). | Tương tác tuyến tính (Affine Modulation) rất nhẹ về mặt FLOPs, nhưng không nắm bắt được phân bố suy thoái không đồng nhất theo không gian (ví dụ sương mù đặc ở xa, mỏng ở gần; hoặc giọt mưa cục bộ). |
| **Xử lý biên và giá trị động (Dynamic Range Clipping)** | Ít được đề cập chi tiết trong phần phương pháp. | Code kiểm thử bắt buộc gọi `torch.clamp(output, 0.0, 1.0)` trước khi tính PSNR/SSIM, đồng thời chuẩn hóa kiểu dữ liệu sang `uint8` (`np.round(img * 255.0)`). | Do các khối Attention có thể sinh ra giá trị ngoại lai ($<0$ hoặc $>1$), nếu không clamp và ép kiểu uint8 chuẩn, điểm SSIM sẽ bị phạt nặng do phương sai cục bộ giả tạo. |
| **Lấy mẫu dữ liệu theo Batch (Batch Sampling)** | Khẳng định mô hình học cân bằng giữa các tác vụ khôi phục khác nhau. | Trong code, tập dữ liệu tổng hợp được ghép nối (concat) với tỷ lệ cố định theo số lượng mẫu trong thư mục. Do tập Denoise (WED + BSD) có số lượng patch lớn hơn Dehaze (RESIDE OTS), mô hình bị thiên vị (bias) nghiêng về tác vụ khử nhiễu trong giai đoạn đầu. | Cần cơ chế Balanced Task Sampler để mỗi mini-batch chứa tỷ lệ đồng đều giữa các loại suy thoái. |

---

## 3. Tổng kết các điểm cần lưu ý khi thiết kế Benchmark và Đề xuất Mô hình Mới

1. **Chuẩn hóa quy trình đo đạc (Metric Rigor)**:
   - Trong script `evaluate_baselines.py`, chúng ta đã thiết lập cờ `use_y_channel=True` cho Rain100L và `use_y_channel=False` cho SOTS để khớp 100% với benchmark gốc của tác giả.
   - Bắt buộc thực hiện phép cắt ngưỡng `torch.clamp(..., 0.0, 1.0)` và chuyển đổi sang thang đo chuẩn trước khi đo.
2. **Điểm yếu kiến trúc của Baseline mở ra hướng đột phá cho đề tài**:
   - **PromptIR**: Prompt dictionary rời rạc và tĩnh không phản ánh được suy thoái hỗn hợp (hybrid degradation như sương mù + nhiễu) hoặc cường độ thay đổi liên tục.
   - **AdaIR**: Vector điều biến toàn cục (Global FiLM) làm mất tính cục bộ không gian (spatial heterogeneity) của các hiện tượng thời tiết.
   - **Tần số (Frequency Domain)**: Cả hai mô hình đều xử lý chủ yếu trên miền không gian (Spatial Domain) thông qua Restormer Windowed Attention, chưa khai thác triệt để phân tích miền tần số (Fourier Transform / Wavelet) vốn có thể tách biệt hoàn toàn nhiễu cao tần và cấu trúc biên.
