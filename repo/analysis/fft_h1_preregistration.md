# ĐĂNG KÝ TRƯỚC (PRE-REGISTRATION) CHO PHÂN TÍCH PHỔ FOURIER (FFT) & KIỂM ĐỊNH GIẢ THUYẾT H1

> **Mã hồ sơ**: PRE-REG-H1-FFT-2026-V2  
> **Trạng thái**: KHÓA CỐ ĐỊNH TRƯỚC THỰC NGHIỆM (FROZEN PRE-REGISTRATION)  
> **Cam kết khoa học**: Văn bản này đăng ký trước định lượng ranh giới tần số, kỳ vọng lý thuyết bằng số, và các tiêu chí bác bỏ (falsification criteria) dứt khoát đối với Giả thuyết H1 (FD-PromptIR) trước khi thực hiện bất kỳ lượt huấn luyện chính thức nào trên GPU Colab T4. Mọi kết luận sau này bắt buộc phải đối chiếu với các tiêu chí định lượng đã đăng ký tại đây.

---

## 1. KHUNG LÝ THUYẾT VÀ GIẢ THUYẾT KHOA HỌC H1

### 1.1. Phát biểu Giả thuyết H1
Trong bài toán Khôi phục Ảnh Đa năng (All-in-One Image Restoration - AiOIR), khi mô hình đối mặt với **suy thoái hỗn hợp phức hợp (Hybrid Degradations)** đồng thời (ví dụ: vừa có sương mù Koschmieder, vừa có nhiễu Gaussian):
1. **Xung đột miền tần số (Frequency Conflict)**: Suy thoái sương mù làm suy giảm độ tương phản toàn cục và ánh sáng nền tập trung ở miền **tần số thấp (Low Frequency)**; trong khi nhiễu Gaussian và vệt mưa tập trung năng lượng nhiễu loạn ở miền **tần số cao (High Frequency)**.
2. **Nút thắt của PromptIR tiêu chuẩn (Nhánh A)**: Cơ chế prompt đơn khối (monolithic prompt learning) cố gắng tối ưu hóa một vector ngữ cảnh chung, dẫn đến sự triệt tiêu hoặc mâu thuẫn tín hiệu gradient: việc tăng cường độ tương phản tần số thấp dễ làm khuếch đại nhiễu tần số cao, và ngược lại, việc làm mịn nhiễu tần số cao dễ làm mờ biên cạnh và mất cân bằng sáng tối ở tần số thấp.
3. **Hiện tượng Bùng nổ Lỗi Đa cực (Cross-band Error Surge)**: Dẫn đến phổ năng lượng sai số tái tạo của PromptIR trên ảnh Hybrid sẽ bộc lộ hình thái phân bố hai cực (bimodal distribution) với sai số tăng vọt ở cả hai đầu dải tần.
4. **Giải pháp Đề xuất (FD-PromptIR - Nhánh B)**: Tách luồng biểu diễn prompt thành 2 nhánh chuyên biệt (Tần số thấp qua Spatial Average Pooling + 1x1 Conv; Tần số cao qua Residual High-pass Filter) giúp giải tỏa xung đột gradient, triệt tiêu phân bố hai cực của sai số và vượt trội so với baseline.

---

## 2. ĐỊNH NGHĨA ĐỊNH LƯỢNG CÁC DẢI TẦN SỐ (FREQUENCY BANDS)

Sử dụng Biến đổi Fourier Rời rạc 2 chiều (2D Discrete Fourier Transform - DFT) đã dịch tâm (shifted to center):
Cho ảnh sai số tái tạo $E = \hat{Y} - Y \in \mathbb{R}^{H \times W \times 3}$, chuyển đổi sang ảnh xám và tính phổ biên độ 2D:
$$\mathcal{F}(E)(u, v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} E(x, y) e^{-j 2\pi \left(\frac{ux}{H} + \frac{vy}{W}\right)}$$

Tọa độ tần số chuẩn hóa tương đối so với tần số cắt Nyquist ($f_N = 0.5 \times f_s$):
$$u_{\text{norm}} = \frac{u - H/2}{H/2} \in [-1, 1], \quad v_{\text{norm}} = \frac{v - W/2}{W/2} \in [-1, 1]$$
Bán kính tần số xuyên tâm chuẩn hóa (Normalized Radial Frequency) $\rho \in [0, 1]$:
$$\rho = \frac{\sqrt{u_{\text{norm}}^2 + v_{\text{norm}}^2}}{\sqrt{2}}$$

### 2.1. Phân chia 3 Dải tần số định lượng:
- **Dải tần số thấp (Low-Frequency Band - $B_{\text{low}}$)**:
  $$\rho \le 0.10 \quad (\le 10\% \text{ tần số Nyquist})$$
  *Ý nghĩa vật lý*: Đại diện cho độ sáng nền, cấu trúc không gian vĩ mô, độ tương phản toàn cục và sương mù.
- **Dải tần số trung bình (Mid-Frequency Band - $B_{\text{mid}}$)**:
  $$0.10 < \rho \le 0.40 \quad (10\% - 40\% \text{ tần số Nyquist})$$
  *Ý nghĩa vật lý*: Đại diện cho các đường viền chính (major edges), hình thái vật thể và bóng râm.
- **Dải tần số cao (High-Frequency Band - $B_{\text{high}}$)**:
  $$\rho > 0.40 \quad (> 40\% \text{ tần số Nyquist})$$
  *Ý nghĩa vật lý*: Đại diện cho chi tiết kết cấu mịn (fine textures), biên cạnh sắc nhọn và nhiễu ngẫu nhiên.

### 2.2. Phổ Năng lượng Sai số Thành phần (Band Error Energy Percentage):
Mật độ phổ công suất sai số tại tọa độ $(u, v)$ là $P_E(u, v) = |\mathcal{F}(E)(u, v)|^2$.
Tỷ lệ phần trăm năng lượng sai số thuộc dải tần $B_k$ ($k \in \{\text{low}, \text{mid}, \text{high}\}$):
$$S_k = \frac{\sum_{(u, v) \in B_k} P_E(u, v)}{\sum_{(u, v)} P_E(u, v)} \times 100\% \quad \left(\sum_k S_k = 100\%\right)$$

---

## 3. KỲ VỌNG ĐỊNH LƯỢNG ĐĂNG KÝ TRƯỚC (QUANTITATIVE PREDICTIONS)

Bảng dưới đây quy định các giá trị dự đoán định lượng trước khi chạy thực nghiệm cho mô hình cơ sở **PromptIR (Nhánh A)**:

| Tác vụ suy thoái | Kỳ vọng dải Thấp ($S_{\text{low}}$) | Kỳ vọng dải Trung ($S_{\text{mid}}$) | Kỳ vọng dải Cao ($S_{\text{high}}$) | Hình thái phổ sai số |
|---|:---:|:---:|:---:|---|
| **1. Nhiễu Gaussian thuần (Pure Noise)** | $S_{\text{low}} \le 15.0\%$ | $15.0\% \le S_{\text{mid}} \le 25.0\%$ | **$S_{\text{high}} \ge 65.0\%$** | Lệch mạnh về tần số cao (Unimodal High) |
| **2. Sương mù Koschmieder thuần (Pure Haze)** | **$S_{\text{low}} \ge 60.0\%$** | $25.0\% \le S_{\text{mid}} \le 35.0\%$ | $S_{\text{high}} \le 15.0\%$ | Lệch mạnh về tần số thấp (Unimodal Low) |
| **3. Suy thoái hỗn hợp (Hybrid = Noise + Haze)** | **$S_{\text{low}} \ge 35.0\%$** | $15.0\% \le S_{\text{mid}} \le 30.0\%$ | **$S_{\text{high}} \ge 35.0\%$** | **Phân bố hai cực rõ rệt (Bimodal Distribution)** |

### Hiện tượng "Bùng nổ Sai số Đa tần" (Cross-Band Error Surge):
Gọi $\mathcal{E}_{\text{hybrid}}$ là tổng công suất sai số trên tập Hybrid: $\mathcal{E}_{\text{hybrid}} = \sum_{(u,v)} P_E^{\text{hybrid}}(u, v)$.  
Gọi $\mathcal{E}_{\text{noise}}$ và $\mathcal{E}_{\text{haze}}$ là tổng công suất sai số khi cùng nội dung ảnh chỉ bị nhiễm riêng rẽ từng loại suy thoái.  
Định nghĩa Tỷ số Bùng nổ Năng lượng (Cross-Band Surge Ratio):
$$\text{Surge Ratio} = \frac{\mathcal{E}_{\text{hybrid}}}{\mathcal{E}_{\text{noise}} + \mathcal{E}_{\text{haze}}}$$
- **Kỳ vọng của H1**: Trên mô hình cơ sở PromptIR (Nhánh A), do xung đột biểu diễn không giải quyết được:
  $$\text{Surge Ratio}_{\text{Branch A}} \ge 1.20$$
  (Sai số hỗn hợp tăng vọt hơn 20% so với tổng rời rạc của hai tác vụ).
- **Kỳ vọng đối với FD-PromptIR (Nhánh B)**: Nhờ có nhánh prompt tần số thấp và cao độc lập:
  $$\text{Surge Ratio}_{\text{Branch B}} \le 1.05 \quad \text{và} \quad S_{\text{low}}^{\text{Branch B}} < S_{\text{low}}^{\text{Branch A}}, \; S_{\text{high}}^{\text{Branch B}} < S_{\text{high}}^{\text{Branch A}}$$

---

## 4. TIÊU CHÍ BÁC BỎ DỨT KHOÁT GIẢ THUYẾT H1 (FALSIFICATION CRITERIA)

Giả thuyết khoa học H1 (FD-PromptIR) được tuyên bố là **BỊ BÁC BỎ HOÀN TOÀN (REJECTED)** và **KHÔNG THỂ BẢO VỆ ĐƯỢC** nếu xảy ra **BẤT KỲ ĐIỀU KIỆN NÀO** trong các điều kiện sau đây:

1. **Bác bỏ vì Không tồn tại Phân bố Hai cực (Non-Bimodality in Hybrid)**:  
   Trên tập kiểm định Held-out Hybrid-OOD, phổ sai số của mô hình cơ sở PromptIR (Nhánh A) không thể hiện tính hai cực, cụ thể:
   $$S_{\text{low}} < 30.0\% \quad \text{HOẶC} \quad S_{\text{high}} < 30.0\%$$
   *Ý nghĩa*: Sai số phục hồi hybrid thực chất chỉ bị chi phối bởi một loại suy thoái đơn lẻ (ví dụ: chỉ mất tương phản hoặc chỉ có nhiễu), tiền đề cốt lõi về "xung đột tần số hai cực" của H1 là sai.

2. **Bác bỏ vì Không có Hiện tượng Bùng nổ Lỗi (No Cross-Band Surge)**:  
   Tỷ số bùng nổ sai số trên PromptIR:
   $$\text{Surge Ratio}_{\text{Branch A}} < 1.05$$
   *Ý nghĩa*: PromptIR tiêu chuẩn xử lý suy thoái hỗn hợp hoàn toàn tuyến tính theo phép cộng của hai tác vụ đơn lẻ, không hề xảy ra hiện tượng mâu thuẫn hay suy giảm năng lực biểu diễn do sự kết hợp của hai suy thoái.

3. **Bác bỏ vì Không có Sự Khác biệt Thống kê về Phổ Sai số giữa B và C (Statistical Insignificance)**:  
   Khi so sánh phân bố phổ sai số giữa Nhánh B (FD-PromptIR) và Nhánh C (Param-Matched PromptIR) trên toàn bộ tập ảnh test:
   Kiểm định thống kê Student's t-test ghép cặp (Paired t-test) hoặc Wilcoxon signed-rank test cho thấy sự khác biệt về $S_{\text{low}}$ hoặc $S_{\text{high}}$ giữa B và C không đạt ý nghĩa thống kê ($p \ge 0.05$).  
   *Ý nghĩa*: Bất kỳ sự cải thiện nào của B (nếu có) đơn thuần là do dung lượng tham số thêm vào chứ KHÔNG PHẢI do cơ chế tách dải tần số của FD-PromptIR.

---

## 5. HÀNH ĐỘNG DỨT KHOÁT KHI H1 BỊ BÁC BỎ

Nếu bất kỳ tiêu chí bác bỏ nào ở Mục 4 xảy ra:
1. **DỪNG NGAY LẬP TỨC** việc phát triển mô hình FD-PromptIR (Nhánh B).
2. **CẤM TUYỆT ĐỐI** việc ngụy biện rằng "tách tần số giúp khôi phục ảnh tốt hơn" trong bài báo khoa học.
3. **KÍCH HOẠT KẾ HOẠCH B**: Chuyển giao toàn bộ tài nguyên huấn luyện sang giả thuyết H4 (Orthogonal Prompt Regularization - OPR) hoặc công bố kết quả âm (Negative Result) về tính vô hiệu của việc tách tần số thủ công trong các kiến trúc Transformer phục hồi ảnh hiện đại.
