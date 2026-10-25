# Đối Chiếu Số Liệu Tái Hiện Baseline (P1-T3)

> [!NOTE]
> Bảng tổng hợp số liệu công bố chính thức từ bài báo của **PromptIR (NeurIPS 2023)** và **AdaIR (ICLR 2025)** trên thiết lập All-in-One Image Restoration (AiOIR-3).
> Tiêu chí nghiệm thu: Độ lệch khi chạy thực nghiệm trên checkpoint chính thức $\le 0.30$ dB.

---

## 1. Bảng Số Liệu Công Bố Chính Thức Từ Bài Báo (Ground Truth Benchmarks)

| Mô hình (Baseline) | Denoise-15 (BSD68) | Denoise-25 (BSD68) | Denoise-50 (BSD68) | Derain (Rain100L) | Dehaze (SOTS) | Trung bình (AiOIR-3) | Số tham số (Params) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **AirNet** (CVPR 2022) | 33.92 dB / 0.933 | 31.27 dB / 0.889 | 27.90 dB / 0.817 | 34.90 dB / 0.957 | 27.90 dB / 0.962 | 31.18 dB | 8.93 M |
| **PromptIR** (NeurIPS 2023) | 33.98 dB / 0.934 | 31.31 dB / 0.891 | 28.04 dB / 0.822 | 36.37 dB / 0.974 | 30.58 dB / 0.974 | 32.06 dB | 35.59 M |
| **AdaIR** (ICLR 2025) | **34.08 dB** / 0.936 | **31.43 dB** / 0.895 | **28.18 dB** / 0.828 | **38.60 dB** / 0.981 | **31.89 dB** / 0.985 | **32.84 dB** | 28.78 M |

*(Ghi chú: Số liệu lấy trực tiếp từ Table 1 của PromptIR.pdf và Table 1 của AdaIR.pdf, tính trên hệ màu RGB tiêu chuẩn).*

---

## 2. Các Nguồn Gây Chênh Lệch Tiềm Ẩn & Quy Chuẩn Đo Đạc (Discrepancy Analysis)

Khi chạy kiểm thử trên Google Colab qua script `evaluate_baselines.py`, nếu có chênh lệch giữa số đo thực tế và bảng trên, cần đối soát theo 4 quy chuẩn sau:

1. **Hệ màu tính toán (RGB vs Y-channel)**:
   - Các bài báo phục hồi cổ điển (Derain, SR) thường chỉ tính PSNR trên kênh độ sáng Y (Y-channel của YCbCr), làm điểm số cao hơn thực tế từ **+1.0 dB đến +1.8 dB**.
   - Cả PromptIR và AdaIR chính thức đều tính trên **RGB toàn dải**. Script `evaluate_baselines.py` đã khóa cứng tính trên RGB.

2. **Miền giá trị (Dynamic Range & Clipping)**:
   - Phải chuẩn hóa ảnh về $[0.0, 1.0]$ với `data_range=1.0` (hoặc $[0, 255]$ nguyên bản).
   - Trước khi đo PSNR/SSIM, tensor phục hồi phải được cắt ngưỡng `np.clip(img, 0.0, 1.0)` để tránh giá trị tràn ngoài dải màu.

3. **Hiện thực hóa nhiễu ngẫu nhiên (AWGN Seed)**:
   - Trong tác vụ khử nhiễu (Denoising), ảnh đầu vào được sinh online bằng cách cộng nhiễu Gauss:
     $$y = x + n, \quad n \sim \mathcal{N}\left(0, \left(\frac{\sigma}{255}\right)^2 I\right)$$
   - Sự khác biệt về seed ngẫu nhiên giữa các lần chạy có thể làm lệch $\pm 0.05 \text{ dB}$. Script đã cố định `np.random.seed(42)` để đảm bảo tính tái lập 100%.

4. **Kích thước và căn lề kích thước ảnh (Padding / Modulo)**:
   - Do mạng sử dụng kiến trúc U-Net 4 tầng phân giải với downsampling/upsampling lũy thừa $2^4 = 16$, các ảnh có chiều cao/rộng không chia hết cho 16 được pad đối xứng hoặc cắt góc (`check_image_size`).
