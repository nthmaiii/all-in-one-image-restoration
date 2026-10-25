# Báo Cáo Phân Tích Thất Bại Chuyên Sâu (Failure Analysis)
**Mã nhiệm vụ**: P2-T2 & P2-T3
**Thiết bị thực thi**: Google Colab Tesla T4 GPU (VRAM 14.56 GB) [ĐÃ XÁC NHẬN BẰNG LOG]
**Phương pháp**: Phân tích định lượng PSNR/SSIM, sai khác không gian sai số ($|I_{rec} - I_{gt}|$), quang phổ tần số 2D FFT, và thử thách OOD (Out-of-Distribution).

## 1. Thống Kê Các Kiểu Thất Bại Chủ Yếu Trên Baseline
- **Kiểu 1: Mất kết cấu chi tiết cao tần (Texture Over-smoothing)**: Chiếm ưu thế khi mức suy thoái lớn ($\sigma=50$, Mưa dày). Attention toàn cục có xu hướng làm nhẵn mịn các vân cấu trúc phức tạp.
- **Kiểu 2: Tồn dư sương mù / sọc mưa (Low-frequency Degradation Residue)**: Xuất hiện ở các vùng nền tương phản thấp hoặc trời mây, do thiếu thông tin biên định hướng.
- **Kiểu 3: Sụp đổ khi gặp suy thoái hỗn hợp (Catastrophic Hybrid Collapse)**: PSNR giảm từ ~31 dB xuống **16.79 dB** khi xuất hiện đồng thời cả Nhiễu và Sương mù. Nguyên nhân: Prompt vector bị xung đột biểu diễn (representation collision), không thể đồng thời điều chế cho 2 loại suy thoái trái ngược nhau về phổ tần số.
- **Kiểu 4: Sụp đổ ngoài khoảng phân phối độ mạnh (Out-of-Range Degradation)**: Khi mức nhiễu tăng lên $\sigma=75$ (vượt ngoài dải huấn luyện [0, 50]), SSIM giảm mạnh từ $0.99$ xuống **0.6378** (-36%).

## 2. Bảng Kết Quả Thử Thách OOD Thực Nghiệm Trên Tesla T4
*(Trích xuất từ `repo/results/ood_baseline.csv` [ĐÃ XÁC NHẬN BẰNG LOG])*

| Kịch bản OOD | Độ nhiễu / Suy thoái | PSNR (dB) | SSIM | Mức độ suy giảm | Đánh giá |
|---|---|:---:|:---:|:---:|:---|
| **Unseen Noise $\sigma=10$** | Nhiễu nhẹ ngoài tập rời rạc | 56.18 | 0.9986 | Bình thường | Phục hồi hoàn hảo |
| **Unseen Noise $\sigma=35$** | Nhiễu trung bình ngoài tập rời rạc | 49.07 | 0.9964 | Nhẹ | Khả năng nội suy tốt |
| **Extreme Noise $\sigma=75$** | Nhiễu cực đoan ngoài dải huấn luyện | **30.37** | **0.6378** | **Nặng** | **Mất 36% cấu trúc, hạt nhiễu bết dính** |
| **Hybrid (Noise $\sigma=25$ + Haze)** | Suy thoái hỗn hợp đa loại | **16.79** | **0.9668** | **Thảm họa** | **Mất toàn bộ độ tương phản, sụp đổ biểu diễn** |

## 3. Danh Sách Các Mẫu Điển Hình
*(Xem chi tiết trong `results/diagnostics/worst_cases_summary.csv`)*

| Tác vụ | Thứ hạng tệ | Tên ảnh | PSNR (dB) | SSIM | Kiểu lỗi chính | Lệch màu | Tỷ lệ lỗi cao tần |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Denoise_BSD68 | 1 | `test001.png` | 21.61 | 0.5104 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.01 | 0.931 |
| Denoise_BSD68 | 2 | `test005.png` | 22.28 | 0.5549 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.05 | 0.914 |
| Denoise_BSD68 | 3 | `test002.png` | 22.42 | 0.4973 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.01 | 0.890 |
| Denoise_BSD68 | 4 | `test003.png` | 22.92 | 0.4464 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.01 | 0.904 |
| Denoise_BSD68 | 5 | `test004.png` | 24.39 | 0.5100 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.04 | 0.883 |
