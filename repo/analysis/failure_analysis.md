# Báo Cáo Phân Tích Thất Bại Chuyên Sâu (Failure Analysis)
**Mã nhiệm vụ**: P2-T2
**Phương pháp**: Phân tích định lượng PSNR/SSIM, sai khác không gian sai số ($|I_{rec} - I_{gt}|$), quang phổ tần số 2D FFT, và độ lệch sắc độ.

## 1. Thống Kê Các Kiểu Thất Bại Chủ Yếu Trên Baseline
- **Kiểu 1: Mất kết cấu chi tiết cao tần (Texture Over-smoothing)**: Chiếm ưu thế khi mức suy thoái lớn (Nhiễu $\sigma=50$, Mưa dày). Attention toàn cục có xu hướng làm nhẵn mịn các vân cấu trúc phức tạp.
- **Kiểu 2: Tồn dư sương mù / sọc mưa (Low-frequency Degradation Residue)**: Xuất hiện ở các vùng nền tương phản thấp hoặc trời mây, do thiếu thông tin biên định hướng.
- **Kiểu 3: Lệch màu sắc (Color Shift)**: Do ma trận biến đổi trong decoder làm dịch chuyển giá trị kỳ vọng giữa các kênh RGB.

## 2. Danh Sách Các Mẫu Điển Hình

| Tác vụ | Thứ hạng tệ | Tên ảnh | PSNR (dB) | SSIM | Kiểu lỗi chính | Lệch màu | Tỷ lệ lỗi cao tần |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Denoise_BSD68 | 1 | `test001.png` | 21.61 | 0.5104 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.01 | 0.931 |
| Denoise_BSD68 | 2 | `test005.png` | 22.28 | 0.5549 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.05 | 0.914 |
| Denoise_BSD68 | 3 | `test002.png` | 22.42 | 0.4973 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.01 | 0.89 |
| Denoise_BSD68 | 4 | `test003.png` | 22.92 | 0.4464 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.01 | 0.904 |
| Denoise_BSD68 | 5 | `test004.png` | 24.39 | 0.51 | Tồn dư suy thoái nền (Low-freq Residue / Haze/Streaks) | 0.04 | 0.883 |

*(Xem toàn bộ chi tiết trong `D:\nckh_coChau\repo\results\diagnostics\worst_cases_summary.csv` và thư mục hình ảnh `D:\nckh_coChau\repo\results\diagnostics\worst_cases`)*
