# Bảng Theo Dõi Kế Hoạch Nghiên Cứu (PLAN_STATUS)

> [!NOTE]
> Bảng trạng thái cập nhật tiến độ thực hiện theo kế hoạch 12 tuần (Kịch bản 1).
> Trạng thái gồm: `CHƯA LÀM`, `ĐANG LÀM`, `ĐẠT`, `KHÔNG ĐẠT`, `CHỜ LOG COLAB`.

| Giai đoạn (Phase) | Mã Task | Tên Task & Nội dung chính | Trạng thái | File đầu ra chính | Tiêu chí nghiệm thu |
|---|---|---|:---:|---|---|
| **P0. Thiết Lập** | **P0-T1** | Tạo cấu trúc repo, `PLAN_STATUS.md`, `research_log.md`, `requirements.txt` | **ĐẠT** | `repo/`, `requirements.txt`, `PLAN_STATUS.md` | Repo chạy được `pip install -r requirements.txt` [ĐÃ XÁC NHẬN] |
| | **P0-T2** | Notebook `colab/00_setup.ipynb` (mount Drive, check T4, test 1 batch) | **ĐẠT** | `repo/colab/00_setup.ipynb` | Chạy sạch trên Colab, in GPU T4 và torch [ĐÃ XÁC NHẬN BẰNG LOG T4] |
| | **P0-T3** | Hướng dẫn & script chuẩn bị dữ liệu theo PromptIR/AdaIR | **ĐẠT** | `data/README.md`, `scripts/download_datasets.py`, `scripts/verify_data_counts.py` | Kiểm tra tự động số ảnh chuẩn khớp bài báo [ĐÃ XÁC NHẬN BẰNG LOG T4: 68/68, 24/24, 100/100, 100/100] |
| | **P0-T4** | Bảng tổng quan 25 bài báo (link xác minh, ý chính, code, điểm yếu) | **ĐẠT** | `repo/analysis/literature_table.md` | Bảng 25 bài đã kiểm chứng link và code [ĐÃ XÁC NHẬN] |
| | **GATE P0** | Chạy `01_master_pipeline.ipynb` trên Colab T4, dữ liệu sẵn sàng | **ĐẠT** | Log Colab từ User | Vượt qua Gate P0 [ĐÃ XÁC NHẬN BẰNG LOG T4] |
| **P1. Tái Hiện Baseline** | **P1-T1** | Script đánh giá pretrained PromptIR & AdaIR trên testsets (PSNR, SSIM, LPIPS) | **ĐẠT** | `repo/scripts/evaluate_baselines.py`, `results/baseline_results.csv` | Lệch $\le 0.03$ dB trên Denoise, Rain100L đạt 37.06 dB RGB / 38.45 dB Y [ĐÃ XÁC NHẬN BẰNG LOG T4] |
| | **P1-T2** | Đo tham số (Params), FLOPs, độ trễ (latency), VRAM đỉnh trên T4 | **ĐẠT** | `repo/scripts/profile_efficiency.py`, `results/efficiency.csv` | PromptIR 35.59M / 79.20 GFLOPs / 72.1ms / 290MB VRAM; AdaIR 28.78M / 73.72 GFLOPs / 81.1ms / 402MB VRAM [ĐÃ XÁC NHẬN BẰNG LOG T4] |
| | **P1-T3** | Đối chiếu bài báo vs code, phân tích chênh lệch Y-channel vs RGB | **ĐẠT** | `repo/analysis/baseline_reproduction.md`, `repo/analysis/paper_vs_code.md` | Phân tích rõ nguyên nhân chênh lệch (Y vs RGB, clipping, progressive) [ĐÃ HOÀN TẤT] |
| | **GATE P1** | Baseline được tái hiện thành công trên GPU T4 | **ĐẠT** | `results/baseline_results.csv` | Toàn bộ 10/10 cấu hình kiểm thử khớp hoặc vượt bài báo công bố [ĐÃ XÁC NHẬN BẰNG LOG T4] |
| **P2. Chẩn Đoán Thất Bại** | **P2-T1** | Phân tích sâu code vs paper (soft prompt lookup, FiLM modulation, progressive) | **ĐẠT** | `repo/analysis/paper_vs_code.md` | Chỉ ra 5 sai khác bản chất giữa code và bài báo |
| | **P2-T2** | Tìm 30 ảnh lỗi tệ nhất mỗi tập, phân tích phổ 2D FFT, heatmap sai số | **ĐẠT (Code)** | `repo/scripts/diagnose_failures.py`, `analysis/failure_analysis.md`, figures | Phân loại tự động 3 kiểu lỗi: texture smoothing, color drift, residue |
| | **P2-T3** | Thử thách OOD (sigma 10/35/75, suy thoái hỗn hợp nhiễu + sương mù) | **ĐẠT** | `repo/results/ood_baseline.csv` | Phát hiện 2 thất bại nghiêm trọng: $\sigma=75$ SSIM giảm còn 0.6378; Nhiễu+Sương mù PSNR sụp đổ còn 16.79 dB [ĐÃ XÁC NHẬN BẰNG LOG T4] |
| | **GATE P2** | Duyệt failure_analysis.md và xác nhận thất bại đáng giải quyết | **SẴN SÀNG DUYỆT** | Xác nhận từ User | Chốt thất bại cốt lõi trước khi sang P3 |
| **P3. Giả Thuyết & Tính Mới** | P3-T1 | Sinh 8-10 giả thuyết từ P2 theo định dạng 4 câu (a)-(d) | CHƯA LÀM | `repo/analysis/hypotheses.md` | Đủ 4 câu (a)-(d), đánh dấu ghép module |
| | P3-T2 | Tự phản biện (Reviewer 2), rút gọn xuống 4-5 giả thuyết | CHƯA LÀM | `repo/analysis/hypotheses.md` | Nêu lý do từ chối mạnh nhất |
| | P3-T3 | Kiểm tra tính mới (arXiv, Google Scholar), liệt kê 5 bài gần nhất | CHƯA LÀM | `repo/analysis/novelty_check.md` | Tránh trùng lặp với prior art |
| | **GATE P3** | User chọn 1-2 giả thuyết đi tiếp kèm Kế hoạch B | CHƯA LÀM | Quyết định từ User | Bạn chọn, tôi không tự quyết |
| **P4. Thí Nghiệm Thăm Dò** | P4-T1 | Thiết kế thí nghiệm tối thiểu (ngân sách nhỏ, đối chứng cùng ngân sách) | CHƯA LÀM | `repo/analysis/pilot_design.md` | Tiêu chí thắng/thua ghi trước khi chạy |
| | P4-T2 | Cài đặt mô hình đề xuất trong `src/`, kiểm tra backward, FLOPs | CHƯA LÀM | `repo/src/` | Chạy thử 100 iteration không lỗi |
| | P4-T3 | Notebook pilot có resume, huấn luyện và phân tích log | CHƯA LÀM | `repo/results/pilot_results.csv`, loss curve | Có tín hiệu tốt hơn hoặc bị bác bỏ rõ |
| | **GATE P4** | Xác nhận tín hiệu khả quan hoặc rẽ nhánh Kế hoạch B | CHƯA LÀM | Xác nhận từ User | Ghi rõ kết quả âm nếu bị bác bỏ |
| **P5. Huấn Luyện Đầy Đủ** | P5-T1 | Cấu hình huấn luyện chính thức (chia chặng $\le 3$h, resume) | CHƯA LÀM | `repo/configs/train_main.yaml` | Cấu hình khóa cố định |
| | P5-T2 | Huấn luyện đề xuất vs đối chứng cùng ngân sách (1-3 seeds) | CHƯA LÀM | `repo/runs/` | Chạy trên Colab với AMP |
| | P5-T3 | Đánh giá đầy đủ trên toàn bộ các tập test | CHƯA LÀM | `repo/results/main_results.csv` | Vượt/ngang baseline có ý nghĩa |
| | **GATE P5** | Xác nhận kết quả đủ độ dày dặn để viết bài | CHƯA LÀM | Quyết định từ User | Chốt số liệu chính |
| **P6. Ablation & Đánh Giá** | P6-T1 | Huấn luyện & đánh giá ablation từng thành phần | CHƯA LÀM | `repo/results/ablation.csv` | Chứng minh đóng góp từng module |
| | P6-T2 | Kiểm tra tổng quát hóa OOD vs baseline | CHƯA LÀM | `repo/results/ood.csv` | Đo độ bền ngoài phân phối |
| | P6-T3 | Đo đạc hiệu năng chi tiết (params, FLOPs, runtime) và hình định tính | CHƯA LÀM | `repo/results/efficiency_final.csv`, figures | Mọi tuyên bố đều có bằng chứng |
| | **GATE P6** | Duyệt toàn bộ bằng chứng trong `results/` | CHƯA LÀM | Xác nhận từ User | Sẵn sàng viết bản thảo |
| **P7. Viết Bài & Đóng Gói** | P7-T1 | Lập dàn ý bài báo theo template venue mục tiêu | CHƯA LÀM | `repo/paper/outline.md` | User duyệt dàn ý |
| | P7-T2 | Viết toàn văn bản thảo (Intro, Related Work, Method, Exp, Concl) | CHƯA LÀM | `repo/paper/draft.md` / LaTeX | Số liệu khớp 100% với CSV |
| | P7-T3 | Kiểm tra chéo số liệu, trích dẫn, đóng gói repo mã nguồn mở | CHƯA LÀM | `repo/README.md`, instructions | User duyệt trước khi nộp bài |
| | **GATE P7** | User đọc toàn bộ và ký duyệt nộp bài báo | CHƯA LÀM | Quyết định cuối cùng | Hoàn tất đề tài |
