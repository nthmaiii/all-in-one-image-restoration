# Thiết Kế Thí Nghiệm Thăm Dò (Pilot Experiment Design)
**Mã nhiệm vụ**: P4-T1
**Giai đoạn**: Phase P4 - Pilot Validation
**Mục tiêu**: Xác thực tính hiệu quả của Giả thuyết **H1 (Frequency-Decoupled Dynamic Prompting - FD-Prompt)** kết hợp **H4 (Orthogonal Prompt Regularization - OPR)** trên GPU Tesla T4 trước khi tiến hành huấn luyện quy mô lớn.

---

## 1. Mục Tiêu Khoa Học & Câu Hỏi Cốt Lõi
- **Câu hỏi thực nghiệm**: 
  Liệu việc phân tách không gian đặc trưng thành hai nhánh phổ tần số trực giao ($f_{low}$ cho suy thoái DC/sương mù và $f_{high}$ cho suy thoái AC/nhiễu/vệt mưa) có giải quyết được hiện tượng **sụp đổ suy thoái hỗn hợp (PSNR 16.79 dB)** của PromptIR hay không?
- **Ràng buộc phần cứng**: Google Colab Free Tesla T4 (VRAM: 14.56 GB), thời gian huấn luyện pilot $\le 60$ phút (3,000 iterations).

---

## 2. Thiết Lập Thí Nghiệm Đối Chứng Công Bằng (Controlled Fair Budget)

Để đảm bảo kết quả trung thực và khách quan, cả hai mô hình (Baseline vs Proposed) phải tuân theo cấu hình đối chứng giống nhau 100%:

| Tham số thực nghiệm | Cấu hình đối chứng (Baseline PromptIR) | Cấu hình đề xuất (FD-PromptIR + OPR) |
|---|:---:|:---:|
| **Kiến trúc Backbone** | Restormer 4 cấp (dim=48, blocks=[4,6,6,8]) | Restormer 4 cấp (dim=48, blocks=[4,6,6,8]) |
| **Cơ chế Prompting** | Standard PromptGenBlock (Single Bank) | **FDPromptGenBlock (Dual Frequency Banks)** |
| **Ràng buộc trực giao** | Không ($\lambda_{orth} = 0$) | **Có ($\lambda_{orth} = 0.01$)** |
| **Số lượng Iterations** | 3,000 iterations | 3,000 iterations |
| **Kích thước Patch** | $128 \times 128$ | $128 \times 128$ |
| **Kích thước Batch** | 4 (T4 AMP fp16) | 4 (T4 AMP fp16) |
| **Tốc độ học (Learning Rate)** | $2 \times 10^{-4}$ với Cosine Annealing | $2 \times 10^{-4}$ với Cosine Annealing |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight_decay=$10^{-4}$) | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight_decay=$10^{-4}$) |
| **Tập dữ liệu huấn luyện pilot** | Hỗn hợp cân bằng (Noise $\sigma \in [15, 50]$, Rain100L, SOTS Haze, Hybrid Noise+Haze) | Hỗn hợp cân bằng (Noise $\sigma \in [15, 50]$, Rain100L, SOTS Haze, Hybrid Noise+Haze) |

---

## 3. Tiêu Chí Thắng / Thua Được Khóa Trước (Pre-defined Win/Loss Criteria)

### A. Tiêu chí THẮNG (PASS - Chuyển sang Phase P5 Huấn Luyện Đầy Đủ)
1. **Khắc phục sụp đổ suy thoái hỗn hợp**:
   - Trên tập kiểm thử Hybrid (Noise $\sigma=25$ + Sương mù), Proposed FD-PromptIR đạt PSNR $\ge 24.0\text{ dB}$ (tăng tối thiểu $\ge +7.0\text{ dB}$ so với mức 16.79 dB của Baseline).
2. **Bảo toàn và cải thiện tác vụ đơn**:
   - Trên các tập test chuẩn (BSD68 Denoise, Rain100L Derain), đạt hiệu năng tương đương hoặc vượt baseline ($\Delta \text{PSNR} \ge +0.2\text{ dB}$).
3. **Chi phí phần cứng**:
   - Số tham số tăng $\le 3\%$ ($< 37\text{M}$).
   - FLOPs tăng $\le 5\%$ ($< 83\text{ GFLOPs}$).
   - Độ trễ trên T4 tăng $\le 10\%$ ($< 80\text{ ms}$).

### B. Tiêu chí THUA (FAIL - Rẽ nhánh sang Kế Hoạch B)
- Nếu sau 3,000 iterations:
  - $\Delta \text{PSNR}$ trên tập Hybrid $< +2.0\text{ dB}$.
  - Hoặc hàm mất mát không hội tụ / FLOPs tăng $> 15\%$.
- $\rightarrow$ **Kế hoạch B**: Chuyển sang Giả thuyết H4 kết hợp cơ chế Routing đa tạp thích ứng (Continuous Manifold Routing) mà không phân tách không gian tần số.

---

## 4. Các File Đầu Ra Kỳ Vọng Của Phase P4
1. `repo/src/models/fd_promptir.py`: Cài đặt hoàn chỉnh mô hình FD-PromptIR.
2. `repo/scripts/train_pilot.py`: Script huấn luyện pilot có checkpointing và resume.
3. `repo/colab/02_pilot_experiment.ipynb`: Notebook Colab thực thi 1-click cho User chạy trên GPU T4.
4. `repo/results/pilot_results.csv`: Bảng đối chứng định lượng Baseline vs Proposed.
