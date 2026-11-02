# Interaction-Aware Residual PromptIR — pilot v4

**Trạng thái:** triển khai và kiểm thử cục bộ; chưa có kết quả huấn luyện T4 của v4.
Log PHASE-HOPFIELD v3 trong `README_CURR.docx` không phải kết quả của phương pháp mới.

## Ý tưởng và các đối chứng

Giữ nguyên lớp `PromptIR` và forward chính thức. Thay mỗi prompt generator bằng
`base_prompt + residual`, lớp chiếu residual cuối khởi tạo bằng zero. Backbone và
prompt gốc được nạp đầy đủ từ checkpoint; thiếu/sai key là lỗi.

| Variant | Descriptor | Toán tử tương tác |
|---|---|---|
| `baseline` | PromptIR gốc | Không có residual |
| `global_linear` | Mean/std toàn ảnh, lặp trên lưới 2×2 | `(A+B)/2` |
| `regional_linear` | Mean/std lưới 2×2 + đường đặc trưng cục bộ | `(A+B)/2` |
| `regional_jordan` | Như trên | `(AB+BA)/2` |
| `regional_lie` | Như trên | `(AB-BA)/2` |

Bốn variant residual có **cùng số tham số**, mọi generator đều tham gia forward.
Đã đếm trên kiến trúc đầy đủ: baseline **35.592.263**, mỗi variant residual
**35.800.503** tham số (thêm **208.240**).
A/B được chặn Frobenius norm trước tích ma trận. Các phép mới, loss và norm chạy
FP32, kể cả khi backbone dùng AMP. Residual bị chặn L2 tối đa 20% của base prompt
trên từng mẫu; ratio chỉ là chẩn đoán, **không ép đạt 15%**. Đánh giá bật/tắt residual
đo đóng góp vào PSNR. Zero-init chỉ đảm bảo gradient vào lớp cuối ở bước đầu;
generator bên trong bắt đầu nhận gradient khi lớp cuối mở.

Đây là giả thuyết nghiên cứu, không phải kết luận rằng ma trận A/B nhận diện được
toán tử vật lý. FrFT và Hopfield chưa được đưa vào v4 để tách riêng đóng góp của
descriptor và tương tác. Dữ liệu haze hiện dùng transmission đồng nhất theo ảnh;
chưa có bằng chứng cho lợi thế trên haze biến đổi không gian thực tế.

## Dữ liệu và endpoint

- Ảnh sạch train/val thật, độc lập với test. `prepare_data_v2.py --require_real_images`
  dừng nếu tải ảnh thật thất bại; không sinh ảnh dự phòng ngẫu nhiên.
- Train xác định theo `(seed, sample_index)`, mỗi 8 mẫu gồm 2 noise, 2 rain, 2 haze,
  1 haze→noise và 1 noise→haze. Chỉ clip sau khi hợp thành hybrid.
- Validation dùng cùng ảnh và seed=999 cho mọi training seed, đo noise/rain/haze
  tổng hợp cùng hai thứ tự hybrid. Chọn best checkpoint bằng PSNR hybrid val.
- **Primary: final / held-out composition**: sigma `{15,35,50}` × t `{0.4,0.8}` ×
  2 thứ tự trên *mỗi* ảnh; A=1. 30 ảnh → 360 mẫu. Một phần support trùng train;
  không gọi toàn bộ tập này là OOD.
- **Secondary extrapolation**: sigma `{10,65}` × t `{0.25,0.85}` × 2 thứ tự;
  30 ảnh → 240 mẫu. Sigma/t nằm ngoài khoảng hybrid train.
- Chọn ảnh round-robin giữa BSD68/Kodak24; cùng noise realization cho cặp hai thứ tự.
  Giữ tỷ lệ khung hình, chỉ thu nhỏ ảnh lớn; không cắt méo về hình vuông.
- Best-validation checkpoint là sensitivity analysis riêng. Mỗi checkpoint đo
  bật/tắt residual trên composition/extrapolation; ratio được trung bình qua mọi ảnh.
- Metrics RGB, clamp [0,1], uint8 rounding, PSNR/SSIM thật từ scikit-image; NaN/Inf
  bị từ chối **trước clamp/cast**. Không làm tròn metrics trước tổng hợp qua seed.

**Giới hạn:** validation tổng hợp không thay thế benchmark chuẩn BSD68/Rain100L/SOTS.
`comparison.json` không tuyên bố đạt đủ 5 gates cũ hoặc ý nghĩa thống kê suy diễn.

## Chạy trên Colab

Mở trực tiếp [notebook v4 trên Colab](https://colab.research.google.com/github/nthmaiii/all-in-one-image-restoration/blob/main/repo/colab/05_interaction_promptir_v4.ipynb).
Không cần upload mã nguồn: notebook tự clone GitHub và checkout `GIT_REF` (mặc định
`main`; có thể dùng commit SHA để tái lập). Notebook dùng `subprocess.run(..., check=True)` để lỗi
ở bất kỳ nhánh nào sẽ dừng cell, lưu checkpoint chính trên Drive để resume.

Thứ tự: smoke FP32/AMP → ổn định 100 iters trên 3 seed → baseline 3 seed → từng
variant residual 3 seed. Giữ FP32 làm mốc đầu tiên; AMP dùng một output directory
riêng nếu muốn kiểm tra thêm. Full grid và intervention tốn đánh giá hơn v3;
chưa có ước lượng thời gian T4 được đo cho v4.

Ví dụ từ thư mục `repo`:

```bash
python scripts/download_datasets.py --pilot_v2
python scripts/prepare_data_v2.py --require_real_images
python scripts/train_interaction_promptir.py --variant baseline --seed 42 --precision fp32
python scripts/train_interaction_promptir.py --variant regional_lie --seed 42 --precision fp32
```

Chạy đủ các variant/seed trước khi tổng hợp:

```bash
python scripts/compare_interaction_runs.py --output_dir runs/interaction_v4
```

Script so sánh chỉ nhận run hoàn tất, cùng protocol theo cặp, cùng evaluation data
qua seed, và cùng ngân sách tham số giữa residual controls. Báo cáo mean ± sample
std, paired deltas, từng thứ tự, và PSNR delta khi tắt residual. Các ngưỡng lịch sử
`max(0.40,2*std_A)`/`max(0.25,1.5*std_A)` được hiển thị như engineering reference,
không tự kết luận thắng toàn bộ nghiên cứu.

## Resume và truy vết

```bash
python scripts/train_interaction_promptir.py --variant baseline --seed 42 --iters 1500 --stop_after 250
python scripts/train_interaction_promptir.py --variant baseline --seed 42 --iters 1500 --resume
```

Giữ nguyên tổng `--iters`: cosine schedule không được đổi từ 1.500 thành 5.000
trong cùng lượt resume. Dữ liệu, full checksum 50 mẫu, mã nguồn, pretrained hash,
evaluation seed và cấu hình được khóa vào `protocol_seed_*.json`. Resume khôi phục
optimizer/scheduler/scaler/RNG và bắt đầu đúng sample tiếp theo; không phát lại batch cũ.
CUDA được yêu cầu dùng thuật toán deterministic khi có, cảnh báo khi kernel như
bilinear/adaptive pooling backward chưa hỗ trợ. Resume CPU đã kiểm tra khớp bit;
không tuyên bố T4 khớp bit. Cần đo sai lệch resume trên T4 so với hai lượt liên tục.

Một run gồm `status.json`, `protocol.json`, `data_manifest.json`, `environment.json`,
`latest_checkpoint.pth`, `best_model.pth`, `final_model.pth`, `summary.json`.
NaN/Inf ở activation/loss/gradient/weight làm run `failed`; không tạo summary chất lượng
hoàn tất. Có thể resume từ checkpoint hữu hạn đã lưu với cùng protocol; nếu sửa code,
cần một output directory mới và baseline đối chứng mới.

Smoke test bị tách vào `runs/smoke_interaction_v4`; `--random_init` chỉ được dùng
với smoke. Runner mặc định yêu cầu checkpoint thật, không âm thầm train từ random.

## Kiểm thử cục bộ

```bash
python -m pytest tests/test_interaction_pipeline.py -q
```

Test gồm phát hiện NaN/Inf trước metrics, lưới đầy đủ, cùng pixel giữa training seeds,
zero-init đồng nhất với forward PromptIR, strict checkpoint load, budget tham số,
gradient mở sau zero-init, cap residual, autocast CPU, resume optimizer/RNG/scheduler,
round-trip runner và báo cáo chỉ nhận run hợp lệ. CPU autocast không thay thế việc
kiểm tra CUDA FP16/T4 và checkpoint pretrained thật.
