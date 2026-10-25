# Research log — All-in-One Image Restoration

## 0. Quy ước làm việc

- Cập nhật đầu tiên: 2026-10-25. Ngôn ngữ trao đổi: tiếng Việt.
- Mục tiêu: đóng góp có cơ chế và khả năng bác bỏ, hướng CVPR/ICCV/ECCV/NeurIPS/ICLR.
- Tài nguyên người dùng xác nhận: Google Colab; chưa có dataset. Chưa biết GPU, quota, ngân sách và thời hạn.
- Bắt đầu từ thất bại, sau đó mới thiết kế phương pháp. Mỗi hướng phải có: hiện tượng, cơ chế, phép bác bỏ, khác biệt với bài gần nhất.
- Không coi việc ghép module, đổi backbone, hoặc tăng PSNR nhỏ là mặc nhiên có tính mới.
- Mọi số trong phần thiết kế là NGƯỠNG ĐỀ XUẤT/ƯỚC TÍNH, không phải kết quả.

### Nhãn bằng chứng

- **[BÀI CHỨNG MINH/BÁO CÁO]**: nội dung tác giả có trình bày/thực nghiệm, trong phạm vi thiết lập của bài. Không đồng nghĩa đã được chúng ta tái lập hoặc chứng minh quan hệ nhân quả.
- **[SUY LUẬN]**: suy ra từ tài liệu, mã nguồn hoặc toán học; phải ghi giả định.
- **[PHỎNG ĐOÁN]**: giả thuyết chưa có dữ liệu thực nghiệm của dự án.
- **[QUAN SÁT CỤC BỘ]**: sự kiện kiểm tra trực tiếp trong thư mục.
- **[THIẾT KẾ] / [ƯỚC TÍNH]**: protocol hoặc ngân sách dự kiến.

## 1. Vòng R0 — khảo sát và sàng lọc, 2026-10-25

### 1.1. Phạm vi đã kiểm tra

[QUAN SÁT CỤC BỘ] Đã xem danh mục các thư mục gốc: All-in-one, transformer, CNN GAN, Diffusion-Model, LLM-Guided, paper2026, survey, Dataset-Metrix, Toolbox.

[QUAN SÁT CỤC BỘ] .gitignore bỏ qua các thư mục tài liệu/repo và PDF; công cụ glob mặc định không trả về chúng. Đã dùng đọc thư mục trực tiếp. Không có research_log.md ở thư mục gốc trước vòng này. Không tìm thấy AGENTS.md bằng glob, nhưng tìm kiếm đó không bảo đảm bao phủ nội dung bị ignore.

Đọc trực tiếp các PDF:

- All-in-one/Dynamic Degradation Decomposition.pdf — D3Net, arXiv:2502.19068v2.
- paper2026/OPIR.pdf — From Physical Degradation Models to Task-Aware All-in-One Image Restoration, arXiv:2601.10192v1.
- paper2026/QuReC.pdf — arXiv:2607.15097v2.
- paper2026/CWP-Net.pdf — arXiv:2603.03839v1.

Đọc README/mã liên quan:

- All-in-one/All-in-One-Image-Restoration-Survey/README.md: dùng như chỉ mục phát hiện tài liệu, không dùng làm bằng chứng duy nhất cho kết quả/venue.
- All-in-one/PromptIR/README.md và net/model.py, đặc biệt dòng 218–235 và 338–367.
- All-in-one/AdaIR/README.md; All-in-one/BaryIR/README.md.
- All-in-one/CPLIR/README.md; All-in-one/MiOIR/README.md.
- All-in-one/abair/README.md; transformer/NAFNet/readme.md; CNN GAN/BSRGAN/README.md.

[QUAN SÁT CỤC BỘ] PromptIR/ckpt chỉ có README.md tại thời điểm đọc. README có liên kết checkpoint, nhưng chưa tải/kiểm tra khả dụng. BaryIR/AdaIR README cũng cung cấp liên kết weights. Chưa kiểm tra toàn bộ checkpoint của mọi repo.

Giới hạn: chưa đọc toàn bộ PDF/repo trong kho; chưa cài môi trường, tải dataset, chạy GPU hoặc tái lập kết quả. Một số repo cục bộ có thể cũ hơn bản trực tuyến. ABAIR README cục bộ vẫn nói sẽ phát hành code; không suy ra tình trạng phát hành hiện tại từ đó.

### 1.2. Kết quả tra cứu tính mới

Đã truy cập arXiv qua trang abstract, API tìm kiếm và HTML. Google trả về trang yêu cầu JavaScript; Bing trả kết quả không liên quan nên không dùng làm bằng chứng. Một số truy vấn API sau đó timeout. Kết quả truy vấn rộng có nhiễu; đây là sàng lọc ban đầu, không phải systematic review và KHÔNG chứng minh không tồn tại prior art.

Các truy vấn đã thử:

1. `"image restoration" AND "order"` — tìm được DiTTo và CURE.
2. `"image restoration" AND (counterfactual OR intervention OR probing)` — tìm được DIL, BiDeMem và các công trình liên quan.
3. `"image restoration" AND ("non-commutative" OR identifiability OR "degradation order")` — kết quả nhiều nhiễu, chưa đủ kết luận khoảng trống.
4. `"all-in-one" AND (Jacobian OR "response signature" OR "system identification")` — ít kết quả liên quan; không coi là bằng chứng tính mới.
5. Truy cập trực tiếp các ID tìm thấy từ chỉ mục cục bộ: QuReC, CWP-Net, CPLIR, DiTTo, CURE, RAR, PreActIR.

### 1.3. Sổ nguồn — chỉ ghi điều đã kiểm tra

| ID | Nguồn và đường dẫn | Đã xem | Điều có thể dùng |
|---|---|---|---|
| S01 | PromptIR, https://arxiv.org/abs/2306.13090 ; repo cục bộ | README + mã PromptGenBlock | Prompt weights lấy từ spatial mean của feature; sau đó trộn prompt có cấu trúc không gian. Không được diễn đạt thành toàn bộ mạng chỉ dùng thông tin global. |
| S02 | AdaIR, https://github.com/c-yn/AdaIR | README cục bộ | Khai thác/modulate low/high frequency theo ảnh; đã có hướng frequency-adaptive AiOIR. |
| S03 | BaryIR, https://arxiv.org/abs/2602.23169 ; https://github.com/xl-tang3/BaryIR | README cục bộ | Tác giả mô tả WB content space và residual degradation spaces, báo cáo OOD generalization. Chưa kiểm chứng lý thuyết hoặc con số. |
| S04 | CWP-Net, https://arxiv.org/abs/2603.03839 | PDF cục bộ + abstract trực tuyến | Đã nêu semantic–degradation spurious correlation, balanced test và wavelet deconfounding. Abstract ghi accepted IEEE TIP 2026. |
| S05 | QuReC, https://arxiv.org/abs/2607.15097 | PDF cục bộ + abstract trực tuyến | Query-wise prototype guidance, local-global calibration, CDD11 và một thử nghiệm unseen noise. Abstract ghi ACM MM 2026. |
| S06 | CPLIR, https://arxiv.org/abs/2504.09973 ; https://github.com/Aitical/CPLIR | README cục bộ + abstract trực tuyến v3 | CPR tối ưu tương tác prompt–restoration, dùng negative prompt khác loại. Không thể nhận là mới chỉ vì dùng restoration outcome thay classification. |
| S07 | DIL, https://arxiv.org/abs/2303.06859 | Abstract từ arXiv API | Counterfactual distortion augmentation và meta-learning interventions cho unseen distortion; API comment ghi CVPR 2023. Chưa đọc full text. |
| S08 | BiDeMem, https://arxiv.org/abs/2606.28112 ; https://arxiv.org/html/2606.28112v1 | Abstract API + HTML các mục phương pháp/thí nghiệm | Shared memory slot cho restoration/forward degradation; wrong/native prior, shuffle và slot intervention controls. Không được nhận là mới phép thay prior hoặc forward-consistency tự thân. Chưa tái lập. |
| S09 | BSRGAN, https://arxiv.org/abs/2103.14006 ; https://github.com/cszn/BSRGAN | README cục bộ | Randomly shuffled degradation synthesis đã có; README nêu ICCV 2021. |
| S10 | DiTTo, https://arxiv.org/abs/2605.30915 | Abstract trực tuyến/API | Chọn thứ tự restoration actions bằng agent/simulator. Đây là thứ tự hành động phục hồi, không tự động đồng nhất với lịch sử sinh ảnh. Chưa đọc full text. |
| S11 | CURE, https://arxiv.org/abs/2607.03044 | Abstract trực tuyến/API | Identity, ratio control, intermediate single-factor removal, permutation-invariant restoration loss. Abstract ghi ICPR 2026. Chưa đọc full text; không kết luận CURE giả định forward operators giao hoán. |
| S12 | RAR, https://arxiv.org/abs/2603.26385 | Abstract trực tuyến | Unified latent IQA + iterative restoration, accepted CVPR 2026 theo abstract. Không mới nếu chỉ assess–restore–repeat. |
| S13 | PreActIR, https://arxiv.org/abs/2610.20384 | Abstract trực tuyến | Dự đoán hậu quả tool trước khi chạy, risk/uncertainty và verifier; ghi under review. Không mới nếu chỉ dự đoán restoration utility. |
| S14 | OPIR, https://arxiv.org/abs/2601.10192 | PDF cục bộ | Predictive inverse kernels, magnitude-derived uncertainty, two-stage refinement. Chưa xác minh checkpoint hoặc venue. |
| S15 | D3Net, https://arxiv.org/abs/2502.19068 | PDF cục bộ v2 | Frequency/spatial prompts và dynamic decomposition; không tái lập các gain được báo cáo. |
| S16 | MiOIR, https://arxiv.org/abs/2401.03379 | README cục bộ | Sequential task learning và prompts đã có. Sequential TRAINING không phải thứ tự sinh suy thoái hoặc thứ tự phục hồi lúc test. |

### 1.4. Những giả định cần thách thức

**Benchmark:**

- [SUY LUẬN] Khi mỗi degradation gắn một dataset riêng, content/domain và task cùng thay đổi. Task clustering/t-SNE có thể phản ánh nguồn dữ liệu, chưa phải degradation mechanism.
- [SUY LUẬN] Mixed-degradation test được train trên đúng các combination đó không chứng minh compositional OOD. Phải tách seen mixtures, held-out combinations, severity và simulator shift.
- [SUY LUẬN] Trung bình 3 noise levels + rain + haze cho denoising trọng số 3/5. Báo cáo cả macro theo TASK và mọi cell, không chỉ số trung bình thuận lợi.
- [THIẾT KẾ] Khóa RGB/Y, data_range, clipping, crop/border, SSIM implementation, resolution và seed. Không so PSNR khác protocol. Bootstrap theo ảnh gốc, không coi nhiều crop/noise realizations là mẫu độc lập.
- [SUY LUẬN] Synthetic gamma không tương đương camera low-light; synthetic haze với constant transmission không thay thế depth-dependent haze. Toy diagnostic chỉ kiểm tra cơ chế đã định nghĩa.
- [THIẾT KẾ] Split theo clean image/scene TRƯỚC khi sinh mọi biến thể; ảnh cùng scene của GoPro phải đi cùng split. Hash/de-duplicate trước đánh giá.

**Cơ chế trong bài:**

- [BÀI CHỨNG MINH/BÁO CÁO] CWP-Net có thống kê bias và thử balanced synthetic test. [SUY LUẬN] Attention map trông như vùng suy thoái không chứng minh độc lập với semantics. Wavelet decomposition cũng không bảo đảm content/degradation độc lập. Balanced test của bài thay cả scene và cách mô phỏng; cần factorial control để tách simulator shift khỏi semantic shift.
- [SUY LUẬN] CWP-Net cần nối rõ đồ thị sinh dữ liệu với đồ thị tính toán: prompted feature thực thi được suy ra từ degraded input; tính đúng của adjustment trong một SCM giả định không tự động xác nhận SCM đó hoặc causal interpretation của code. Đây là câu hỏi kiểm tra, chưa phải kết luận chứng minh bài sai.
- [BÀI CHỨNG MINH/BÁO CÁO] QuReC softmax matching trên prototype bank và minh họa spatial responses. [SUY LUẬN] Heatmap khác nhau chưa chứng minh mỗi response là severity vật lý; sum-to-one matching không trực tiếp biểu diễn nhiều severity độc lập. Các feature/decoder vẫn có thể giải mixed degradation, nên không suy ra bất khả thi.
- [BÀI CHỨNG MINH/BÁO CÁO] OPIR Eq.11 lấy mean absolute kernel weights làm uncertainty. [SUY LUẬN] Đây chưa là calibrated probability of error. Cần reliability/risk curves và so với gradient/texture controls trước gọi là uncertainty đáng tin cậy.
- [QUAN SÁT CỤC BỘ] D3Net Eq.3 ghi frequency centering bằng nhân phase vào phổ. [SUY LUẬN] Nhân số phức modulus 1 không thay đổi amplitude, nên biểu thức này không thực hiện fftshift amplitude như mô tả; chưa xem implementation để biết là lỗi công thức hay thực thi.
- [QUAN SÁT CỤC BỘ] D3Net cấu hình attention ở Eq.6–7 khác cấu hình f được chọn trong Table VI. Cần đối chiếu code trước tái lập, không xem các gain lớn là bằng chứng cơ chế đã chắc chắn.
- [SUY LUẬN] CPLIR/BiDeMem làm wrong prompt gây giảm chất lượng có thể xác nhận conditioning có ảnh hưởng; chưa đủ chứng minh mã chỉ chứa degradation hoặc có thể chuyển sang ảnh khác. Training ép wrong-prior gap còn có thể làm metric đó tăng mà không làm native restoration tốt hơn.

## 2. Hướng A — bảo thủ hơn: điều kiện suy thoái có khả năng chuyển nội dung

Tên làm việc: **Cross-content transportability of degradation conditioning**. Không phải tên mô hình đã chốt; chưa khẳng định mới.

### A(a). Hiện tượng cần xác nhận

[PHỎNG ĐOÁN H-A1] Hai ảnh khác nội dung nhưng có cùng global degradation operator và severity nhận condition khác nhau đến mức khi hoán đổi condition, chất lượng giảm đáng kể. Một phần condition đang phục vụ shortcut/content-specific correction thay vì adaptation theo operator.

Điều quan trọng: giảm PSNR khi swap CHƯA chứng minh shortcut có hại. Condition có thể chứa uncertainty hoặc content cues hợp lệ. Chỉ đáng sửa nếu ràng buộc transport tăng native restoration trên held-out content/severity.

### A(b). Cơ chế và cách hiện thực tối thiểu

[THIẾT KẾ] Sinh factorial quadruplet y_i=D_theta(x_i), y_j=D_theta(x_j), y'_i=D_phi(x_i), y'_j=D_phi(x_j), với i != j và theta != phi. Theta gồm operator, severity; realization noise độc lập, không copy noise pattern giữa ảnh. Lúc đầu chỉ dùng global noise/blur/JPEG, tránh giả định sai về depth/spatial alignment.

Với condition encoder E và restoration R:

    x_native = R(y_i, E(y_i))
    x_same   = R(y_i, E(y_j))       # cùng theta, khác nội dung
    x_other  = R(y_i, E(y'_j))      # khác theta, khác nội dung

[SUY LUẬN] Nếu condition chỉ dùng để chọn inverse strategy cho global theta, same-theta transfer đáng lẽ ít gây hại, trong khi backbone vẫn nhận y_i để xử lý texture/content. Đây là giả định có thể sai, không phải định lý chung về restoration.

[THIẾT KẾ] Cách học đầu tiên: native reconstruction + lambda * same-theta cross-content reconstruction. Không thêm decoder hay memory mới. So với embedding-invariance loss và matched-data augmentation để biết gain đến từ đâu. Không ép mọi wrong-type prompt gây hỏng nếu hai cấu hình có tác dụng phục hồi tương đương.

Phép tương tự: transportability và controlled interventions trong causal representation learning. Không gọi swapping latent tự thân là can thiệp vật lý hoặc tuyên bố nhận dạng SCM từ quan sát.

### A(c). Bác bỏ

- H-A1 bị yếu đi nếu same-theta swap gap <0.05 dB ở mọi nhóm kiểm tra và native condition vẫn có tác dụng (không collapse).
- Giả thuyết phương pháp thất bại nếu chỉ giảm swap gap nhưng native OOD không tăng, hoặc zero/mean condition cũng đạt như vậy.
- Nếu matched factorial augmentation không có transport loss đạt bằng phương pháp, không có bằng chứng cho cơ chế loss mới.
- Nếu native OOD tăng nhưng in-distribution giảm >0.10 dB, cần báo trade-off; không gọi là cải thiện đồng đều.
- Kết quả chỉ trên synthetic blur/JPEG không đủ kết luận real-world AiOIR; phải kiểm tra lại với natural/mixed benchmarks khi qua pilot.

### A(d). Khác biệt dự kiến, gần nhất và rủi ro tính mới

- CWP-Net [S04]: cùng vấn đề semantic bias; đề xuất của ta yêu cầu operator-controlled cross-content transfer và native OOD benefit, không suy independence từ wavelet/attention.
- CPLIR [S06]: đã regularize restoration outcomes với mismatched prompts. Ta thử positive transfer giữa hai ảnh cùng theta, không chỉ tăng khoảng cách wrong/correct.
- BiDeMem [S08]: đã wrong/native prior intervention và forward-degradation explanation. Ta cần thêm matched-operator, different-content transfer và đối chứng cho prompt collapse; đổi prior nói chung không mới.
- DIL [S07], BaryIR [S03]: đã invariant representation/generalization. Phải đọc full text/appendix và tìm latent swapping/degradation consistency trước khẳng định novelty.

[SUY LUẬN] Chỉ thêm same-theta swap loss rất có thể là cải tiến gia tăng. Hướng chỉ có sức nặng lớn nếu phát hiện một failure phổ biến, phân biệt content cần thiết với nuisance condition, và cho thấy restoration/OOD benefit vượt matched-data controls trên ít nhất hai backbone. Hiện tại: rủi ro trùng lặp cao, khả thi Colab tương đối tốt.

### A — thí nghiệm tối thiểu E-A

**Dataset và split:** DIV2K train 800 chia cố định 700 train/100 development; DIV2K validation 100 và Kodak24 làm test khóa. Có thể dùng CBSD68 làm external test sau kiểm tra overlap/pretraining exposure. Không dùng test để tune. Nguồn DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/ ; Kodak: https://r0k.us/graphics/kodak/ . Chưa tải dữ liệu.

**Simulator pilot:** AWGN sigma in [5,50]/255; Gaussian blur sigma in [0.4,2.0] pixel với kernel support/padding cố định và ghi config; JPEG quality [30,90]. Giữ same resolution. Ba SINGLE tasks train cân bằng; test single, held-out noise+blur/JPEG combinations và severity ngoài khoảng đã định nghĩa trước. Mixed tests phải báo riêng. JPEG codec/version cố định. Đây là synthetic diagnostic AiOIR, không thay thế AIO-3 rain/haze benchmark.

**Audit rẻ:** PromptIR checkpoint chỉ audit task được train, ban đầu AWGN; không gán thất bại blur/JPEG OOD cho semantic shortcut. Mã PromptGenBlock cho phép thay prompt weights từ donor; giữ prompt bank/feature recipient cố định. Phép audit này phải ghi rõ thay weights ở layer nào. So same-theta donor, other-theta donor, native, mean và zero. Model self-generated prompt ở nhiều layer nên audit từng layer và tất cả, tránh gán nhầm nguyên nhân.

**Controlled training:** NAFNet width16, enc=[1,1,1,4], middle=1, dec=[1,1,1,1], một global condition và affine modulation tối thiểu dùng chung cho mọi variant. Số tham số đo thực tế. Model nhỏ là vật đo cơ chế, không đem so bảng SOTA như NAFNet đầy đủ. Khi pass, xác nhận trên PromptIR fine-tune với đầy đủ task/data phù hợp.

**Sáu variant cùng data/steps:** (1) native-only; (2) native-only factorial sampler, cùng số ảnh recipient/donor và forward budget; (3) condition embedding consistency; (4) cross-content reconstruction; (5) variant4 nhưng donor theta ngẫu nhiên; (6) variant4 dùng mean/constant condition. Variant5 là negative control, không phương pháp khuyến nghị. Với variant không dùng donor, dùng thêm native passes để khớp sample exposure/compute.

**Training:** 15k optimizer updates/variant, batch4, 128x128, mixed precision, 3 seeds cho xác nhận. Pilot 1 seed/ít variant được phép để dừng sớm nhưng không xem là kết luận. Optimizer/LR và validation checkpoint rule khóa trước so sánh.

**Metrics:** native PSNR RGB, SSIM; LPIPS supplementary; task/group macro; worst severity/domain cell; transport gap G_same=PSNR_native-PSNR_same; sensitivity G_other; zero/mean-code gap; latency và peak VRAM. G_other lớn tự nó không là thành công. Bootstrap theo ảnh gốc và báo phân tán qua seeds.

**Go threshold đề xuất:** trên >=2 model audit phù hợp task, G_same >=0.20 dB với paired 95% CI lower bound >0; sau training, native held-out-content/severity macro gain >=0.20 dB so với strongest matched control, 95% CI lower bound >0 và mean gain dương ở cả 3 seeds; ID macro loss <=0.10 dB; G_same giảm >=50% mà condition không collapse. Không đạt audit threshold thì không bắt đầu dự án loss trên cơ sở failure này. Không đạt native benefit thì bác bỏ phương pháp dù transport metric đẹp.

**Compute [ƯỚC TÍNH]:** audit 1–3 T4 GPU-h tùy số ảnh/checkpoint; controlled 6 variants * 3 seeds * 15k updates * (0.2–0.8 s/update) /3600 =15–60 GPU-h, cộng 3–8 h evaluation ≈18–68 h. Đây là giả định throughput, chưa đo; một update bao gồm mọi forward của loss. Pilot một seed 2 variants ≈2–7 h cộng setup/eval. Chưa tính xác nhận PromptIR/AIO benchmark đầy đủ; cần lập ngân sách riêng sau khi đo. Không hứa hoàn tất trên quota Colab miễn phí.

## 3. Hướng B — mạo hiểm hơn, ưu tiên câu hỏi mới: tính nhận dạng có liên quan quyết định

Tên làm việc: **Identifiability-aware restoration under uncertain degradation histories**. Chưa khẳng định mới.

### B(a). Hiện tượng cần xác nhận

[PHỎNG ĐOÁN H-B1] Exact degradation-history labels có thể vừa không nhận dạng được từ ảnh, vừa bị dùng làm conditioning/routing quá cứng. Một bộ phận lỗi là hard commitment vào history khi bằng chứng yếu; bộ phận khác không cần history vì các inverse decisions gần tương đương.

Phân biệt ba khái niệm:

1. Forward history/order tạo ra y.
2. Order các tool/steps dùng để khôi phục y.
3. Thông tin về history có làm giảm restoration risk hay không.

[SUY LUẬN] Cả ba không tương đương. Forward non-commutativity không tự động chứng minh blind order identifiable hoặc inverse order tối ưu là reverse order.

### B(b). Cơ chế toán học và hướng xử lý

Ví dụ 1 — observable cue trong mô hình lý tưởng:

    y_A = Bx + n,       Cov(n) = sigma^2 I
    y_B = B(x+n),       Cov(Bn) = sigma^2 B B^T

[SUY LUẬN] Với B là nontrivial linear blur, không clipping/quantization, blur trước noise và blur sau noise tạo noise covariance khác. Cues có thể bị texture, unknown B/sigma và ISP che lấp. Chưa đo mức độ phân biệt thực tế.

Ví dụ 2 — exact non-identifiability control:

    L_a(x)=a*x; N_sigma(x)=x+sigma*epsilon; epsilon~N(0,I)
    L_a(N_sigma(x)) = a*x+a*sigma*epsilon
    N_(a*sigma)(L_a(x)) = a*x+a*sigma*epsilon

[SUY LUẬN — đẳng thức trực tiếp] Hai history có cùng conditional law y|x sau reparameterization noise. Nếu paired sampling có cùng prior của a và effective noise, same clean prior, không intermediate clipping/quantization, thì joint law (x,y) bằng nhau. Với history prior cân bằng, từ y đơn lẻ không có classifier tốt hơn chance trong kỳ vọng. Đây là sanity control; không phải kết quả mới về AiOIR.

[SUY LUẬN] Trong ví dụ tương đương này, optimum inverse dưới squared loss cũng giống nhau; nhận dạng order là mục tiêu dư thừa. Không được dùng nó để tuyên bố ambiguity gây restoration error. Cần đo đồng thời OBSERVABILITY và DECISION REGRET ở các trường hợp không tương đương.

[THIẾT KẾ] Chia evaluation thành: (i) distinguishable histories + inverse differences; (ii) overlapping evidence + inverse differences; (iii) distributionally equivalent histories + inverse equivalence. Báo risk/cost riêng, không gộp thành một accuracy.

[PHỎNG ĐOÁN H-B2] Học trên classes tương đương vật lý và marginalize khi history còn ambiguous sẽ tránh overconfident conditioning, cải thiện risk trên unseen order/severity so với hard routing.

Candidate estimator để kiểm tra nguyên lý, với squared loss:

    x_hat = sum_h q(h|y) * R(y,h)

Chỉ có ý nghĩa Bayes nếu q là posterior phù hợp và R gần conditional mean. Với nonlinear perceptual loss không có bảo đảm tương tự. Physical equivalence: hai history cùng p(y|x) thì collapse vào cùng class; gần tương đương phải định nghĩa tolerance/loss và kiểm tra, không gộp theo trực giác.

Phép tương tự: system identification/observability và Bayesian decision theory. Điểm trung tâm là xác định phần history cần biết để quyết định, không phải thêm uncertainty head.

### B(c). Bác bỏ — baseline nguy hiểm nhất

- **Joint blind regressor train trên random-order mixtures** cùng backbone/data/compute: đã có khả năng học E[x|y] trực tiếp. Nếu nó ngang hoặc tốt hơn proposed estimator, không có bằng chứng cho explicit history modeling.
- True-history oracle improvement <0.10 dB trên mọi meaningful test cell: history không đáng học trong setting đó; dừng đề tài model theo history.
- Soft averaging chỉ hơn hard routing nhưng không hơn equal-compute uniform ensemble/joint model: chỉ ensemble/calibration benefit, chưa chứng minh nguyên lý mới.
- Tăng order accuracy nhưng không giảm restoration regret: sai mục tiêu.
- Exact-equivalence control cho classifier >55% ổn định: kiểm tra leakage, prior mismatch, clipping, codec hoặc simulator artifacts; không tuyên bố phá giới hạn nhận dạng.
- Noncommutative pairs vẫn có thể không cần order-conditioned model. Không dùng khác biệt ảnh degraded làm proxy cho restoration benefit.

### B(d). Khác biệt với gần nhất và kiểm tra còn thiếu

- BSRGAN [S09] đã shuffle forward degradations; random order augmentation tự thân không mới.
- DiTTo [S10] tối ưu action scheduling với tool pool; ta hỏi identifiability của forward history và value-of-history trong một shared restorer. Chưa đọc full text nên chưa loại trừ có analysis tương tự.
- CURE [S11] khuyến khích restoration permutation invariance; ta không mặc định mọi forward history khác nhau phải được phân biệt, hoặc mọi inverse order phải bất biến. Phải đọc full protocol của CURE trước nói đối lập.
- OPIR [S14] dự đoán inverse kernel/uncertainty, không đủ bằng chứng từ PDF đã đọc rằng giải decision-relevant history equivalence.
- RAR/PreActIR [S12–13] đã assess/forecast action quality; utility prediction hoặc uncertainty đơn thuần không là điểm mới ở đây.

[SUY LUẬN] Tiềm năng đóng góp: formalize observable history classes + benchmark tách identifiability khỏi value-of-information + shared restoration rule chỉ dùng phần phân biệt có ích. Chưa có bằng chứng đủ để tuyên bố novel. Cần thêm literature về blind inverse identifiability, camera pipeline forensics, noise covariance và Bayesian model averaging; các nền tảng này không mới.

### B — thí nghiệm tối thiểu E-B

**Data:** dùng cùng DIV2K split 700/100 và held-out validation100 + Kodak24 như E-A để tiết kiệm. Xuất simulator manifest gồm source ID, pipeline, effective severity, seed, clipping stage. Chưa tải/chạy.

**Families:**

1. Blur–AWGN ở cả hai order; grid blur sigma {0.6,1.2,1.8}, noise sigma {10,25,40}/255; train hai order cân bằng, hold out một phần cross-grid, khóa test ngoài grid. Không chỉ train một order rồi coi test order đảo là attribution về identifiability.
2. Global exposure–AWGN equivalence control, a in {0.4,0.7}, effective sigma matched. Phân tích chính float/no intermediate clipping, scoring unclipped và clipped output báo riêng; final clipping giống nhau vẫn giữ equivalence, intermediate clipping có thể phá nó.
3. JPEG–noise hai order, q in {40,70,90}; đây là test nonlinear/codec robustness, đo matched input severity và chất lượng theo cell. Không ép input PSNR bằng nhau rồi coi nuisance đã được kiểm soát hoàn toàn.

**Baselines:** (1) joint random-order blind NAFNet-small; (2) cùng shared backbone có history condition + hard predicted routing; (3) cùng model với soft posterior averaging và equivalence treatment; (4) uniform averaging cùng số calls; (5) true-history conditional oracle (privileged diagnostic, không deployable baseline). Train controls trên cùng source data, epochs/updates, capacity gần bằng nhau. Variant2/3/4 dùng chung conditional backbone khi so inference để tách training effects; thêm exact-label CE vs equivalence-aware objective để tách đóng góp supervision.

**Compute fairness:** averaging 2 histories =2 backbone calls. Báo riêng single-pass curve và equal-2-call curve (joint model được self-ensemble hoặc capacity/step budget tương đương), latency/VRAM thật. Không gọi multi-pass nhanh hơn chỉ vì chung weights. Identity/no-op family và single degradations cùng train để duy trì AiOIR phạm vi nhỏ.

**Metrics:** RGB PSNR, SSIM, LPIPS supplementary; per-order/task/severity cells; worst-cell PSNR; history balanced accuracy/AUROC, NLL, Brier và calibration curves; restoration regret tương đối oracle; gain relative joint baseline; runtime. Empirical classifier near chance không chứng minh information-theoretic impossibility ngoài exact-equivalence case đã chứng minh.

**Ablations bắt buộc:** exact history labels vs equivalence classes; hard vs soft vs uniform; single order vs balanced random orders; matched vs mismatched priors; same/independent noise seeds; no intermediate clipping vs realistic ISP; raw noise covariance statistic vs learned order predictor. Dùng paired seeds chỉ cho diagnostic coupling, không lộ seed cho model hoặc chia biến thể cùng ảnh sang train/test.

**Training budget:** 20k updates/model cho pilot xác nhận, batch4 crop128, 3 seeds ở stage xác nhận. Calibration chỉ fit development set. Chưa chọn final posterior/equivalence implementation; cần chứng minh cell có value-of-history trước.

**Go/no-go đề xuất:** oracle history gain >=0.30 dB trên ít nhất hai predeclared family/severity groups; hard routing có excess MSE đáng kể ở vùng uncertain. Phương pháp phải cải thiện macro PSNR >=0.20 dB và worst-cell >=0.30 dB so với strongest equal-compute joint/uniform control, macro paired 95% CI lower bound >0, mean gain dương cả 3 seeds. Ở equivalence control, classifier 45–55% và Brier gần 0.25 cho balanced binary case; không hy sinh reconstruction để làm confidence đẹp. Nếu oracle gain <0.10 dB mọi cell hoặc joint baseline đóng gap tới <0.10 dB, dừng phương pháp history-aware trong setting này. Vùng 0.10–0.30 dB là inconclusive, không đổi threshold sau khi thấy test.

**Compute [ƯỚC TÍNH]:** analytic/simulator checks CPU; pilot oracle-vs-joint 2 runs x20k x0.2–0.8s ≈2.2–8.9 T4 GPU-h, thêm ~1–2h eval. Có thể mở đầu 5k updates để smoke-test nhưng undertrained negative không đủ bác bỏ. Xác nhận tối đa 5 variants x3 seeds x20k x0.2–0.8s ≈16.7–66.7h +4–8h eval ≈21–75 GPU-h. Đây là giả định bao gồm multi-forward trong một update; phải đo lại 100 warmup+500 timed updates trên GPU thực. Chưa tính full-scale standard AIO-3/AIO-5/real-world confirmation hoặc phân tích paper-level nhiều simulator.

## 4. Các hướng bị hạ mức sau tra cứu

- Wavelet/frequency + prompt + MoE: chưa có failure và mechanism riêng; xem là incremental nếu chỉ ghép.
- Content/degradation disentanglement nói chung: đã rất gần CWP-Net, BaryIR, DIL; không nhận là mới.
- Spatial prompt/local routing: QuReC và nhiều công trình khác đã có; không nhận là mới.
- Wrong-prompt contrastive loss / bidirectional explanation nói chung: CPLIR và BiDeMem đã trực tiếp làm.
- Iterative IQA feedback / predict action utility: RAR và PreActIR đã làm.
- Random-order synthesis hoặc tìm best restoration order: BSRGAN/DiTTo đã có.
- Response signatures từ perturbing đầu vào: chưa chọn làm hướng. [SUY LUẬN] deterministic probes không thêm thông tin quan sát về history không nhận dạng được; Jacobian phản ánh cả backbone/content. Chưa có cơ chế đủ chắc để ưu tiên.

## 5. Quyết định tạm thời và lộ trình

[SUY LUẬN / KHUYẾN NGHỊ] Ưu tiên QUESTION B để tìm contribution có thể khác bản chất; dùng E-B oracle-vs-joint như cửa kiểm tra đầu tiên, không chốt kiến trúc. A là nhánh bảo thủ hơn về thực thi nhưng rủi ro novelty lớn; bắt đầu bằng audit checkpoint, chỉ phát triển loss nếu native OOD benefit vượt matched controls.

Không train cả hai ma trận xác nhận ngay. Pilot đầu tiên giới hạn một nhánh, đọc full text nearest works trước gán nhãn novel. Một kết quả âm đáng tin cũng được ghi lại, không đổi narrative để giữ ý tưởng.

Điều kiện để có câu chuyện hội nghị top: failure tái lập trên >1 backbone, phân tích giải thích đúng nơi phương pháp giúp/không giúp, đối chứng mạnh loại data/compute effects, và xác nhận ngoài toy simulator. Gain 0.2 dB là decision threshold pilot, không phải cam kết đủ publish top conference.

### Việc cần làm vòng R1

- [ ] Xác nhận Colab Free/Pro, GPU thường nhận, quota/tuần, ngân sách và deadline.
- [ ] Chốt B-pilot hoặc A-audit; hiện khuyến nghị B-pilot.
- [ ] Đọc full text/appendix DIL, CPLIR, BaryIR, DiTTo, CURE; tìm đúng cross-content swaps và identifiability/equivalence prior art.
- [ ] Kiểm tra download weights/data, dependencies và licenses của nguồn thực sự dùng; lưu version/hash phục vụ tái lập.
- [ ] Tạo manifest split theo clean image và benchmark config trước khi sinh degradation.
- [ ] Nếu B: xác minh exact-equivalence simulator; chạy oracle-vs-joint trước posterior head.
- [ ] Nếu A: hook prompt weights và audit native/same-theta/other-theta/mean/zero với checkpoint task-aligned.
- [ ] Profile throughput/VRAM; thay ước tính GPU-h bằng đo thực tế.
- [ ] Lưu metrics từng ảnh, seeds, config, failure examples và kết luận go/no-go vào log.

## 6. Bảng kết quả thực nghiệm của dự án

**Chưa có kết quả thực nghiệm.** Không có training run, checkpoint mới, PSNR/SSIM/LPIPS hoặc GPU timing nào được tạo trong R0. Các số ở trên chỉ là ngưỡng và chi phí dự kiến. Những đẳng thức trong B(b) là suy luận toán học dưới giả định ghi rõ, chưa phải discovery thực nghiệm.

| Run ID | Hypothesis | Config / data hash | Seed / GPU | Metrics | Kết luận |
|---|---|---|---|---|---|
| Chưa chạy | — | — | — | — | R0 chỉ phân tích và thiết kế |

### Mẫu ghi vòng sau

    Ngày / vòng / run:
    Giả thuyết và dự đoán trước thí nghiệm:
    Nguồn bằng chứng và mức [BÀI]/[SUY LUẬN]/[PHỎNG ĐOÁN]:
    Dataset split / hashes / simulator / protocol metric:
    Baselines / matched compute / ablations:
    GPU, thời gian đo và chi phí:
    Kết quả từng nhóm + khoảng tin cậy + artifacts:
    Quan sát có thể bác bỏ cơ chế:
    Kết luận giữ/sửa/bỏ và lý do:
    Nearest works đã kiểm tra / câu hỏi còn mở:
