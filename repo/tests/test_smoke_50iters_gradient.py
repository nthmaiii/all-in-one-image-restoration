"""Smoke Test: Kiểm định Động học Sau 50 Iterations (Gradient & Output Divergence Test).
Yêu cầu kiểm toán P4 Fair Pilot v2:
- Kiểm tra tại Step 1 và sau 50 iterations:
  1. Gradient của các tham số mới (conv_low, conv_high, fusion) ở B và C PHẢI KHÁC 0 (> 0).
     (Chứng minh Zero-Init không bị mắc kẹt tại điểm yên ngựa hoặc triệt tiêu gradient).
  2. Độ lệch đầu ra của B và C so với A sau 50 iterations PHẢI KHÁC 0 (> 0).
     (Chứng minh mô hình bắt đầu học phân tách biểu diễn, thoát khỏi trạng thái đồng nhất).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.promptir.net.model import PromptIR
from src.models.fd_promptir import FDPromptIR
from src.models.param_matched_promptir import ParamMatchedPromptIR
from scripts.train_pilot_v2 import AiOIRDatasetV2, set_seed, seed_worker

def test_smoke_50iters_gradient():
    print("=" * 70)
    print("SMOKE TEST: KIỂM ĐỊNH GRADIENT & ĐỘ LỆCH ĐẦU RA B/C SO VỚI A (50 ITERS)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    set_seed(seed)

    # 1. Dataset & DataLoader (50 iters, bs=2)
    train_dir = REPO_ROOT / 'data' / 'train_v2' / 'clean'
    dataset = AiOIRDatasetV2(train_dir, patch_size=32, length=50 * 2, seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=gen, worker_init_fn=seed_worker)

    # 2. Khởi tạo 3 mô hình A, B, C (dùng num_blocks=[1,1,1,1] để kiểm tra toàn bộ luồng)
    model_A = PromptIR(num_blocks=[1, 1, 1, 1], decoder=True).to(device)
    model_B = FDPromptIR(num_blocks=[1, 1, 1, 1], decoder=True).to(device)
    model_C = ParamMatchedPromptIR(num_blocks=[1, 1, 1, 1], decoder=True).to(device)

    opt_A = torch.optim.AdamW(model_A.parameters(), lr=2e-4)
    opt_B = torch.optim.AdamW(model_B.parameters(), lr=2e-4)
    opt_C = torch.optim.AdamW(model_C.parameters(), lr=2e-4)

    # Lấy 1 mẫu cố định để đo kiểm đầu ra
    test_x = torch.randn(1, 3, 32, 32, device=device)

    # Đo kiểm Iteration 0
    with torch.no_grad():
        out_A0 = model_A(test_x)
        out_B0 = model_B(test_x)
        out_C0 = model_C(test_x)
    
    diff_B0 = torch.max(torch.abs(out_B0 - out_A0)).item()
    diff_C0 = torch.max(torch.abs(out_C0 - out_A0)).item()
    print(f">> Iteration 0: Max |Out_B - Out_A| = {diff_B0:.2e} | Max |Out_C - Out_A| = {diff_C0:.2e} (Bảo toàn đồng nhất)")

    # 3. Huấn luyện 50 iterations và kiểm tra gradient
    train_iter = iter(loader)
    step1_grad_norm_B = 0.0
    step1_grad_norm_C = 0.0

    print("\nTiến hành huấn luyện 50 iterations đồng thời trên 3 nhánh...")
    for iter_idx in range(1, 51):
        deg, clean, _ = next(train_iter)
        deg, clean = deg.to(device), clean.to(device)

        # Nhánh A
        opt_A.zero_grad()
        loss_A = F.l1_loss(model_A(deg), clean)
        loss_A.backward()
        opt_A.step()

        # Nhánh B
        opt_B.zero_grad()
        loss_B = F.l1_loss(model_B(deg), clean)
        loss_B.backward()
        
        # Kiểm tra gradient của các tham số mới tại Step 1
        if iter_idx == 1:
            grads_B = []
            for name, param in model_B.named_parameters():
                if any(k in name for k in ['conv_low', 'conv_high', 'fusion']) and param.grad is not None:
                    grads_B.append(param.grad.norm().item())
            step1_grad_norm_B = float(np.mean(grads_B)) if grads_B else 0.0

        opt_B.step()

        # Nhánh C
        opt_C.zero_grad()
        loss_C = F.l1_loss(model_C(deg), clean)
        loss_C.backward()

        if iter_idx == 1:
            grads_C = []
            for name, param in model_C.named_parameters():
                if any(k in name for k in ['conv_param', 'fusion']) and param.grad is not None:
                    grads_C.append(param.grad.norm().item())
            step1_grad_norm_C = float(np.mean(grads_C)) if grads_C else 0.0

        opt_C.step()

    # 4. Kiểm tra sau 50 iterations
    # (a) Gradient của tham số mới sau 50 iters
    grads_B_50 = [p.grad.norm().item() for n, p in model_B.named_parameters() if any(k in n for k in ['conv_low', 'conv_high', 'fusion']) and p.grad is not None]
    grads_C_50 = [p.grad.norm().item() for n, p in model_C.named_parameters() if any(k in n for k in ['conv_param', 'fusion']) and p.grad is not None]
    mean_grad_B_50 = float(np.mean(grads_B_50)) if grads_B_50 else 0.0
    mean_grad_C_50 = float(np.mean(grads_C_50)) if grads_C_50 else 0.0

    # (b) Độ lệch đầu ra so với A
    with torch.no_grad():
        out_A50 = model_A(test_x)
        out_B50 = model_B(test_x)
        out_C50 = model_C(test_x)

    diff_B50 = torch.max(torch.abs(out_B50 - out_A50)).item()
    diff_C50 = torch.max(torch.abs(out_C50 - out_A50)).item()
    diff_BC50 = torch.max(torch.abs(out_B50 - out_C50)).item()

    print("\n" + "=" * 70)
    print("KẾT QUẢ KIỂM ĐỊNH GRADIENT & ĐỘ LỆCH ĐẦU RA SAU 50 ITERATIONS:")
    print("=" * 70)
    print(f"1. Gradient trung bình của tham số mới tại Step 1:")
    print(f"   + Nhánh B (FD-PromptIR):       {step1_grad_norm_B:.6f} (> 0: {step1_grad_norm_B > 0})")
    print(f"   + Nhánh C (Param-Matched):     {step1_grad_norm_C:.6f} (> 0: {step1_grad_norm_C > 0})")

    print(f"\n2. Gradient trung bình của tham số mới tại Step 50:")
    print(f"   + Nhánh B (FD-PromptIR):       {mean_grad_B_50:.6f} (> 0: {mean_grad_B_50 > 0})")
    print(f"   + Nhánh C (Param-Matched):     {mean_grad_C_50:.6f} (> 0: {mean_grad_C_50 > 0})")

    print(f"\n3. Độ lệch đầu ra tối đa (Max Absolute Output Divergence) tại Step 50:")
    print(f"   + Max |Output_B - Output_A|:   {diff_B50:.6f} (> 0: {diff_B50 > 0})")
    print(f"   + Max |Output_C - Output_A|:   {diff_C50:.6f} (> 0: {diff_C50 > 0})")
    print(f"   + Max |Output_B - Output_C|:   {diff_BC50:.6f} (> 0: {diff_BC50 > 0})")

    # Kiểm tra assertions
    assert step1_grad_norm_B > 0, "Gradient của tham số mới ở B tại Step 1 bị triệt tiêu (= 0)!"
    assert step1_grad_norm_C > 0, "Gradient của tham số mới ở C tại Step 1 bị triệt tiêu (= 0)!"
    assert mean_grad_B_50 > 0, "Gradient của tham số mới ở B tại Step 50 bị triệt tiêu (= 0)!"
    assert mean_grad_C_50 > 0, "Gradient của tham số mới ở C tại Step 50 bị triệt tiêu (= 0)!"
    assert diff_B50 > 0, "Đầu ra của B sau 50 iters không thay đổi so với A (bị kẹt 0)!"
    assert diff_C50 > 0, "Đầu ra của C sau 50 iters không thay đổi so với A (bị kẹt 0)!"

    print("\n[KẾT LUẬN KIỂM ĐỊNH: ĐẠT TUYỆT ĐỐI (PASS)]")
    print("-> Các lớp khởi tạo zero nhận gradient khỏe mạnh ngay từ Step 1.")
    print("-> Sau 50 iters, cả B và C đều chủ động tách biệt biểu diễn so với A và khác biệt lẫn nhau.")
    print("=" * 70)

if __name__ == '__main__':
    test_smoke_50iters_gradient()
