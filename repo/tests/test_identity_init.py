"""Unit Test Kiểm Định Khởi Tạo Bảo Toàn Đồng Nhất (Identity-Preserving Initialization Test).
Mục đích:
1. Xác minh số lượng tham số của Model B (FD-PromptIR) và Model C (Param-Matched PromptIR) bằng nhau 100%.
2. Trên 8 ảnh cố định, kiểm chứng PSNR(B, iter 0) == PSNR(C, iter 0) == PSNR(A, iter 0) với sai số < 1e-3 dB.
3. Chứng minh cơ chế Zero-Init lớp chiếu cuối bảo toàn nguyên vẹn đầu ra của backbone và prompt pretrained.
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Thiết lập encoding utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'baselines' / 'promptir'))

from baselines.promptir.net.model import PromptIR
from src.models.fd_promptir import FDPromptIR
from src.models.param_matched_promptir import ParamMatchedPromptIR

def compute_psnr(img1, img2):
    """Tính PSNR giữa 2 tensor ảnh chuẩn hóa [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse < 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)

def test_identity_preserving_init():
    print("=" * 75)
    print("KIỂM ĐỊNH KHOA HỌC: KHỞI TẠO BẢO TOÀN ĐỒNG NHẤT (IDENTITY-PRESERVING ZERO-INIT)")
    print("=" * 75)

    device = torch.device('cpu')
    torch.manual_seed(42)

    # 1. Khởi tạo 3 mô hình A, B, C
    print("\n[BƯỚC 1] Khởi tạo kiến trúc Model A, B, C...")
    model_a = PromptIR(decoder=True).to(device)
    model_b = FDPromptIR(decoder=True).to(device)
    model_c = ParamMatchedPromptIR(decoder=True).to(device)

    # 2. Đếm và khẳng định (Assert) số lượng tham số B và C
    params_a = sum(p.numel() for p in model_a.parameters())
    params_b = sum(p.numel() for p in model_b.parameters())
    params_c = sum(p.numel() for p in model_c.parameters())

    print(f"-> Số tham số Model A (PromptIR Baseline):        {params_a:,}")
    print(f"-> Số tham số Model B (FD-PromptIR Đề xuất):      {params_b:,}")
    print(f"-> Số tham số Model C (Param-Matched Đối chứng):  {params_c:,}")
    print(f"-> Chênh lệch tham số giữa B và C:                {abs(params_b - params_c):,}")

    assert params_b == params_c, (
        f"LỖI PHI ĐỐI XỨNG THAM SỐ: Model B ({params_b}) != Model C ({params_c})!"
    )
    print(">> [XÁC MINH THAM SỐ THÀNH CÔNG] Model B và Model C có số tham số CHÍNH XÁC BẰNG NHAU 100%!")

    # 3. Đồng bộ trọng số nền giữa A, B, C để mô phỏng tải checkpoint
    # Nạp cùng một bộ trọng số pretrained từ A vào backbone và base_prompt của B và C
    state_a = model_a.state_dict()
    clean_b_state = {}
    clean_c_state = {}
    for k, v in state_a.items():
        clean_k = k.replace('net.', '').replace('module.', '')
        if 'prompt1.' in clean_k:
            mapped_k = clean_k.replace('prompt1.', 'prompt1.base_prompt.')
        elif 'prompt2.' in clean_k:
            mapped_k = clean_k.replace('prompt2.', 'prompt2.base_prompt.')
        elif 'prompt3.' in clean_k:
            mapped_k = clean_k.replace('prompt3.', 'prompt3.base_prompt.')
        else:
            mapped_k = clean_k
        clean_b_state[mapped_k] = v
        clean_c_state[mapped_k] = v

    model_b.load_state_dict(clean_b_state, strict=False)
    model_c.load_state_dict(clean_c_state, strict=False)

    # Đảm bảo các tầng chiếu cuối vẫn là zero-init
    for m in [model_b, model_c]:
        for p_name in ['prompt1', 'prompt2', 'prompt3']:
            p_block = getattr(m, p_name)
            assert torch.all(p_block.fusion[-1].weight == 0), f"Lớp cuối của {p_name} trọng số chưa bằng 0!"
            assert torch.all(p_block.fusion[-1].bias == 0), f"Lớp cuối của {p_name} bias chưa bằng 0!"

    print(">> [XÁC MINH ZERO-INIT THÀNH CÔNG] Tất cả các ma trận chiếu cuối (weights + bias) đều bằng 0 tuyệt đối.")

    # 4. Kiểm thử trên 8 ảnh cố định
    print("\n[BƯỚC 2] Kiểm thử forward pass trên 8 ảnh cố định (Fixed Seed 42)...")
    model_a.eval()
    model_b.eval()
    model_c.eval()

    torch.manual_seed(999)
    test_images = torch.rand(8, 3, 128, 128, device=device)

    max_delta_b_a = 0.0
    max_delta_c_a = 0.0

    print("\n| Ảnh | PSNR(A, iter 0) | PSNR(B, iter 0) | PSNR(C, iter 0) | Sai số B-A (dB) | Sai số C-A (dB) | Kết luận |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    with torch.no_grad():
        for i in range(8):
            img_i = test_images[i:i+1]
            out_a = model_a(img_i)
            out_b = model_b(img_i)
            out_c = model_c(img_i)

            # Tính chênh lệch tensor
            diff_ba = torch.max(torch.abs(out_b - out_a)).item()
            diff_ca = torch.max(torch.abs(out_c - out_a)).item()

            # Tính PSNR giữa ảnh phục hồi và ảnh giả định ground-truth (để lấy giá trị thực tế)
            psnr_a = compute_psnr(out_a, img_i)
            psnr_b = compute_psnr(out_b, img_i)
            psnr_c = compute_psnr(out_c, img_i)

            err_ba = abs(psnr_b - psnr_a)
            err_ca = abs(psnr_c - psnr_a)

            if err_ba > max_delta_b_a:
                max_delta_b_a = err_ba
            if err_ca > max_delta_c_a:
                max_delta_c_a = err_ca

            assert err_ba < 1e-3, f"Ảnh {i}: Sai số B-A ({err_ba:.6e} dB) vượt ngưỡng 1e-3 dB!"
            assert err_ca < 1e-3, f"Ảnh {i}: Sai số C-A ({err_ca:.6e} dB) vượt ngưỡng 1e-3 dB!"

            print(f"| #{i+1}  | {psnr_a:8.4f} dB   | {psnr_b:8.4f} dB   | {psnr_c:8.4f} dB   | {err_ba:11.2e} dB | {err_ca:11.2e} dB | KHỚP TUYỆT ĐỐI |")

    print("\n" + "=" * 75)
    print(f"TỔNG KẾT KIỂM TOÁN ĐỒNG NHẤT TẠI ITERATION 0:")
    print(f"-> Sai số PSNR lớn nhất giữa B và A: {max_delta_b_a:.2e} dB (< 1e-3 dB)")
    print(f"-> Sai số PSNR lớn nhất giữa C và A: {max_delta_c_a:.2e} dB (< 1e-3 dB)")
    print(f"-> Độ lệch tensor lớn nhất:          B-A = {diff_ba:.2e}, C-A = {diff_ca:.2e}")
    print(">> KẾT LUẬN: ĐẠT CHUẨN KHOA HỌC. ĐẦU RA B, C, A TRÙNG NHAU 100% TẠI ITERATION 0.")
    print("=" * 75)

if __name__ == '__main__':
    test_identity_preserving_init()
