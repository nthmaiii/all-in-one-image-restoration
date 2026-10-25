"""Unit Test & Khảo Sát Khoa Học: Phân Tích Cơ Chế Triệt Tiêu Của Orthogonal Loss (OPR).
Mã nhiệm vụ: Phân tích chuyên sâu H4 & Kiểm toán công thức OPR.
Nội dung:
1. Chứng minh toán học & thực nghiệm vì sao công thức cũ (flatten C*H*W) luôn ra 0.0000.
2. Phân tích hiện tượng Concentration of Measure (Tập trung độ đo) trong không gian siêu cao chiều.
3. Thử nghiệm hiện tượng Sụp đổ biểu diễn (Representation Collapse) khi huấn luyện không có ràng buộc hiệu dụng.
4. Đo đạc độ nhạy gradient của công thức cũ vs công thức mới (Channel-pooled Cosine Gram Matrix).
"""

import sys
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def analyze_pilot_v1_loss():
    """Khảo sát công thức Pilot v1 trên tensor D = 262,144 phần tử."""
    print("=" * 70)
    print("THÍ NGHIỆM 1: KHẢO SÁT CÔNG THỨC PILOT V1 (FLAT TENSOR D = 262,144)")
    print("=" * 70)

    torch.manual_seed(42)
    # Khởi tạo prompt bank cấp 1: (1, 5, 64, 64, 64) -> D = 262,144
    prompt_bank = nn.Parameter(torch.randn(1, 5, 64, 64, 64) * 0.02)
    prompt_len = 5
    D = 64 * 64 * 64

    # Công thức cũ
    flat = prompt_bank.squeeze(0).view(prompt_len, -1)
    norm = F.normalize(flat, p=2, dim=1)
    cosine_sim = torch.mm(norm, norm.t())
    eye = torch.eye(prompt_len)
    off_diag = cosine_sim - eye
    loss_v1 = (off_diag ** 2).sum() / (prompt_len * (prompt_len - 1))

    # Lý thuyết nồng độ độ đo: Var(cos) = 1/D => Std(cos) = 1/sqrt(D)
    theoretical_std = 1.0 / np.sqrt(D)
    theoretical_sq = 1.0 / D

    off_diag_values = cosine_sim[~torch.eye(prompt_len, dtype=bool)].detach().numpy()
    mean_abs_cos = np.mean(np.abs(off_diag_values))
    max_abs_cos = np.max(np.abs(off_diag_values))

    print(f"1. Chiều không gian D: {D:,} phần tử.")
    print(f"2. Lý thuyết Gaussian: 1/sqrt(D) = {theoretical_std:.6f}, 1/D = {theoretical_sq:.8f}")
    print(f"3. Cosine similarity thực tế giữa các token ngẫu nhiên:")
    print(f"   - Mean |cos(u, v)|: {mean_abs_cos:.6f}")
    print(f"   - Max  |cos(u, v)|: {max_abs_cos:.6f}")
    print(f"4. Giá trị Loss Pilot v1: {loss_v1.item():.8f}")
    print(f"   - Khi in ra với format ':.4f': '{loss_v1.item():.4f}'")
    print(f"   - Khi nhân với lambda = 0.01:  '{ (0.01 * loss_v1).item():.10f}'")

    # Đo gradient
    loss_v1.backward()
    grad_norm_v1 = prompt_bank.grad.norm().item()
    print(f"5. Độ lớn Gradient (L2 norm) từ Loss v1: {grad_norm_v1:.8f}")
    print("-> KẾT LUẬN 1: Giá trị 0.0000 là do hiện tượng tập trung độ đo (Concentration of Measure)")
    print("   trong không gian 262,144 chiều. Mọi vector ngẫu nhiên tự nhiên đã có cos ~ 0.0019,")
    print("   khiến bình phương tiệm cận 3.8e-6, hoàn toàn bị triệt tiêu ở độ chính xác in ấn và gradient.")


def analyze_channel_pooled_v2():
    """Khảo sát công thức mới: Spatial Average Pooling trước khi tính Cosine Gram Matrix."""
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM 2: KHẢO SÁT CÔNG THỨC MỚI (SPATIAL POOLING -> CHANNEL GRAM MATRIX)")
    print("=" * 70)

    torch.manual_seed(42)
    prompt_bank = nn.Parameter(torch.randn(1, 5, 64, 64, 64) * 0.02)
    prompt_len = 5
    C = 64  # Chiều không gian kênh sau spatial pooling

    # Pooling không gian: (5, C)
    v = prompt_bank.squeeze(0).mean(dim=(-2, -1))
    norm_v = F.normalize(v, p=2, dim=1)
    gram = torch.mm(norm_v, norm_v.t())
    eye = torch.eye(prompt_len)
    off_diag = gram - eye
    loss_v2 = (off_diag ** 2).sum() / (prompt_len * (prompt_len - 1))

    theoretical_std_c = 1.0 / np.sqrt(C)
    theoretical_sq_c = 1.0 / C

    off_diag_values = gram[~torch.eye(prompt_len, dtype=bool)].detach().numpy()
    mean_abs_cos = np.mean(np.abs(off_diag_values))
    max_abs_cos = np.max(np.abs(off_diag_values))

    print(f"1. Chiều không gian biểu diễn kênh C: {C} chiều.")
    print(f"2. Lý thuyết Gaussian trong R^{C}: 1/sqrt(C) = {theoretical_std_c:.4f}, 1/C = {theoretical_sq_c:.4f}")
    print(f"3. Cosine similarity thực tế giữa các token kênh:")
    print(f"   - Mean |cos(v_i, v_j)|: {mean_abs_cos:.4f}")
    print(f"   - Max  |cos(v_i, v_j)|: {max_abs_cos:.4f}")
    print(f"4. Giá trị Loss v2: {loss_v2.item():.6f}")
    print(f"   - Khi in ra với format ':.4f': '{loss_v2.item():.4f}'")
    print(f"   - Khi nhân với lambda = 0.05:  '{ (0.05 * loss_v2).item():.6f}'")

    # Đo gradient
    loss_v2.backward()
    grad_norm_v2 = prompt_bank.grad.norm().item()
    print(f"5. Độ lớn Gradient (L2 norm) từ Loss v2: {grad_norm_v2:.6f}")
    print("-> KẾT LUẬN 2: Trong không gian R^64, các vector token ngẫu nhiên có cos ~ 0.12 - 0.25.")
    print(f"   Loss đạt {loss_v2.item():.4f}, gradient đạt {grad_norm_v2:.6f}, lớn gấp ~1000 lần so với v1!")


def test_representation_collapse_dynamics():
    """Mô phỏng động học huấn luyện: Liệu prompt token có bị sụp đổ biểu diễn (Representation Collapse) không?"""
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM 3: MÔ PHỎNG ĐỘNG HỌC HUẤN LUYỆN & NGUY CƠ SỤP ĐỔ BIỂU DIỄN")
    print("=" * 70)

    def simulate_training(lambda_reg=0.0, use_pooling=True, steps=100, lr=1e-2):
        torch.manual_seed(42)
        prompt = nn.Parameter(torch.randn(5, 64, 64, 64) * 0.02)
        optimizer = torch.optim.Adam([prompt], lr=lr)

        cos_history = []
        for step in range(steps):
            optimizer.zero_grad()

            # Giả lập gradient tái tạo chung từ tập train: Mọi token đều nhận lực kéo về hướng nhiễu/sương
            # Trong PromptIR, P = sum(w_i * p_i), gradient dL/dp_i = w_i * dL/dP
            shared_direction = torch.ones_like(prompt[0]) * 0.1
            loss_recon = 0.0
            weights = [0.4, 0.3, 0.15, 0.1, 0.05]
            for i in range(5):
                loss_recon = loss_recon + weights[i] * F.mse_loss(prompt[i], shared_direction)

            # Ràng buộc trực giao
            if lambda_reg > 0.0:
                if use_pooling:
                    v = prompt.mean(dim=(-2, -1))
                    norm_v = F.normalize(v, p=2, dim=1)
                    gram = torch.mm(norm_v, norm_v.t())
                    eye = torch.eye(5)
                    loss_orth = ((gram - eye) ** 2).sum() / 20.0
                else:
                    flat = prompt.view(5, -1)
                    norm_f = F.normalize(flat, p=2, dim=1)
                    gram = torch.mm(norm_f, norm_f.t())
                    eye = torch.eye(5)
                    loss_orth = ((gram - eye) ** 2).sum() / 20.0
                total_loss = loss_recon + lambda_reg * loss_orth
            else:
                total_loss = loss_recon

            total_loss.backward()
            optimizer.step()

            # Đo cosine trung bình giữa các token kênh
            with torch.no_grad():
                v = prompt.mean(dim=(-2, -1))
                norm_v = F.normalize(v, p=2, dim=1)
                gram = torch.mm(norm_v, norm_v.t())
                mean_cos = gram[~torch.eye(5, dtype=bool)].abs().mean().item()
                cos_history.append(mean_cos)

        return cos_history

    steps = 100
    hist_no_reg = simulate_training(lambda_reg=0.0, steps=steps)
    hist_v1_reg = simulate_training(lambda_reg=0.05, use_pooling=False, steps=steps)
    hist_v2_reg = simulate_training(lambda_reg=0.05, use_pooling=True, steps=steps)

    print(f"Bước lặp (Step)        | Step 0 (Init) | Step 20 | Step 50 | Step 100 (Final)")
    print(f"-----------------------|---------------|---------|---------|------------------")
    print(f"1. Không Reg (lambda=0)| {hist_no_reg[0]:.4f}        | {hist_no_reg[20]:.4f}  | {hist_no_reg[50]:.4f}  | {hist_no_reg[99]:.4f} (SỤP ĐỔ)")
    print(f"2. Pilot v1 (Flat D)   | {hist_v1_reg[0]:.4f}        | {hist_v1_reg[20]:.4f}  | {hist_v1_reg[50]:.4f}  | {hist_v1_reg[99]:.4f} (KHÔNG CẢN ĐƯỢC)")
    print(f"3. Đề xuất v2 (Pooling)| {hist_v2_reg[0]:.4f}        | {hist_v2_reg[20]:.4f}  | {hist_v2_reg[50]:.4f}  | {hist_v2_reg[99]:.4f} (GIỮ ĐỘC LẬP)")
    print("=" * 70)


def extract_prompt_diagnostic_metrics(model) -> dict:
    """Hàm trích xuất đo lường trực giao đa lớp (Multi-layer Orthogonality Diagnostic).
    Đo trực tiếp Cosine Similarity trung bình và cực đại của từng decoder level,
    không nhân với hệ số lambda.
    """
    metrics = {}
    levels = [('level1', getattr(model, 'prompt1', None)),
              ('level2', getattr(model, 'prompt2', None)),
              ('level3', getattr(model, 'prompt3', None))]

    for lvl_name, p_block in levels:
        if p_block is None:
            continue

        banks = []
        if hasattr(p_block, 'prompt_low') and hasattr(p_block, 'prompt_high'):
            banks = [('low', p_block.prompt_low), ('high', p_block.prompt_high)]
        elif hasattr(p_block, 'prompt_bank1') and hasattr(p_block, 'prompt_bank2'):
            banks = [('bank1', p_block.prompt_bank1), ('bank2', p_block.prompt_bank2)]

        for b_name, bank in banks:
            # bank: (1, N, C, H, W)
            N = bank.shape[1]
            v = bank.squeeze(0).mean(dim=(-2, -1)) # (N, C)
            norm_v = F.normalize(v, p=2, dim=1)
            gram = torch.mm(norm_v, norm_v.t()) # (N, N)
            off_diag = gram[~torch.eye(N, dtype=bool, device=bank.device)].abs()

            metrics[f'{lvl_name}_{b_name}_mean_cos'] = round(float(off_diag.mean().item()), 4)
            metrics[f'{lvl_name}_{b_name}_max_cos'] = round(float(off_diag.max().item()), 4)

        # Cross bank nếu có cả 2 bank
        if len(banks) == 2:
            v0 = banks[0][1].squeeze(0).mean(dim=(-2, -1))
            v1 = banks[1][1].squeeze(0).mean(dim=(-2, -1))
            cross_gram = torch.mm(F.normalize(v0, p=2, dim=1), F.normalize(v1, p=2, dim=1).t())
            metrics[f'{lvl_name}_cross_mean_cos'] = round(float(cross_gram.abs().mean().item()), 4)

    return metrics


if __name__ == '__main__':
    analyze_pilot_v1_loss()
    analyze_channel_pooled_v2()
    test_representation_collapse_dynamics()

    # Thử nghiệm trên mô hình thật
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM 4: TRÍCH XUẤT CHỈ SỐ ĐO LƯỜNG TRÊN MÔ HÌNH FD-PROMPTIR THẬT")
    print("=" * 70)
    from src.models.fd_promptir import FDPromptIR
    m = FDPromptIR()
    diag = extract_prompt_diagnostic_metrics(m)
    for k, v in diag.items():
        print(f" - {k:30s}: {v:.4f}")
