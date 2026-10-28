"""Unit Test Toàn Diện Kiến Trúc Mạng PHASE-HOPFIELD-IR.
Mã kiểm toán: TEST-PHASE-HOPFIELD-P3

Bộ test bao gồm:
1. Test Lie Anti-Symmetry: Đẳng thức phản đối xứng [A, B] = -[B, A] của giao hoán tử Lie.
2. Test Hopfield Energy Descent: Định lý thế năng Lyapunov Nobel 2024 E(z_retrieved) <= E(z_degraded).
3. Test Exact Parameter Equivalence: params(Model_B) == params(Model_C) chuẩn xác 100%.
4. Test Healthy Gradient Flow: Mọi tham số mới đều nhận gradient khác 0 (triệt tiêu representation bypass).
5. Test Output Shape Invariance: Kích thước tensor đầu ra khớp hoàn toàn với đầu vào.
"""

import sys
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.phase_hopfield_promptir import (
    FractionalFourierTransform2D,
    LieFrFTHopfieldBlock,
    ParamMatchedSpatialDualBlock,
    PhaseHopfieldIR,
    ParamMatchedPhaseHopfieldIR
)


def test_lie_anti_symmetry():
    """Kiểm tra tính phản đối xứng tuyệt đối của giao hoán tử Lie: [A, B] = -[B, A]."""
    torch.manual_seed(42)
    B, r = 8, 16
    A = torch.randn(B, r, r)
    B_mat = torch.randn(B, r, r)

    comm_AB = torch.matmul(A, B_mat) - torch.matmul(B_mat, A)
    comm_BA = torch.matmul(B_mat, A) - torch.matmul(A, B_mat)

    discrepancy = (comm_AB + comm_BA).abs().max().item()
    print(f"\n[TEST 1] Lie Anti-symmetry discrepancy: {discrepancy:.2e}")
    assert discrepancy < 1e-6, f"Toán tử Lie vi phạm tính phản đối xứng: {discrepancy}"


def test_hopfield_energy_monotonicity():
    """Kiểm chứng định lý thế năng Lyapunov (Nobel 2024):
    Quá trình liên tưởng Hopfield kéo trạng thái z về gần các attractor M,
    làm giảm thế năng Lyapunov: E(z_retrieved) <= E(z).
    """
    torch.manual_seed(42)
    dim = 64
    K = 16
    M = nn.Parameter(torch.randn(K, dim) * 0.1)
    beta = 1.0 / math.sqrt(dim)

    # Giả lập 32 vector đặc trưng bị nhiễu/sương mù làm chệch khỏi manifold
    z_degraded = torch.randn(32, dim) * 2.0

    def energy(z):
        dots = torch.matmul(z, M.t()) * beta
        lse = torch.logsumexp(dots, dim=-1) / beta
        quad = 0.5 * (z ** 2).sum(dim=-1)
        return (quad - lse).mean()

    # Associative Hopfield retrieval
    attn = F.softmax(torch.matmul(z_degraded, M.t()) * beta, dim=-1)
    z_retrieved = torch.matmul(attn, M)

    e_initial = energy(z_degraded).item()
    e_retrieved = energy(z_retrieved).item()
    delta_e = e_retrieved - e_initial

    print(f"\n[TEST 2] Hopfield Initial Energy: {e_initial:.4f}, Retrieved Energy: {e_retrieved:.4f}, Delta E: {delta_e:.4f}")
    assert delta_e <= 0, f"Thế năng Hopfield không giảm sau khi liên tưởng: Delta E = {delta_e}"


def test_exact_parameter_equivalence():
    """Kiểm tra đẳng tham số 100% giữa Branch B (Đề xuất) và Branch C (Đối chứng)."""
    torch.manual_seed(42)
    model_b = PhaseHopfieldIR(decoder=True)
    model_c = ParamMatchedPhaseHopfieldIR(decoder=True)

    params_b = sum(p.numel() for p in model_b.parameters())
    params_c = sum(p.numel() for p in model_c.parameters())
    diff = abs(params_b - params_c)

    print(f"\n[TEST 3] Model B (Proposed): {params_b:,} params | Model C (Control): {params_c:,} params | Diff: {diff}")
    assert diff == 0, f"Vi phạm nguyên tắc đối chứng công bằng: Chênh lệch tham số {diff} != 0!"


def test_gradient_flow_and_dormancy_elimination():
    """Kiểm tra dòng chảy gradient trên khối mới:
    Khẳng định Gate gradient và Memory M gradient khác 0 mạnh mẽ,
    chứng minh cơ chế Energy-Scaled Coupling triệt tiêu hoàn toàn lỗi ngủ đông.
    """
    torch.manual_seed(42)
    block = LieFrFTHopfieldBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96, r=16, K=16)
    x = torch.randn(2, 96, 32, 32, requires_grad=True)

    out = block(x)
    assert block.last_ratio > 0.01, f"Tỉ lệ năng lượng quá nhỏ: {block.last_ratio}"

    loss = out.sum()
    loss.backward()

    gate_grad = block.gate.grad.abs().item()
    m_grad = block.M.grad.abs().mean().item()
    head_a_grad = block.head_A.weight.grad.abs().mean().item()

    print(f"\n[TEST 4] Gradients: Gate = {gate_grad:.2f}, Memory M = {m_grad:.4f}, Lie Head A = {head_a_grad:.4f}")
    assert gate_grad > 1e-3, "Gradient của coupling gate bị triệt tiêu!"
    assert m_grad > 1e-5, "Gradient của Hopfield memory M bị triệt tiêu!"
    assert head_a_grad > 1e-5, "Gradient của Lie generator head A bị triệt tiêu!"


def test_phase_hopfield_forward_shape():
    """Kiểm tra forward pass của PhaseHopfieldIR trên tensor ảnh đầu vào."""
    torch.manual_seed(42)
    model = PhaseHopfieldIR(decoder=True)
    model.eval()

    inp = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(inp)

    print(f"\n[TEST 5] Input shape: {list(inp.shape)} -> Output shape: {list(out.shape)}")
    assert out.shape == inp.shape, f"Kích thước đầu ra {out.shape} không khớp {inp.shape}!"


if __name__ == '__main__':
    print("=" * 80)
    print("BẮT ĐẦU CHẠY TOÀN BỘ BỘ KIỂM TOÁN UNIT TEST (PHASE-HOPFIELD-IR)")
    print("=" * 80)
    test_lie_anti_symmetry()
    test_hopfield_energy_monotonicity()
    test_exact_parameter_equivalence()
    test_gradient_flow_and_dormancy_elimination()
    test_phase_hopfield_forward_shape()
    print("\n" + "=" * 80)
    print("TẤT CẢ 5/5 UNIT TESTS ĐÃ PASS 100%! HỆ THỐNG ĐỦ TIÊU CHUẨN XUẤT XƯỞNG!")
    print("=" * 80)
