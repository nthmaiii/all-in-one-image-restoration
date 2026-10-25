"""Mô hình đối chứng tham số (Parameter-Matched PromptIR - Branch C).
Mã nhiệm vụ: P4-V2 (Branch C Control)
Mục đích khoa học:
- Giữ chính xác 100% cùng kiến trúc backbone, cùng số lượng tham số, cùng cấu trúc 2 kho prompt,
  cùng 2 tầng Linear, cùng khối Fusion như FD-PromptIR (Branch B).
- KHÔNG sử dụng bất kỳ bộ lọc miền tần số nào (không Low-pass AvgPool, không High-pass Subtraction).
- Cả 2 luồng prompt đều trích xuất đặc trưng thuần túy trong miền không gian (Spatial Domain).
- Giúp cô lập biến số: Chứng minh hiệu quả (nếu có) của FD-PromptIR là nhờ cơ chế tách miền tần số
  chứ không phải do việc "tăng thêm ~1.2M tham số / thêm tầng ngẫu nhiên".
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.models.fd_promptir import FDPromptIR
    from baselines.promptir.net.model import PromptGenBlock
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.models.fd_promptir import FDPromptIR
    from baselines.promptir.net.model import PromptGenBlock

class SpatialDualPromptGenBlock(nn.Module):
    """Khối sinh Prompt kép trong miền không gian (Parameter-Matched Spatial Dual Prompt Block).
    Cấu trúc tham số giống hệt FDPromptGenBlock nhưng hoạt động thuần túy trong miền không gian:
    - prompt_bank1 & prompt_bank2: Cùng kích thước với prompt_low & prompt_high.
    - fc_1 & fc_2: Cùng kích thước với fc_low & fc_high.
    - fusion: Cùng kích thước với fusion block của FDPromptGenBlock.
    - Không lọc tần số: Trích xuất đặc trưng không gian trực tiếp (Mean & Std).
    - Bảo toàn đồng nhất (Identity-Preserving): đầu ra = prompt cũ + prompt mới.
    - Zero-init lớp chiếu cuối để tại iter 0 đầu ra trùng khớp 100% với PromptIR pretrained.
    """
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(SpatialDualPromptGenBlock, self).__init__()
        self.prompt_dim = prompt_dim
        self.prompt_len = prompt_len
        self.prompt_size = prompt_size

        # 0. Khối Prompt pretrained nguyên bản của PromptIR (bảo toàn 100% tri thức nền)
        self.base_prompt = PromptGenBlock(prompt_dim=prompt_dim, prompt_len=prompt_len, prompt_size=prompt_size, lin_dim=lin_dim)

        # 1. Hai kho prompt tham số hóa
        self.prompt_bank1 = nn.Parameter(torch.randn(1, prompt_len, prompt_dim, prompt_size, prompt_size) * 0.02)
        self.prompt_bank2 = nn.Parameter(torch.randn(1, prompt_len, prompt_dim, prompt_size, prompt_size) * 0.02)

        # 2. Hai tầng Linear dự đoán trọng số
        self.fc_1 = nn.Linear(lin_dim, prompt_len)
        self.fc_2 = nn.Linear(lin_dim, prompt_len)

        # 3. Tầng hợp nhất thích ứng
        self.fusion = nn.Sequential(
            nn.Conv2d(prompt_dim * 2, prompt_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # KHỞI TẠO BẢO TOÀN ĐỒNG NHẤT:
        # Zero-init ma trận chiếu CUỐI (weights + bias) của khối mới, KHÔNG dùng scalar alpha
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, x):
        # A. Prompt pretrained nguyên bản
        base_p = self.base_prompt(x)

        B, C, H, W = x.shape

        # Đặc trưng miền không gian thuần túy (Global Spatial Descriptors)
        # KHÔNG tách tần số, KHÔNG lọc low-pass, KHÔNG trừ high-pass
        emb_1 = x.mean(dim=(-2, -1))
        emb_2 = torch.sqrt(torch.var(x, dim=(-2, -1), unbiased=False) + 1e-6)

        # Trọng số softmax trên 2 kho prompt không gian
        weights_1 = F.softmax(self.fc_1(emb_1), dim=1)
        weights_2 = F.softmax(self.fc_2(emb_2), dim=1)

        # Sinh 2 luồng prompt không gian
        p1 = (weights_1.view(B, self.prompt_len, 1, 1, 1) * self.prompt_bank1).sum(dim=1)
        p2 = (weights_2.view(B, self.prompt_len, 1, 1, 1) * self.prompt_bank2).sum(dim=1)

        # Nội suy không gian về kích thước đặc trưng
        p1 = F.interpolate(p1, (H, W), mode="bilinear", align_corners=False)
        p2 = F.interpolate(p2, (H, W), mode="bilinear", align_corners=False)

        # Hợp nhất 2 luồng prompt
        prompt_fused = self.fusion(torch.cat([p1, p2], dim=1))

        # Bảo toàn đồng nhất
        return base_p + prompt_fused

    def get_orthogonal_loss(self):
        """Tính hàm mất mát OPR tương đương để bảo đảm công bằng tối đa với Branch B."""
        v1 = self.prompt_bank1.squeeze(0).mean(dim=(-2, -1))
        v2 = self.prompt_bank2.squeeze(0).mean(dim=(-2, -1))

        norm1 = F.normalize(v1, p=2, dim=1)
        norm2 = F.normalize(v2, p=2, dim=1)

        gram1 = torch.mm(norm1, norm1.t())
        gram2 = torch.mm(norm2, norm2.t())
        eye = torch.eye(self.prompt_len, device=self.prompt_bank1.device)

        loss_intra_1 = ((gram1 - eye) ** 2).sum() / (self.prompt_len * (self.prompt_len - 1))
        loss_intra_2 = ((gram2 - eye) ** 2).sum() / (self.prompt_len * (self.prompt_len - 1))

        gram_cross = torch.mm(norm1, norm2.t())
        loss_cross = (gram_cross ** 2).mean()

        return 0.5 * (loss_intra_1 + loss_intra_2) + loss_cross


class ParamMatchedPromptIR(FDPromptIR):
    """Mô hình đối chứng tham số cho PromptIR (Branch C).
    Kế thừa toàn bộ kiến trúc từ FDPromptIR, chỉ thay thế FDPromptGenBlock
    bằng SpatialDualPromptGenBlock. Đảm bảo số lượng tham số và cấu trúc backbone
    khớp chính xác 100% với Branch B.
    """
    def __init__(self, *args, **kwargs):
        super(ParamMatchedPromptIR, self).__init__(*args, **kwargs)
        if self.decoder:
            self.prompt1 = SpatialDualPromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = SpatialDualPromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = SpatialDualPromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)
