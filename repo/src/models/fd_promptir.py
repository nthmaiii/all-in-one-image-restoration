"""Frequency-Decoupled Prompting for All-in-One Image Restoration (FD-PromptIR).
Mã nhiệm vụ: P4-T2
Đề xuất khoa học:
- Tách biệt phổ tần số cao và thấp để triệt tiêu hiện tượng xung đột biểu diễn (representation collision).
- Tích hợp hàm phạt trực giao (Orthogonal Prompt Regularization - OPR) để tối đa hóa tính độc lập giữa các prompt vector.
"""

import sys
from pathlib import Path
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Nạp các khối cốt lõi từ base PromptIR
try:
    from baselines.promptir.net.model import (
        TransformerBlock, Downsample, Upsample, OverlapPatchEmbed, LayerNorm
    )
except ImportError:
    # Dự phòng nạp tương đối
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from baselines.promptir.net.model import (
        TransformerBlock, Downsample, Upsample, OverlapPatchEmbed, LayerNorm
    )

class FDPromptGenBlock(nn.Module):
    """Khối sinh Prompt phân tách miền tần số (Frequency-Decoupled Prompt Generation Block).
    Tách đặc trưng thành:
    - x_low: Tần số thấp (chứa độ sáng nền, sương mù, độ tương phản toàn cục)
    - x_high: Tần số cao (chứa nhiễu ngẫu nhiên, vệt mưa, chi tiết cạnh biên)
    Tạo hai bộ prompt độc lập và hợp nhất thích ứng qua khối fusion.
    """
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(FDPromptGenBlock, self).__init__()
        self.prompt_dim = prompt_dim
        self.prompt_len = prompt_len
        self.prompt_size = prompt_size

        # 1. Hai kho tham số Prompt độc lập: Thấp tần & Cao tần
        self.prompt_low = nn.Parameter(torch.randn(1, prompt_len, prompt_dim, prompt_size, prompt_size) * 0.02)
        self.prompt_high = nn.Parameter(torch.randn(1, prompt_len, prompt_dim, prompt_size, prompt_size) * 0.02)

        # 2. Bộ lọc phân tách tần số không gian (Spatial Frequency Filter)
        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 3. Bộ dự đoán trọng số cho 2 luồng tần số
        self.fc_low = nn.Linear(lin_dim, prompt_len)
        self.fc_high = nn.Linear(lin_dim, prompt_len)

        # 4. Tầng hợp nhất thích ứng đa tần (Adaptive Dual-Frequency Fusion)
        self.fusion = nn.Sequential(
            nn.Conv2d(prompt_dim * 2, prompt_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # A. Phân tách tần số
        x_low = self.low_pass(x)
        x_high = x - x_low

        # B. Trích xuất vector mô tả tần số
        # - Low-frequency descriptor: Cường độ DC trung bình
        emb_low = x_low.mean(dim=(-2, -1))
        # - High-frequency descriptor: Năng lượng / Độ lệch chuẩn AC
        emb_high = torch.sqrt(torch.var(x_high, dim=(-2, -1), unbiased=False) + 1e-6)

        # C. Trọng số softmax trên 2 kho prompt độc lập
        weights_low = F.softmax(self.fc_low(emb_low), dim=1)
        weights_high = F.softmax(self.fc_high(emb_high), dim=1)

        # D. Sinh prompt đặc thù cho từng dải tần
        p_low = (weights_low.view(B, self.prompt_len, 1, 1, 1) * self.prompt_low).sum(dim=1)
        p_high = (weights_high.view(B, self.prompt_len, 1, 1, 1) * self.prompt_high).sum(dim=1)

        # E. Nội suy không gian về (H, W)
        p_low = F.interpolate(p_low, (H, W), mode="bilinear", align_corners=False)
        p_high = F.interpolate(p_high, (H, W), mode="bilinear", align_corners=False)

        # F. Hợp nhất hai luồng prompt
        prompt_fused = self.fusion(torch.cat([p_low, p_high], dim=1))
        return prompt_fused

    def get_orthogonal_loss(self):
        """Tính hàm mất mát trực giao hóa (OPR) trên cả 2 kho prompt.
        Ép cosine similarity giữa các vector prompt khác nhau tiến về 0.
        """
        loss = 0.0
        for bank in [self.prompt_low, self.prompt_high]:
            flat = bank.squeeze(0).view(self.prompt_len, -1)
            norm = F.normalize(flat, p=2, dim=1)
            cosine_sim = torch.mm(norm, norm.t())
            eye = torch.eye(self.prompt_len, device=bank.device)
            off_diag = cosine_sim - eye
            loss += (off_diag ** 2).sum() / (self.prompt_len * (self.prompt_len - 1))
        return loss / 2.0


class FDPromptIR(nn.Module):
    """Mô hình đề xuất FD-PromptIR (Frequency-Decoupled Prompting Image Restoration).
    Kế thừa kiến trúc Restormer 4 cấp của PromptIR, nhưng nâng cấp toàn bộ hệ thống
    PromptGenBlock thành FDPromptGenBlock.
    """
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=True,
    ):
        super(FDPromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        # Tích hợp khối FD-PromptGenBlock tại 3 cấp decoder
        if self.decoder:
            self.prompt1 = FDPromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = FDPromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = FDPromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)

        # Các tầng giảm kênh tương thích
        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        # Encoder Levels
        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim * 2**1) + 128, int(dim * 2**1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2**1))

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim * 2**2) + 256, int(dim * 2**2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        # Decoder Level 3
        self.up4_3 = Upsample(int(dim * 2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**1) + 192, int(dim * 2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim * 2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2**2) + 512, int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        # Decoder Level 2
        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim * 2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2**1) + 224, int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        # Decoder Level 1
        self.up2_1 = Upsample(int(dim * 2**1))
        self.noise_level1 = TransformerBlock(dim=int(dim * 2**1) + 64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2**1) + 64, int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        # Refinement & Output
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1

    def get_orthogonal_loss(self):
        """Tổng hàm mất mát trực giao OPR trên cả 3 cấp decoder."""
        if not self.decoder:
            return torch.tensor(0.0)
        return (
            self.prompt1.get_orthogonal_loss()
            + self.prompt2.get_orthogonal_loss()
            + self.prompt3.get_orthogonal_loss()
        ) / 3.0

    def load_pretrained_promptir(self, ckpt_path):
        """Tải trọng số backbone từ pretrained PromptIR để warm-start."""
        print(f"Khởi tạo trọng số từ pretrained PromptIR: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('params', checkpoint))
        clean_state = {}
        for k, v in state_dict.items():
            clean_k = k.replace('net.', '').replace('module.', '')
            # Bỏ qua các trọng số của PromptGenBlock cũ để FDPromptGenBlock tự khởi tạo mới
            if 'prompt1.' in clean_k or 'prompt2.' in clean_k or 'prompt3.' in clean_k:
                continue
            clean_state[clean_k] = v

        missing, unexpected = self.load_state_dict(clean_state, strict=False)
        print(f">> Backbone PromptIR đã nạp thành công! (Số layer nạp: {len(clean_state)}, lớp mới tự khởi tạo: {len(missing)})")
