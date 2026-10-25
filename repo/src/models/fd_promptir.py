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
        TransformerBlock, Downsample, Upsample, OverlapPatchEmbed, LayerNorm, PromptGenBlock
    )
except ImportError:
    # Dự phòng nạp tương đối
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from baselines.promptir.net.model import (
        TransformerBlock, Downsample, Upsample, OverlapPatchEmbed, LayerNorm, PromptGenBlock
    )

class FDPromptGenBlock(nn.Module):
    """Khối sinh Prompt phân tách miền tần số (Frequency-Decoupled Prompt Generation Block).
    Tách đặc trưng thành:
    - x_low: Tần số thấp (chứa độ sáng nền, sương mù, độ tương phản toàn cục)
    - x_high: Tần số cao (chứa nhiễu ngẫu nhiên, vệt mưa, chi tiết cạnh biên)
    Tạo hai bộ prompt độc lập và hợp nhất thích ứng qua khối fusion.
    Bảo toàn đồng nhất (Identity-Preserving): đầu ra = prompt cũ + prompt mới.
    Lớp chiếu cuối cùng được zero-init để tại iter 0 đầu ra trùng khớp 100% với PromptIR pretrained.
    """
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(FDPromptGenBlock, self).__init__()
        self.prompt_dim = prompt_dim
        self.prompt_len = prompt_len
        self.prompt_size = prompt_size

        # 0. Khối Prompt pretrained nguyên bản của PromptIR (bảo toàn 100% tri thức nền)
        self.base_prompt = PromptGenBlock(prompt_dim=prompt_dim, prompt_len=prompt_len, prompt_size=prompt_size, lin_dim=lin_dim)

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
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # KHỞI TẠO BẢO TOÀN ĐỒNG NHẤT:
        # Zero-init ma trận chiếu CUỐI (weights + bias) của khối mới, KHÔNG dùng scalar alpha
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, x):
        # A. Prompt pretrained nguyên bản
        base_p = self.base_prompt(x)

        # B. Phân tách tần số
        B, C, H, W = x.shape
        x_low = self.low_pass(x)
        x_high = x - x_low

        emb_low = x_low.mean(dim=(-2, -1))
        emb_high = torch.sqrt(torch.var(x_high, dim=(-2, -1), unbiased=False) + 1e-6)

        weights_low = F.softmax(self.fc_low(emb_low), dim=1)
        weights_high = F.softmax(self.fc_high(emb_high), dim=1)

        p_low = (weights_low.view(B, self.prompt_len, 1, 1, 1) * self.prompt_low).sum(dim=1)
        p_high = (weights_high.view(B, self.prompt_len, 1, 1, 1) * self.prompt_high).sum(dim=1)

        p_low = F.interpolate(p_low, (H, W), mode="bilinear", align_corners=False)
        p_high = F.interpolate(p_high, (H, W), mode="bilinear", align_corners=False)

        prompt_fused = self.fusion(torch.cat([p_low, p_high], dim=1))

        # Bảo toàn đồng nhất: đầu ra = prompt cũ + prompt mới
        return base_p + prompt_fused

    def get_orthogonal_loss(self):
        """Tính hàm mất mát trực giao hóa (OPR) trên cả 2 kho prompt.
        Thay vì dẹp toàn bộ tensor (D > 260,000 khiến tích vô hướng ngẫu nhiên triệt tiêu về 0),
        chúng ta tính trung bình không gian (spatial pooling) của từng token để thu được
        vector đặc trưng kênh (C-dim), sau đó chuẩn hóa và tính ma trận Gram cosine:
        Gram = (V / ||V||) (V / ||V||)^T.
        Hàm mất mát phạt cả trực giao nội bộ (intra-bank) và trực giao chéo (inter-bank).
        """
        # bank shape: (1, prompt_len, prompt_dim, H_p, W_p)
        v_low = self.prompt_low.squeeze(0).mean(dim=(-2, -1))   # (prompt_len, prompt_dim)
        v_high = self.prompt_high.squeeze(0).mean(dim=(-2, -1)) # (prompt_len, prompt_dim)

        norm_low = F.normalize(v_low, p=2, dim=1)
        norm_high = F.normalize(v_high, p=2, dim=1)

        # 1. Intra-bank orthogonality (nội bộ bank thấp tần và cao tần)
        gram_low = torch.mm(norm_low, norm_low.t())
        gram_high = torch.mm(norm_high, norm_high.t())
        eye = torch.eye(self.prompt_len, device=self.prompt_low.device)

        loss_intra_low = ((gram_low - eye) ** 2).sum() / (self.prompt_len * (self.prompt_len - 1))
        loss_intra_high = ((gram_high - eye) ** 2).sum() / (self.prompt_len * (self.prompt_len - 1))

        # 2. Inter-bank cross-orthogonality (trực giao giữa thấp tần và cao tần)
        gram_cross = torch.mm(norm_low, norm_high.t())
        loss_cross = (gram_cross ** 2).mean()

        return 0.5 * (loss_intra_low + loss_intra_high) + loss_cross


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
        """Tải trọng số backbone và prompt pretrained từ PromptIR để bảo toàn đồng nhất."""
        print(f"Khởi tạo trọng số từ pretrained PromptIR: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('params', checkpoint))
        clean_state = {}
        for k, v in state_dict.items():
            clean_k = k.replace('net.', '').replace('module.', '')
            # Chuyển hướng các trọng số prompt cũ vào base_prompt
            if 'prompt1.' in clean_k:
                clean_k = clean_k.replace('prompt1.', 'prompt1.base_prompt.')
            elif 'prompt2.' in clean_k:
                clean_k = clean_k.replace('prompt2.', 'prompt2.base_prompt.')
            elif 'prompt3.' in clean_k:
                clean_k = clean_k.replace('prompt3.', 'prompt3.base_prompt.')
            clean_state[clean_k] = v

        missing, unexpected = self.load_state_dict(clean_state, strict=False)
        print(f">> Backbone & Pretrained Prompt đã nạp thành công! (Số layer nạp: {len(clean_state)}, lớp mới zero-init: {len(missing)})")
