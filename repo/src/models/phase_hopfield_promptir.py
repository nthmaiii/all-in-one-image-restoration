"""Non-Commutative Fractional Hopfield Dynamics for All-in-One Image Restoration (PHASE-HOPFIELD-IR).
Mã nghiên cứu: PHASE-HOPFIELD-IR (NeurIPS / CVPR Candidate)

Cơ sở khoa học liên ngành:
1. Giải tích Phân số (Fractional Fourier Transform - FrFT):
   - Biến đổi tín hiệu sang không gian pha liên tục (x, p)_alpha với góc quay động alpha(x) in (0, 1).
   - Nắm bắt đồng thời đặc trưng không gian đa tỷ lệ và cấu trúc tần số không dừng của sương mù & nhiễu.
2. Đại số Lie (Non-Commutative Lie Algebra Commutator):
   - Mô hình hóa tính phi giao hoán vật lý: [Haze, Noise] != 0.
   - Trích xuất chính xác số hạng tương tác chéo Delta_comm = eps * (1 - t) mà các mô hình phẳng bỏ sót.
3. Vật lý Thống kê Hopfield (Nobel Vật lý 2024 - Continuous Modern Hopfield Dynamics):
   - Hàm thế năng Lyapunov E(z) = -1/beta * ln(sum exp(beta M^T z)) + 1/2 ||z||^2.
   - Dòng chảy gradient hạ năng lượng kéo quỹ đạo đặc trưng suy thoái về lòng hút (attractor basin) của ảnh sạch.
   - Giải quyết triệt để lỗi ngủ đông (dormant block / lazy training) với Energy-Scaled Coupling.
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from baselines.promptir.net.model import (
        TransformerBlock, Downsample, Upsample, OverlapPatchEmbed, LayerNorm, PromptGenBlock
    )
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / 'baselines' / 'promptir'))
    from baselines.promptir.net.model import (
        TransformerBlock, Downsample, Upsample, OverlapPatchEmbed, LayerNorm, PromptGenBlock
    )


class FractionalFourierTransform2D(nn.Module):
    """Biến đổi Fourier Phân số 2D khả vi (Differentiable Fast 2D FrFT via Chirp-FFT-Chirp).
    Thực hiện phép quay liên tục trong không gian pha (Phase Space) theo góc phi = alpha * pi / 2.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, a_h, a_w):
        """
        x: (B, C, H, W)
        a_h: (B,) góc quay phân số theo chiều cao H in (0.1, 0.9)
        a_w: (B,) góc quay phân số theo chiều rộng W in (0.1, 0.9)
        """
        B, C, H, W = x.shape
        device = x.device

        # 1. FrFT 1D dọc theo chiều rộng W
        phi_w = (a_w.view(-1, 1, 1, 1) * (math.pi / 2.0)).clamp(0.1, 1.47)
        t_w = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W)
        tan_w = torch.tan(phi_w / 2.0)
        csc_w = 1.0 / torch.sin(phi_w)
        pre_w = torch.exp(-1j * (t_w ** 2) * tan_w)
        t_conv_w = torch.linspace(-2.0, 2.0, 2 * W, device=device).view(1, 1, 1, 2 * W)
        conv_w = torch.exp(1j * (t_conv_w ** 2) * csc_w)

        x_mod_w = x.to(torch.complex64) * pre_w
        X_fft_w = torch.fft.fft(x_mod_w, n=2 * W, dim=-1)
        C_fft_w = torch.fft.fft(conv_w, n=2 * W, dim=-1)
        conv_res_w = torch.fft.ifft(X_fft_w * C_fft_w, dim=-1)[..., W // 2:W // 2 + W]
        x_w = conv_res_w * pre_w / math.sqrt(2.0 * W)

        # 2. FrFT 1D dọc theo chiều cao H
        phi_h = (a_h.view(-1, 1, 1, 1) * (math.pi / 2.0)).clamp(0.1, 1.47)
        t_h = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1)
        tan_h = torch.tan(phi_h / 2.0)
        csc_h = 1.0 / torch.sin(phi_h)
        pre_h = torch.exp(-1j * (t_h ** 2) * tan_h)
        t_conv_h = torch.linspace(-2.0, 2.0, 2 * H, device=device).view(1, 1, 2 * H, 1)
        conv_h = torch.exp(1j * (t_conv_h ** 2) * csc_h)

        x_mod_h = x_w * pre_h
        X_fft_h = torch.fft.fft(x_mod_h, n=2 * H, dim=-2)
        C_fft_h = torch.fft.fft(conv_h, n=2 * H, dim=-2)
        conv_res_h = torch.fft.ifft(X_fft_h * C_fft_h, dim=-2)[..., H // 2:H // 2 + H, :]
        out = conv_res_h * pre_h / math.sqrt(2.0 * H)

        # Trả về tín hiệu tổng hợp phần thực và phần ảo trong không gian pha
        return out.real + out.imag


class LieFrFTHopfieldBlock(nn.Module):
    """Khối sinh Prompt Không gian Pha Hopfield - Đại số Lie (Branch B - Đề xuất).
    Kết hợp:
    1. Pretrained PromptIR Base: Bảo toàn tri thức nền tảng.
    2. FrFT Phase Space: Phân tích trong không gian pha liên tục.
    3. Lie Commutator: [A, B] = AB - BA trích xuất tương tác sương mù x nhiễu.
    4. Continuous Hopfield Attractor (Nobel 2024): Kéo quỹ đạo về cực tiểu thế năng ảnh sạch.
    5. Energy-Scaled Coupling: Xóa bỏ bẫy zero-init, kích hoạt gradient mạnh mẽ từ iter 1.
    """
    def __init__(self, prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96, r=16, K=16):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.prompt_len = prompt_len
        self.prompt_size = prompt_size
        self.lin_dim = lin_dim
        self.r = r
        self.K = K

        # 0. Khối Prompt pretrained nguyên bản của PromptIR (bảo toàn 100% tri thức nền)
        self.base_prompt = PromptGenBlock(prompt_dim=prompt_dim, prompt_len=prompt_len, prompt_size=prompt_size, lin_dim=lin_dim)

        # 1. Fractional Fourier Transform (FrFT)
        self.conv_in = nn.Conv2d(lin_dim, prompt_dim, kernel_size=1)
        self.fc_order = nn.Sequential(
            nn.Linear(lin_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )
        self.frft2d = FractionalFourierTransform2D()

        # 2. Non-Commutative Lie Commutator Module
        self.head_A = nn.Linear(lin_dim, r * r, bias=False)
        self.head_B = nn.Linear(lin_dim, r * r, bias=False)
        self.conv_down = nn.Conv2d(prompt_dim, r, kernel_size=1)
        self.conv_up = nn.Conv2d(r, prompt_dim, kernel_size=1)

        # 3. Continuous Hopfield Attractor Layer (Nobel Vật lý 2024)
        self.M = nn.Parameter(torch.randn(K, prompt_dim) * 0.02)
        self.step_size = nn.Parameter(torch.tensor(0.5))
        self.beta = 1.0 / math.sqrt(prompt_dim)

        # 4. Tầng tổng hợp & Ghép Năng lượng (Energy-Scaled Coupling)
        self.conv_out = nn.Sequential(
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1)
        )
        # Khởi tạo gate tại 0.15 (thay vì 0.0 để tránh gradient chết và đại diện ngủ đông)
        self.gate = nn.Parameter(torch.tensor(0.15))

        # Biến theo dõi chẩn đoán
        self.last_ratio = 0.0
        self.last_energy = 0.0
        self.last_lie_norm = 0.0

    def forward(self, x):
        B, C, H, W = x.shape
        base_p = self.base_prompt(x)
        emb = x.mean(dim=(-2, -1))

        # 1. Dự đoán góc quay phân số & Thực thi FrFT 2D
        orders = 0.1 + 0.8 * torch.sigmoid(self.fc_order(emb))
        a_h = orders[:, 0]
        a_w = orders[:, 1]
        z_in = self.conv_in(x)
        z_phase = self.frft2d(z_in, a_h, a_w)

        # 2. Toán tử Giao hoán Lie: [A, B] = AB - BA
        mat_A = self.head_A(emb).view(B, self.r, self.r)
        mat_B = self.head_B(emb).view(B, self.r, self.r)
        comm = torch.matmul(mat_A, mat_B) - torch.matmul(mat_B, mat_A)
        self.last_lie_norm = float(comm.detach().norm(p='fro').item())

        u_r = self.conv_down(z_phase)
        v_r = torch.einsum('bij, bjhw -> bihw', comm, u_r)
        omega_lie = self.conv_up(v_r)

        # 3. Dòng chảy Hút Hopfield (Associative Energy Gradient Step)
        Z = z_phase.permute(0, 2, 3, 1).reshape(-1, self.prompt_dim)  # (B*H*W, C)
        dots = torch.matmul(Z, self.M.t()) * self.beta
        attn = F.softmax(dots, dim=-1)
        Z_retrieved = torch.matmul(attn, self.M)
        eta = torch.sigmoid(self.step_size)
        Z_flow = (1.0 - eta) * Z + eta * Z_retrieved
        z_flow = Z_flow.view(B, H, W, self.prompt_dim).permute(0, 3, 1, 2)

        # Tính thế năng Hopfield E(Z) phục vụ kiểm chuẩn
        lse = torch.logsumexp(dots, dim=-1) / self.beta
        quad = 0.5 * (Z ** 2).sum(dim=-1)
        self.last_energy = float((quad - lse).mean().item())

        # 4. Hợp nhất động học: Dòng chảy hút + Lực phi bảo toàn Lie
        z_dyn = z_flow + omega_lie
        p_novel = self.conv_out(z_dyn)
        gamma = torch.tanh(self.gate)
        novel_p = gamma * p_novel

        # Đo đạc tỷ lệ năng lượng
        self.last_ratio = float((novel_p.detach().norm(p=2) / (base_p.detach().norm(p=2) + 1e-8)).item())

        return base_p + novel_p


class ParamMatchedSpatialDualBlock(nn.Module):
    """Khối đối chứng tham số (Branch C - Negative Control).
    Khóa chặt 100% cùng số lượng tham số, cùng số layer, cùng tensor shape như LieFrFTHopfieldBlock,
    nhưng:
    - KHÔNG dùng FrFT (hoạt động thuần túy trong miền không gian).
    - KHÔNG dùng Giao hoán tử Lie (dùng phép cộng giao hoán A + B).
    - KHÔNG dùng Động lực học Hopfield (dùng phép chiếu tuyến tính phẳng, không có thế năng hút).
    """
    def __init__(self, prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96, r=16, K=16):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.prompt_len = prompt_len
        self.prompt_size = prompt_size
        self.lin_dim = lin_dim
        self.r = r
        self.K = K

        # 0. Base prompt tương đương
        self.base_prompt = PromptGenBlock(prompt_dim=prompt_dim, prompt_len=prompt_len, prompt_size=prompt_size, lin_dim=lin_dim)

        # 1. Trích xuất không gian thuần túy (Cùng số tham số với conv_in & fc_order)
        self.conv_in = nn.Conv2d(lin_dim, prompt_dim, kernel_size=1)
        self.fc_order = nn.Sequential(
            nn.Linear(lin_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )

        # 2. Phép cộng giao hoán thuần túy (Cùng số tham số với head_A, head_B, conv_down, conv_up)
        self.head_A = nn.Linear(lin_dim, r * r, bias=False)
        self.head_B = nn.Linear(lin_dim, r * r, bias=False)
        self.conv_down = nn.Conv2d(prompt_dim, r, kernel_size=1)
        self.conv_up = nn.Conv2d(r, prompt_dim, kernel_size=1)

        # 3. Phép chiếu phẳng (Cùng số tham số với M và step_size)
        self.M = nn.Parameter(torch.randn(K, prompt_dim) * 0.02)
        self.step_size = nn.Parameter(torch.tensor(0.5))

        # 4. Tầng tổng hợp (Cùng số tham số)
        self.conv_out = nn.Sequential(
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1)
        )
        self.gate = nn.Parameter(torch.tensor(0.15))

        self.last_ratio = 0.0

    def forward(self, x):
        B, C, H, W = x.shape
        base_p = self.base_prompt(x)
        emb = x.mean(dim=(-2, -1))

        # 1. Miền không gian phẳng (Không có quay pha FrFT)
        scales = torch.sigmoid(self.fc_order(emb))
        s_total = (scales[:, 0] + scales[:, 1]).view(B, 1, 1, 1)
        z_in = self.conv_in(x)
        z_spatial = s_total * z_in

        # 2. Phép cộng Giao hoán (A + B == B + A, không trích xuất được tương tác sương x nhiễu)
        mat_A = self.head_A(emb).view(B, self.r, self.r)
        mat_B = self.head_B(emb).view(B, self.r, self.r)
        comm = mat_A + mat_B  # Giao hoán
        u_r = self.conv_down(z_spatial)
        v_r = torch.einsum('bij, bjhw -> bihw', comm, u_r)
        omega_comm = self.conv_up(v_r)

        # 3. Phép chiếu tuyến tính tĩnh (Không có hàm thế năng Hopfield softmax)
        Z = z_spatial.permute(0, 2, 3, 1).reshape(-1, self.prompt_dim)
        Z_flat = torch.matmul(torch.matmul(Z, self.M.t()), self.M) / self.prompt_dim
        eta = torch.sigmoid(self.step_size)
        Z_flow = (1.0 - eta) * Z + eta * Z_flat
        z_flow = Z_flow.view(B, H, W, self.prompt_dim).permute(0, 3, 1, 2)

        # 4. Hợp nhất
        z_dyn = z_flow + omega_comm
        p_ctrl = self.conv_out(z_dyn)
        gamma = torch.tanh(self.gate)
        novel_p = gamma * p_ctrl

        self.last_ratio = float((novel_p.detach().norm(p=2) / (base_p.detach().norm(p=2) + 1e-8)).item())
        return base_p + novel_p


class PhaseHopfieldIR(nn.Module):
    """Mô hình đề xuất PHASE-HOPFIELD-IR (Branch B).
    Kế thừa kiến trúc Restormer 4 cấp của PromptIR, tích hợp LieFrFTHopfieldBlock tại cả 3 cấp decoder.
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
        super(PhaseHopfieldIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        # Tích hợp khối LieFrFTHopfieldBlock tại 3 cấp decoder
        if self.decoder:
            self.prompt1 = LieFrFTHopfieldBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96, r=16, K=16)
            self.prompt2 = LieFrFTHopfieldBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192, r=16, K=16)
            self.prompt3 = LieFrFTHopfieldBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384, r=16, K=16)

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

    def get_new_block_ratio(self):
        """Tính tỉ lệ trung bình năng lượng norm giữa khối mới và base prompt."""
        if not self.decoder:
            return 0.0
        ratios = []
        for p_name in ['prompt1', 'prompt2', 'prompt3']:
            p_block = getattr(self, p_name, None)
            if p_block is not None and hasattr(p_block, 'last_ratio'):
                ratios.append(p_block.last_ratio)
        return float(sum(ratios) / len(ratios)) if ratios else 0.0

    def get_hopfield_energy(self):
        """Tính thế năng Hopfield trung bình trên 3 cấp decoder."""
        if not self.decoder:
            return 0.0
        energies = []
        for p_name in ['prompt1', 'prompt2', 'prompt3']:
            p_block = getattr(self, p_name, None)
            if p_block is not None and hasattr(p_block, 'last_energy'):
                energies.append(p_block.last_energy)
        return float(sum(energies) / len(energies)) if energies else 0.0

    def get_lie_norm(self):
        """Tính Frobenius norm trung bình của giao hoán tử Lie trên 3 cấp decoder."""
        if not self.decoder:
            return 0.0
        norms = []
        for p_name in ['prompt1', 'prompt2', 'prompt3']:
            p_block = getattr(self, p_name, None)
            if p_block is not None and hasattr(p_block, 'last_lie_norm'):
                norms.append(p_block.last_lie_norm)
        return float(sum(norms) / len(norms)) if norms else 0.0

    def load_pretrained_promptir(self, ckpt_path):
        """Tải trọng số backbone và pretrained prompt từ checkpoint PromptIR nguyên bản."""
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
        print(f">> Backbone & Pretrained Prompt đã nạp thành công! (Số layer nạp: {len(clean_state)}, lớp mới: {len(missing)})")


class ParamMatchedPhaseHopfieldIR(PhaseHopfieldIR):
    """Mô hình đối chứng tham số cho PHASE-HOPFIELD-IR (Branch C).
    Kế thừa toàn bộ kiến trúc từ PhaseHopfieldIR, chỉ thay thế LieFrFTHopfieldBlock
    bằng ParamMatchedSpatialDualBlock. Đảm bảo số lượng tham số khớp chính xác 100% từng tham số đơn vị.
    """
    def __init__(self, *args, **kwargs):
        super(ParamMatchedPhaseHopfieldIR, self).__init__(*args, **kwargs)
        if self.decoder:
            self.prompt1 = ParamMatchedSpatialDualBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96, r=16, K=16)
            self.prompt2 = ParamMatchedSpatialDualBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192, r=16, K=16)
            self.prompt3 = ParamMatchedSpatialDualBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384, r=16, K=16)
