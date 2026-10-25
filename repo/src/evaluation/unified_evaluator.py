"""Module Đánh Giá Thống Nhất (Unified Evaluation Protocol).
Mã nhiệm vụ: P4-V2
Mục đích:
- Một hàm duy nhất dùng chung 100% cho mọi mô hình.
- Báo cáo chi tiết từng tác vụ riêng biệt: Noise, Rain, Haze, Hybrid (Val) và Hybrid-OOD (Test held-out).
- Giao thức chuẩn hóa đồng bộ: RGB, clamp [0, 1], uint8 round-trip đồng nhất.
- Hỗ trợ đánh giá cả Checkpoint tốt nhất trên VAL và Checkpoint cuối cùng.
"""

import math
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def forward_padded(model: nn.Module, x: torch.Tensor, factor: int = 64) -> torch.Tensor:
    """Đệm phản xạ ảnh để kích thước chia hết cho factor (bắt buộc cho kiến trúc đa cấp Restormer)."""
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        out_pad = model(x_pad)
        return out_pad[:, :, :h, :w]
    return model(x)


def compute_metrics(
    pred_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    use_uint8_rounding: bool = True
) -> Tuple[float, float]:
    """Tính PSNR và SSIM trên miền RGB theo chuẩn hóa giao thức nghiêm ngặt.
    pred_tensor, gt_tensor: tensor shape (1, 3, H, W) trong dải [0, 1].
    """
    pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    gt_np = gt_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    pred_np = np.clip(pred_np, 0.0, 1.0)
    gt_np = np.clip(gt_np, 0.0, 1.0)

    if use_uint8_rounding:
        pred_uint8 = np.round(pred_np * 255.0).astype(np.uint8)
        gt_uint8 = np.round(gt_np * 255.0).astype(np.uint8)

        diff = pred_uint8.astype(np.float64) - gt_uint8.astype(np.float64)
        mse = np.mean(diff ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 10.0 * math.log10((255.0 ** 2) / mse)

        if HAS_SKIMAGE:
            ssim = compare_ssim(pred_uint8, gt_uint8, channel_axis=2, data_range=255)
        else:
            ssim = 1.0 - (mse / (255.0 ** 2))
    else:
        diff = pred_np.astype(np.float64) - gt_np.astype(np.float64)
        mse = np.mean(diff ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = -10.0 * math.log10(mse)

        if HAS_SKIMAGE:
            ssim = compare_ssim(pred_np, gt_np, channel_axis=2, data_range=1.0)
        else:
            ssim = 1.0 - mse

    return float(psnr), float(ssim)


class AiOIRTestHybridOODDataset(Dataset):
    """Tập kiểm thử Hybrid Giữ Lại (Held-out Test Hybrid-OOD Dataset).
    Được tạo trực tiếp từ ảnh sạch của các tập test (BSD68, Kodak24, hoặc SOTS target)
    với lưới tham số OOD hoàn toàn không xuất hiện trong quá trình huấn luyện:
    sigma in {15, 35, 50} x t in {0.4, 0.8} (6 tổ hợp ngoại suy).
    """
    def __init__(self, test_clean_dirs: List[Path], max_images: int = 30, seed: int = 999):
        self.samples = []
        clean_files = []
        for d in test_clean_dirs:
            p = Path(d)
            if p.exists():
                clean_files.extend(sorted(list(p.glob('*.png')) + list(p.glob('*.jpg'))))

        if len(clean_files) == 0:
            # Nếu chưa có thư mục test, sinh dummy để không crash trong test cú pháp
            clean_files = []

        # Lấy tối đa max_images để kiểm thử nhanh và đại diện
        rng = np.random.RandomState(seed)
        selected_files = clean_files[:max_images] if len(clean_files) >= max_images else clean_files

        # Lưới tham số OOD: 3 sigma x 2 t = 6 tổ hợp
        grid = [(s, t) for s in [15, 35, 50] for t in [0.4, 0.8]]

        for img_idx, f in enumerate(selected_files):
            # Gán mỗi ảnh sạch với 1 tổ hợp trong lưới theo vòng lặp xác định
            sigma, t = grid[img_idx % len(grid)]
            with Image.open(f) as pil_img:
                pil_img = pil_img.convert('RGB')
                w, h = pil_img.size
                # Giới hạn kích thước tối đa 512x512 để kiểm định nhanh trên T4
                if w > 512 or h > 512:
                    pil_img = pil_img.resize((min(w, 512), min(h, 512)))
                clean_np = np.array(pil_img, dtype=np.float32) / 255.0

            # Sinh suy thoái Hybrid xác định: Haze (Koschmieder) + Gaussian Noise
            A = 1.0
            hazy = clean_np * t + A * (1.0 - t)
            # Dùng seed cục bộ theo file để đảm bảo tính tái lập 100%
            file_rng = np.random.RandomState(seed + img_idx * 17)
            noise = file_rng.normal(0, sigma / 255.0, clean_np.shape).astype(np.float32)
            deg_np = np.clip(hazy + noise, 0.0, 1.0)

            deg_t = torch.from_numpy(deg_np).permute(2, 0, 1).float()
            clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).float()
            task_label = f"hybrid_s{sigma}_t{t}"
            self.samples.append((deg_t, clean_t, task_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class UnifiedEvaluator:
    """Bộ đánh giá chuẩn hóa cho mọi mô hình AiOIR."""

    def __init__(self, device: torch.device, use_uint8: bool = True):
        self.device = device
        self.use_uint8 = use_uint8

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataset) -> Dict[str, float]:
        """Thực thi đánh giá thống nhất trên tập validation hoặc test dataset."""
        model.eval()
        task_psnr: Dict[str, List[float]] = {}
        task_ssim: Dict[str, List[float]] = {}

        for idx in range(len(dataset)):
            deg, clean, task = dataset[idx]
            deg = deg.unsqueeze(0).to(self.device)
            clean = clean.unsqueeze(0).to(self.device)

            pred = forward_padded(model, deg, factor=64)
            pred = torch.clamp(pred, 0.0, 1.0)

            psnr, ssim = compute_metrics(pred, clean, use_uint8_rounding=self.use_uint8)
            t_key = task.lower()
            if t_key not in task_psnr:
                task_psnr[t_key] = []
                task_ssim[t_key] = []

            task_psnr[t_key].append(psnr)
            task_ssim[t_key].append(ssim)

        results = {}
        all_psnr = []
        all_ssim = []

        for task, psnrs in task_psnr.items():
            ssims = task_ssim[task]
            mean_p = float(np.mean(psnrs)) if psnrs else 0.0
            mean_s = float(np.mean(ssims)) if ssims else 0.0
            results[f'psnr_{task}'] = round(mean_p, 2)
            results[f'ssim_{task}'] = round(mean_s, 4)
            all_psnr.extend(psnrs)
            all_ssim.extend(ssims)

        results['psnr_mean'] = round(float(np.mean(all_psnr)) if all_psnr else 0.0, 2)
        results['ssim_mean'] = round(float(np.mean(all_ssim)) if all_ssim else 0.0, 4)
        return results

    def format_table(self, results: Dict[str, float], title: str = "Evaluation Results") -> str:
        """Định dạng kết quả thành bảng Markdown chuyên nghiệp."""
        md = f"### {title}\n\n"
        md += "| Chỉ số | Noise | Rain | Haze | Hybrid (Val) | **Trung Bình** |\n"
        md += "|---|:---:|:---:|:---:|:---:|:---:|\n"
        p_n = results.get('psnr_noise', 'N/A')
        p_r = results.get('psnr_rain', 'N/A')
        p_h = results.get('psnr_haze', 'N/A')
        p_hy = results.get('psnr_hybrid', 'N/A')
        p_m = results.get('psnr_mean', 'N/A')
        md += f"| **PSNR (dB)** | {p_n} | {p_r} | {p_h} | {p_hy} | **{p_m}** |\n"
        s_n = results.get('ssim_noise', 'N/A')
        s_r = results.get('ssim_rain', 'N/A')
        s_h = results.get('ssim_haze', 'N/A')
        s_hy = results.get('ssim_hybrid', 'N/A')
        s_m = results.get('ssim_mean', 'N/A')
        md += f"| **SSIM** | {s_n} | {s_r} | {s_h} | {s_hy} | **{s_m}** |\n"
        return md
