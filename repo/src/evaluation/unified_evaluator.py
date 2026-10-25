"""Module Đánh Giá Thống Nhất (Unified Evaluation Protocol).
Mã nhiệm vụ: P4-V2
Mục đích:
- Một hàm duy nhất dùng chung 100% cho mọi mô hình (PromptIR Baseline, FD-PromptIR, ParamMatchedPromptIR).
- Báo cáo chi tiết từng tác vụ riêng biệt: Noise, Rain, Haze, Hybrid và Mean.
- Giao thức chuẩn hóa đồng bộ: RGB, clamp [0, 1], uint8 round-trip đồng nhất.
- Hỗ trợ đánh giá cả Checkpoint tốt nhất trên VAL và Checkpoint cuối cùng.
"""

import math
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    # Kẹp khoảng hợp lệ [0, 1]
    pred_np = np.clip(pred_np, 0.0, 1.0)
    gt_np = np.clip(gt_np, 0.0, 1.0)

    if use_uint8_rounding:
        # Làm tròn uint8 chuẩn hóa benchmark thị giác máy tính
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


class UnifiedEvaluator:
    """Bộ đánh giá chuẩn hóa cho mọi mô hình AiOIR."""

    def __init__(self, device: torch.device, use_uint8: bool = True):
        self.device = device
        self.use_uint8 = use_uint8

    @torch.no_grad()
    def evaluate(self, model: nn.Module, val_dataset) -> Dict[str, float]:
        """Thực thi đánh giá thống nhất trên tập validation.
        val_dataset trả về tuple: (deg_img, clean_img, task_name)
        với task_name thuộc {'noise', 'rain', 'haze', 'hybrid'}.
        """
        model.eval()
        task_psnr: Dict[str, List[float]] = {'noise': [], 'rain': [], 'haze': [], 'hybrid': []}
        task_ssim: Dict[str, List[float]] = {'noise': [], 'rain': [], 'haze': [], 'hybrid': []}

        for idx in range(len(val_dataset)):
            deg, clean, task = val_dataset[idx]
            deg = deg.unsqueeze(0).to(self.device)
            clean = clean.unsqueeze(0).to(self.device)

            pred = forward_padded(model, deg, factor=64)
            pred = torch.clamp(pred, 0.0, 1.0)

            psnr, ssim = compute_metrics(pred, clean, use_uint8_rounding=self.use_uint8)
            t_key = task.lower()
            if t_key not in task_psnr:
                t_key = 'hybrid'

            task_psnr[t_key].append(psnr)
            task_ssim[t_key].append(ssim)

        # Tính trung bình từng tác vụ
        results = {}
        all_psnr = []
        all_ssim = []

        for task in ['noise', 'rain', 'haze', 'hybrid']:
            psnrs = task_psnr[task]
            ssims = task_ssim[task]
            if len(psnrs) > 0:
                mean_p = float(np.mean(psnrs))
                mean_s = float(np.mean(ssims))
            else:
                mean_p, mean_s = 0.0, 0.0
            results[f'psnr_{task}'] = round(mean_p, 2)
            results[f'ssim_{task}'] = round(mean_s, 4)
            if len(psnrs) > 0:
                all_psnr.extend(psnrs)
                all_ssim.extend(ssims)

        results['psnr_mean'] = round(float(np.mean(all_psnr)) if all_psnr else 0.0, 2)
        results['ssim_mean'] = round(float(np.mean(all_ssim)) if all_ssim else 0.0, 4)
        return results

    def format_table(self, results: Dict[str, float], title: str = "Evaluation Results") -> str:
        """Định dạng kết quả thành bảng Markdown chuyên nghiệp."""
        md = f"### {title}\n\n"
        md += "| Chỉ số đo lường | Nhiễu (Noise) | Mưa (Rain) | Sương mù (Haze) | Hỗn hợp (Hybrid) | **Trung bình (Mean)** |\n"
        md += "|---|:---:|:---:|:---:|:---:|:---:|\n"
        md += f"| **PSNR (dB)** | {results['psnr_noise']} | {results['psnr_rain']} | {results['psnr_haze']} | {results['psnr_hybrid']} | **{results['psnr_mean']}** |\n"
        md += f"| **SSIM** | {results['ssim_noise']} | {results['ssim_rain']} | {results['ssim_haze']} | {results['ssim_hybrid']} | **{results['ssim_mean']}** |\n"
        return md
