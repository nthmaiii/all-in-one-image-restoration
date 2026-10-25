"""P4-T3: Script huấn luyện Pilot Thăm Dò & Đối Chứng Khoa Học Sòng Phẳng.
So sánh trực tiếp:
1. Baseline PromptIR (chuẩn)
2. Proposed FD-PromptIR (Phân tách miền tần số + Ràng buộc trực giao OPR)
Trên cùng một ngân sách huấn luyện (3,000 iterations, batch size 4, patch 128x128).
Hỗ trợ: AMP FP16, Warm-start từ checkpoint chính thức, Checkpointing & Resume, Logging tự động.
"""

import sys
import os
import time
import math
import random
import argparse
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines' / 'promptir'))

# -------------------------------------------------------------
# 1. Dataset Sinh Suy Thoái Hỗn Hợp Trực Tiếp (On-the-Fly Synthesis)
# -------------------------------------------------------------
class AiOIRPilotDataset(Dataset):
    """Dataset huấn luyện đa suy thoái cân bằng:
    - 25% Nhiễu Gaussian ngẫu nhiên (sigma in [15, 50])
    - 25% Vệt mưa (Rain streaks)
    - 25% Sương mù (Koschmieder Haze)
    - 25% Suy thoái hỗn hợp (Nhiễu + Sương mù)
    """
    def __init__(self, image_paths, patch_size=128, is_train=True):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths) * (10 if self.is_train else 1)

    def synthesize_degradation(self, clean_np, mode=None):
        """clean_np: float32 in [0, 1], shape (H, W, 3)"""
        if mode is None:
            mode = random.randint(0, 3)
        h, w, c = clean_np.shape

        if mode == 0:
            # 1. Nhiễu Gaussian
            sigma = random.uniform(15, 50) / 255.0
            noise = np.random.normal(0, sigma, clean_np.shape).astype(np.float32)
            degraded = np.clip(clean_np + noise, 0.0, 1.0)
            tag = "noise"

        elif mode == 1:
            # 2. Vệt mưa nhân tạo (Linear directional motion streaks)
            angle = random.uniform(70, 110)
            length = random.randint(15, 25)
            kernel = np.zeros((length, length), dtype=np.float32)
            kernel[length // 2, :] = 1.0
            M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (length, length))
            kernel = kernel / (np.sum(kernel) + 1e-6)

            rain_noise = np.random.uniform(0, 1, (h, w)).astype(np.float32)
            rain_pattern = cv2.filter2D(rain_noise, -1, kernel)
            rain_pattern = np.clip(rain_pattern * 0.8, 0, 1)
            rain_pattern = np.expand_dims(rain_pattern, -1)
            degraded = np.clip(clean_np + rain_pattern * 0.6, 0.0, 1.0)
            tag = "rain"

        elif mode == 2:
            # 3. Sương mù (Mô hình tán xạ Koschmieder I = J*t + A*(1-t))
            t = random.uniform(0.4, 0.75)
            A = random.uniform(0.7, 0.95)
            degraded = np.clip(clean_np * t + A * (1.0 - t), 0.0, 1.0)
            tag = "haze"

        else:
            # 4. Suy thoái hỗn hợp: Nhiễu + Sương mù (Điểm mù đo được 16.79 dB của PromptIR)
            t = random.uniform(0.5, 0.7)
            A = random.uniform(0.75, 0.95)
            hazed = clean_np * t + A * (1.0 - t)
            sigma = random.uniform(20, 35) / 255.0
            noise = np.random.normal(0, sigma, clean_np.shape).astype(np.float32)
            degraded = np.clip(hazed + noise, 0.0, 1.0)
            tag = "hybrid"

        return degraded, tag

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_paths)
        img_p = self.image_paths[real_idx]

        try:
            with Image.open(img_p) as pil_img:
                clean_img = pil_img.convert('RGB')
        except Exception:
            clean_img = Image.new('RGB', (self.patch_size, self.patch_size), (128, 128, 128))

        clean_np = np.array(clean_img, dtype=np.float32) / 255.0

        # Cắt patch ngẫu nhiên
        h, w, _ = clean_np.shape
        if h < self.patch_size or w < self.patch_size:
            clean_np = cv2.resize(clean_np, (max(w, self.patch_size), max(h, self.patch_size)))
            h, w, _ = clean_np.shape

        if self.is_train:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            mode = random.randint(0, 3)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2
            mode = real_idx % 4

        patch_clean = clean_np[top:top + self.patch_size, left:left + self.patch_size]

        # Sinh ảnh suy thoái
        patch_degraded, tag = self.synthesize_degradation(patch_clean, mode=mode)

        tensor_clean = torch.from_numpy(patch_clean).permute(2, 0, 1).float()
        tensor_degraded = torch.from_numpy(patch_degraded).permute(2, 0, 1).float()

        return tensor_degraded, tensor_clean, tag

# -------------------------------------------------------------
# 2. Khởi Tạo Mô Hình & Loss Function
# -------------------------------------------------------------
def build_model(model_name, init_weights=None, device='cuda'):
    m_name = model_name.lower().strip()
    if 'fd' in m_name:
        try:
            from src.models.fd_promptir import FDPromptIR
        except ImportError:
            from repo.src.models.fd_promptir import FDPromptIR

        print("[MÔ HÌNH] Khởi tạo FD-PromptIR (Frequency-Decoupled Dynamic Prompting)...")
        model = FDPromptIR(decoder=True)
        if init_weights and Path(init_weights).exists():
            model.load_pretrained_promptir(init_weights)
    else:
        from baselines.promptir.net.model import PromptIR
        print("[MÔ HÌNH] Khởi tạo Baseline PromptIR (Chuẩn)...")
        model = PromptIR(decoder=True)
        if init_weights and Path(init_weights).exists():
            print(f"Nạp trọng số baseline từ: {init_weights}")
            ckpt = torch.load(init_weights, map_location='cpu')
            sd = ckpt.get('state_dict', ckpt.get('params', ckpt))
            clean_sd = {k.replace('net.', '').replace('module.', ''): v for k, v in sd.items()}
            model.load_state_dict(clean_sd, strict=False)

    return model.to(device)

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

# -------------------------------------------------------------
# 3. Hàm Đánh Giá Nhanh (Validation)
# -------------------------------------------------------------
def evaluate_model(model, val_loader, device='cuda'):
    model.eval()
    psnr_scores = {'all': [], 'noise': [], 'rain': [], 'haze': [], 'hybrid': []}

    with torch.no_grad():
        for deg, clean, tags in val_loader:
            deg, clean = deg.to(device), clean.to(device)
            restored = model(deg)
            restored = torch.clamp(restored, 0.0, 1.0)

            for i in range(deg.size(0)):
                t = tags[i]
                r_np = restored[i].cpu().permute(1, 2, 0).numpy()
                c_np = clean[i].cpu().permute(1, 2, 0).numpy()

                mse = np.mean((r_np - c_np) ** 2)
                psnr = 40.0 if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))
                psnr_scores['all'].append(psnr)
                if t in psnr_scores:
                    psnr_scores[t].append(psnr)

    res = {k: float(np.mean(v)) if v else 0.0 for k, v in psnr_scores.items()}
    return res

# -------------------------------------------------------------
# 4. Huấn Luyện Chính (Training Loop)
# -------------------------------------------------------------
def train_pilot():
    parser = argparse.ArgumentParser(description="Chạy Pilot Experiment so sánh PromptIR vs FD-PromptIR")
    parser.add_argument('--model', type=str, default='fd_promptir', choices=['promptir', 'fd_promptir'])
    parser.add_argument('--init_weights', type=str, default='baselines/weights/promptir_all.ckpt')
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='data/test')
    parser.add_argument('--output_dir', type=str, default='runs/pilot')
    parser.add_argument('--eval_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== BẮT ĐẦU HUẤN LUYỆN PILOT ({args.model.upper()}) TRÊN {device} ===")

    # Thu thập ảnh sạch từ các tập benchmark
    root_data = Path(args.data_dir)
    clean_candidates = []
    for sub in [root_data / 'denoise' / 'bsd68', root_data / 'denoise' / 'kodak24', root_data / 'derain' / 'Rain100L' / 'target']:
        if sub.exists():
            clean_candidates.extend(list(sub.glob('*.png')) + list(sub.glob('*.jpg')))

    if not clean_candidates:
        print("[!] Không tìm thấy ảnh sạch trong data/test. Quét toàn bộ data/...")
        clean_candidates = list(Path('data').rglob('*.png'))[:100]

    print(f"Tổng số ảnh sạch phục vụ sinh suy thoái: {len(clean_candidates)} ảnh.")
    random.seed(42)
    random.shuffle(clean_candidates)
    split_idx = int(len(clean_candidates) * 0.8)
    train_imgs = clean_candidates[:split_idx]
    val_imgs = clean_candidates[split_idx:]

    train_ds = AiOIRPilotDataset(train_imgs, patch_size=args.patch_size, is_train=True)
    val_ds = AiOIRPilotDataset(val_imgs, patch_size=args.patch_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = build_model(args.model, init_weights=args.init_weights, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_charb = CharbonnierLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / 'latest_checkpoint.pth'
    best_ckpt_path = out_dir / 'best_model.pth'

    start_iter = 0
    best_hybrid_psnr = -1.0

    if args.resume and ckpt_path.exists():
        print(f"Phục hồi trạng thái huấn luyện từ {ckpt_path}...")
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])
        start_iter = checkpoint['iteration']
        best_hybrid_psnr = checkpoint.get('best_hybrid_psnr', -1.0)
        print(f">> Phục hồi thành công tại iteration {start_iter} (Best Hybrid PSNR: {best_hybrid_psnr:.2f} dB)")

    print(f"Tiến hành huấn luyện từ iteration {start_iter + 1} đến {args.iterations}...")
    train_iter = iter(train_loader)
    loss_history = []

    start_time = time.time()
    for current_iter in range(start_iter + 1, args.iterations + 1):
        model.train()
        try:
            deg, clean, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            deg, clean, _ = next(train_iter)

        deg, clean = deg.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        optimizer.zero_grad()

        # Cosine Annealing Learning Rate
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * current_iter / args.iterations))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            restored = model(deg)
            loss_recon = criterion_charb(restored, clean)

            # Bổ sung Orthogonal Loss nếu là FD-PromptIR
            if hasattr(model, 'get_orthogonal_loss'):
                loss_orth = model.get_orthogonal_loss()
                total_loss = loss_recon + 0.01 * loss_orth
            else:
                loss_orth = torch.tensor(0.0)
                total_loss = loss_recon

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_history.append(total_loss.item())

        if current_iter % 50 == 0 or current_iter == 1:
            elapsed = time.time() - start_time
            print(f"Iter [{current_iter:4d}/{args.iterations}] | Recon Loss: {loss_recon.item():.4f} | Orth Loss: {loss_orth.item():.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")

        # Đánh giá Validation định kỳ
        if current_iter % args.eval_freq == 0 or current_iter == args.iterations:
            val_scores = evaluate_model(model, val_loader, device=device)
            print(f"\n--- [VAL ITER {current_iter}] PSNR Tổng: {val_scores['all']:.2f} dB | Noise: {val_scores['noise']:.2f} | Rain: {val_scores['rain']:.2f} | Haze: {val_scores['haze']:.2f} | HYBRID: {val_scores['hybrid']:.2f} dB ---")

            if val_scores['hybrid'] > best_hybrid_psnr:
                best_hybrid_psnr = val_scores['hybrid']
                torch.save({'model_state': model.state_dict(), 'iteration': current_iter, 'val_scores': val_scores}, str(best_ckpt_path))
                print(f">> [KỶ LỤC MỚI] Đã lưu model tốt nhất vào: {best_ckpt_path}")

        # Lưu Checkpoint định kỳ
        if current_iter % args.save_freq == 0 or current_iter == args.iterations:
            torch.save({
                'iteration': current_iter,
                'model_state': model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'best_hybrid_psnr': best_hybrid_psnr
            }, str(ckpt_path))
            print(f"Đã lưu checkpoint trạng thái tại iter {current_iter} -> {ckpt_path}")

    # Đánh giá cuối cùng và ghi nhận kết quả vào CSV
    final_scores = evaluate_model(model, val_loader, device=device)
    csv_file = Path('results/pilot_results.csv')
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_file.exists()

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'iterations', 'psnr_all', 'psnr_noise', 'psnr_rain', 'psnr_haze', 'psnr_hybrid', 'best_hybrid_psnr'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'model': args.model,
            'iterations': args.iterations,
            'psnr_all': round(final_scores['all'], 2),
            'psnr_noise': round(final_scores['noise'], 2),
            'psnr_rain': round(final_scores['rain'], 2),
            'psnr_haze': round(final_scores['haze'], 2),
            'psnr_hybrid': round(final_scores['hybrid'], 2),
            'best_hybrid_psnr': round(best_hybrid_psnr, 2)
        })

    print(f"\n[HOÀN TẤT HUẤN LUYỆN PILOT {args.model.upper()}] Kết quả đã lưu vào: {csv_file.resolve()}")

if __name__ == '__main__':
    train_pilot()
