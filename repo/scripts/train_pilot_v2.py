"""Script Huấn Luyện Thăm Dò Khóa Trước v2 (Pilot Training v2).
Mã nhiệm vụ: P4-V2
Đặc tính khoa học:
- Hỗ trợ 4 nhánh đối chứng sòng phẳng:
  + Branch A: PromptIR Pretrained Fine-tuned.
  + Branch B: FD-PromptIR (Đề xuất: Tách tần số + OPR cải tiến).
  + Branch C: ParamMatchedPromptIR (Đối chứng tham số: Cùng số layer/param nhưng không tách tần số).
  + Branch D: PromptIR Zero-Shot (Đo sàn baseline không qua fine-tune).
- Kiểm tra khẳng định (assert) băm SHA-256 chống rò rỉ dữ liệu trước khi train.
- Cùng seed, cùng thứ tự batch, cùng ngân sách 1,500 iters.
- Đánh giá thống nhất bằng UnifiedEvaluator trên 4 tác vụ riêng biệt và trung bình.
- Báo cáo và lưu trữ cả Checkpoint tốt nhất trên VAL và Checkpoint cuối cùng.
- Đo đạc chính xác số giờ GPU thực tế (GPU hours).
"""

import os
import sys
import time
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Thiết lập đường dẫn import
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.leakage_checker import verify_zero_leakage
from src.evaluation.unified_evaluator import UnifiedEvaluator, forward_padded

def get_prompt_cosine_stats(model) -> str:
    """Tính chỉ số Cosine Similarity thực tế (không nhân lambda) giữa các prompt tokens ở 3 cấp decoder."""
    stats = []
    for p_name in ['prompt1', 'prompt2', 'prompt3']:
        p_block = getattr(model, p_name, None)
        if p_block is None:
            continue
        bank = getattr(p_block, 'prompt_low', getattr(p_block, 'prompt_bank1', None))
        if bank is not None:
            N = bank.shape[1]
            v = bank.squeeze(0).mean(dim=(-2, -1))
            norm_v = F.normalize(v, p=2, dim=1)
            gram = torch.mm(norm_v, norm_v.t())
            off_diag = gram[~torch.eye(N, dtype=bool, device=bank.device)].abs()
            stats.append(f"{off_diag.mean().item():.3f}")
    return "/".join(stats) if stats else "N/A"


def set_seed(seed: int = 42):
    """Khóa toàn bộ hạt giống ngẫu nhiên để bảo đảm khả năng tái lập 100%."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AiOIRDatasetV2(Dataset):
    """Dataset sinh suy thoái trực tiếp từ ảnh sạch của tập train_v2."""
    def __init__(self, clean_dir: Path, patch_size: int = 128, length: int = 1500 * 4):
        self.clean_dir = Path(clean_dir)
        self.patch_size = patch_size
        self.length = length
        self.image_files = sorted(list(self.clean_dir.glob('*.png')) + list(self.clean_dir.glob('*.jpg')))
        if len(self.image_files) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh sạch nào trong {self.clean_dir}")

    def __len__(self):
        return self.length

    def synthesize_degradation(self, clean_np: np.ndarray, mode: int) -> Tuple[np.ndarray, str]:
        """Sinh 4 kiểu suy thoái trực tiếp trên ảnh sạch:
        0: Gaussian Noise
        1: Rain Streaks
        2: Koschmieder Haze
        3: Hybrid (Noise + Haze)
        """
        img = clean_np.copy()
        h, w, _ = img.shape

        if mode == 0:
            # Mode 0: Nhiễu Gaussian
            sigma = random.uniform(15.0, 50.0) / 255.0
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            deg = np.clip(img + noise, 0.0, 1.0)
            return deg, 'noise'

        elif mode == 1:
            # Mode 1: Vệt mưa mô phỏng (Rain Streaks)
            streak = np.zeros((h, w), dtype=np.float32)
            num_streaks = random.randint(150, 300)
            for _ in range(num_streaks):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 25)
                length = random.randint(10, 25)
                intensity = random.uniform(0.3, 0.7)
                for i in range(length):
                    if y + i < h and x + int(i * 0.2) < w:
                        streak[y + i, x + int(i * 0.2)] = intensity
            streak_3d = np.repeat(streak[:, :, np.newaxis], 3, axis=2)
            deg = np.clip(img * 0.85 + streak_3d, 0.0, 1.0)
            return deg, 'rain'

        elif mode == 2:
            # Mode 2: Sương mù mô hình Koschmieder I = J*t + A*(1-t)
            t = random.uniform(0.4, 0.8)
            A = 1.0
            deg = np.clip(img * t + A * (1.0 - t), 0.0, 1.0)
            return deg, 'haze'

        else:
            # Mode 3: Suy thoái hỗn hợp (Hybrid: Noise sigma=25 + Haze t=0.6)
            t = 0.6
            A = 1.0
            hazy = img * t + A * (1.0 - t)
            sigma = 25.0 / 255.0
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            deg = np.clip(hazy + noise, 0.0, 1.0)
            return deg, 'hybrid'

    def __getitem__(self, idx):
        img_path = self.image_files[idx % len(self.image_files)]
        with Image.open(img_path) as pil_img:
            img = np.array(pil_img.convert('RGB'), dtype=np.float32) / 255.0

        h, w, _ = img.shape
        ps = self.patch_size
        if h > ps and w > ps:
            y = random.randint(0, h - ps)
            x = random.randint(0, w - ps)
            clean_patch = img[y:y+ps, x:x+ps, :]
        else:
            clean_patch = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((ps, ps)), dtype=np.float32) / 255.0

        # Augmentation: Random Flip
        if random.random() > 0.5:
            clean_patch = np.fliplr(clean_patch)
        if random.random() > 0.5:
            clean_patch = np.flipud(clean_patch)

        mode = random.randint(0, 3)
        deg_patch, task_name = self.synthesize_degradation(clean_patch, mode)

        deg_t = torch.from_numpy(deg_patch).permute(2, 0, 1).float()
        clean_t = torch.from_numpy(clean_patch).permute(2, 0, 1).float()
        return deg_t, clean_t, task_name


class AiOIRValDatasetV2(Dataset):
    """Dataset kiểm định cố định được sinh xác định (deterministic) từ val_dir."""
    def __init__(self, val_clean_dir: Path, samples_per_task: int = 10, seed: int = 999):
        self.samples = []
        files = sorted(list(val_clean_dir.glob('*.png')) + list(val_clean_dir.glob('*.jpg')))
        if len(files) == 0:
            raise RuntimeError(f"Không có ảnh val trong {val_clean_dir}")

        rng = np.random.RandomState(seed)
        tasks = ['noise', 'rain', 'haze', 'hybrid']

        for task_idx, task_name in enumerate(tasks):
            for i in range(samples_per_task):
                f = files[(task_idx * samples_per_task + i) % len(files)]
                with Image.open(f) as pil_img:
                    # Resize về kích thước chuẩn hóa 256x256 để kiểm định nhanh và ổn định
                    pil_img = pil_img.convert('RGB').resize((256, 256))
                    clean_np = np.array(pil_img, dtype=np.float32) / 255.0

                h, w, _ = clean_np.shape
                if task_name == 'noise':
                    noise = rng.normal(0, 25.0 / 255.0, clean_np.shape).astype(np.float32)
                    deg_np = np.clip(clean_np + noise, 0.0, 1.0)
                elif task_name == 'rain':
                    streak = np.zeros((h, w), dtype=np.float32)
                    for _ in range(200):
                        x = rng.randint(0, w - 1)
                        y = rng.randint(0, h - 25)
                        for l in range(18):
                            if y + l < h and x + int(l * 0.2) < w:
                                streak[y + l, x + int(l * 0.2)] = 0.5
                    deg_np = np.clip(clean_np * 0.85 + np.repeat(streak[:, :, np.newaxis], 3, axis=2), 0.0, 1.0)
                elif task_name == 'haze':
                    t = 0.6
                    deg_np = np.clip(clean_np * t + 1.0 * (1.0 - t), 0.0, 1.0)
                else:  # hybrid
                    t = 0.6
                    hazy = clean_np * t + 1.0 * (1.0 - t)
                    noise = rng.normal(0, 25.0 / 255.0, clean_np.shape).astype(np.float32)
                    deg_np = np.clip(hazy + noise, 0.0, 1.0)

                deg_t = torch.from_numpy(deg_np).permute(2, 0, 1).float()
                clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).float()
                self.samples.append((deg_t, clean_t, task_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_model(branch: str, init_weights: str, device: torch.device):
    """Khởi tạo mô hình tương ứng cho từng nhánh đối chứng."""
    branch = branch.upper()
    print(f"\n[KHỞI TẠO MÔ HÌNH NHÁNH {branch}]")

    if branch in ['A', 'D']:
        from baselines.promptir.net.model import PromptIR
        model = PromptIR(decoder=True)
        if init_weights and Path(init_weights).exists():
            print(f"Nạp trọng số PromptIR từ: {init_weights}")
            ckpt = torch.load(init_weights, map_location='cpu')
            state = ckpt.get('state_dict', ckpt.get('params', ckpt))
            clean_state = {k.replace('net.', '').replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(clean_state, strict=False)

    elif branch == 'B':
        from src.models.fd_promptir import FDPromptIR
        model = FDPromptIR(decoder=True)
        if init_weights and Path(init_weights).exists():
            model.load_pretrained_promptir(init_weights)

    elif branch == 'C':
        from src.models.param_matched_promptir import ParamMatchedPromptIR
        model = ParamMatchedPromptIR(decoder=True)
        if init_weights and Path(init_weights).exists():
            model.load_pretrained_promptir(init_weights)

    else:
        raise ValueError(f"Không nhận diện được nhánh '{branch}'. Hợp lệ: A, B, C, D.")

    return model.to(device)


def run_experiment():
    parser = argparse.ArgumentParser(description="Huấn luyện Pilot v2 đối chứng sòng phẳng")
    parser.add_argument('--branch', type=str, required=True, choices=['A', 'B', 'C', 'D'], help="Nhánh thực nghiệm (A, B, C, D)")
    parser.add_argument('--seed', type=int, default=42, help="Hạt giống ngẫu nhiên (seed)")
    parser.add_argument('--iters', type=int, default=1500, help="Số iterations huấn luyện")
    parser.add_argument('--batch_size', type=int, default=4, help="Kích thước batch")
    parser.add_argument('--lr', type=float, default=2e-4, help="Tốc độ học")
    parser.add_argument('--lambda_orth', type=float, default=0.05, help="Hệ số phạt trực giao OPR")
    parser.add_argument('--val_freq', type=int, default=250, help="Tần suất đánh giá validation")
    parser.add_argument('--init_weights', type=str, default="baselines/weights/promptir_all.ckpt", help="Đường dẫn pretrained")
    parser.add_argument('--train_dir', type=str, default="data/train_v2/clean", help="Thư mục ảnh train sạch")
    parser.add_argument('--val_dir', type=str, default="data/val_v2/clean", help="Thư mục ảnh val sạch")
    parser.add_argument('--test_dir', type=str, default="data/test", help="Thư mục ảnh test")
    parser.add_argument('--output_dir', type=str, default="runs/pilot_v2", help="Thư mục lưu log và checkpoints")
    args = parser.parse_args()

    # 1. Khóa seed
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = (REPO_ROOT / args.train_dir).resolve()
    val_path = (REPO_ROOT / args.val_dir).resolve()
    test_path = (REPO_ROOT / args.test_dir).resolve()

    # 2. KHẲNG ĐỊNH CHỐNG RÒ RỈ DỮ LIỆU (ASSERTION ZERO-LEAKAGE)
    print("=" * 60)
    print(f"BẮT ĐẦU THỰC NGHIỆM PILOT V2 — NHÁNH {args.branch} (SEED {args.seed}) TRÊN {device}")
    print("=" * 60)
    if test_path.exists():
        verify_zero_leakage([train_path], [val_path], [test_path], verbose=True)
    else:
        print("[LƯU Ý] Không tìm thấy thư mục test để đối chiếu băm.")

    # 3. Tạo thư mục lưu kết quả
    run_dir = (REPO_ROOT / args.output_dir / f"branch_{args.branch}_seed_{args.seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # 4. Khởi tạo mô hình
    weights_path = (REPO_ROOT / args.init_weights).resolve()
    model = build_model(args.branch, str(weights_path), device)

    # 5. Khởi tạo Dataset & Evaluator
    evaluator = UnifiedEvaluator(device=device, use_uint8=True)
    val_dataset = AiOIRValDatasetV2(val_path, samples_per_task=10, seed=args.seed)

    # Nếu là nhánh D: Đánh giá Zero-shot và kết thúc
    if args.branch == 'D':
        print("\n[NHÁNH D] Đánh giá Zero-shot của PromptIR gốc trên tập Val v2...")
        t_start = time.perf_counter()
        zero_shot_res = evaluator.evaluate(model, val_dataset)
        gpu_time_s = time.perf_counter() - t_start
        print(evaluator.format_table(zero_shot_res, title=f"Kết quả PromptIR Zero-Shot (Seed {args.seed})"))

        out_summary = {
            'branch': 'D',
            'model': 'PromptIR_ZeroShot',
            'seed': args.seed,
            'gpu_hours': round(gpu_time_s / 3600.0, 4),
            'metrics': zero_shot_res
        }
        with open(run_dir / "summary.json", "w", encoding='utf-8') as f:
            json.dump(out_summary, f, indent=2)
        print(f"Đã lưu kết quả Nhánh D vào: {run_dir / 'summary.json'}")
        return

    # 6. Chuẩn bị huấn luyện cho Nhánh A, B, C
    train_dataset = AiOIRDatasetV2(train_path, patch_size=128, length=args.iters * args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_hybrid_psnr = -1.0
    best_iter = -1
    best_metrics = {}
    history = []

    print(f"\nTiến hành huấn luyện {args.iters} iterations...")
    iter_count = 0
    t_train_start = time.perf_counter()
    train_iter = iter(train_loader)

    while iter_count < args.iters:
        iter_count += 1
        t_step = time.perf_counter()
        try:
            deg_img, clean_img, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            deg_img, clean_img, _ = next(train_iter)

        deg_img = deg_img.to(device)
        clean_img = clean_img.to(device)

        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred = model(deg_img)
            loss_recon = F.l1_loss(pred, clean_img)

            # Tính loss trực giao nếu có
            if hasattr(model, 'get_orthogonal_loss') and args.branch in ['B', 'C']:
                loss_orth = model.get_orthogonal_loss()
                total_loss = loss_recon + args.lambda_orth * loss_orth
            else:
                loss_orth = torch.tensor(0.0, device=device)
                total_loss = loss_recon

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        elapsed_step = time.perf_counter() - t_step

        if iter_count % 50 == 0 or iter_count == 1:
            cos_str = f" | Cos(L1/L2/L3): {get_prompt_cosine_stats(model)}" if args.branch in ['B', 'C'] else ""
            print(f"Iter [{iter_count:4d}/{args.iters}] | Recon Loss: {loss_recon.item():.4f} | Orth Loss: {loss_orth.item():.4f}{cos_str} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed_step:.2f}s")

        # 7. Đánh giá Validation định kỳ
        if iter_count % args.val_freq == 0 or iter_count == args.iters:
            print(f"\n--- [VAL ITER {iter_count}] Đang đánh giá trên Val v2 ---")
            val_res = evaluator.evaluate(model, val_dataset)
            print(f"PSNR Mean: {val_res['psnr_mean']:.2f} dB | Hybrid: {val_res['psnr_hybrid']:.2f} dB | Noise: {val_res['psnr_noise']:.2f} | Rain: {val_res['psnr_rain']:.2f} | Haze: {val_res['psnr_haze']:.2f}")

            val_res['iter'] = iter_count
            history.append(val_res)

            # Chọn model tốt nhất dựa trên tác vụ mục tiêu HYBRID trên Val set
            if val_res['psnr_hybrid'] > best_hybrid_psnr:
                best_hybrid_psnr = val_res['psnr_hybrid']
                best_iter = iter_count
                best_metrics = val_res
                torch.save({
                    'iter': iter_count,
                    'branch': args.branch,
                    'seed': args.seed,
                    'model_state': model.state_dict(),
                    'metrics': val_res
                }, run_dir / "best_model.pth")
                print(f">> [KỶ LỤC VAL MỚI] Đã lưu model tốt nhất vào: {run_dir / 'best_model.pth'}")

    total_gpu_time_s = time.perf_counter() - t_train_start
    gpu_hours = round(total_gpu_time_s / 3600.0, 4)

    # 8. Lưu Checkpoint cuối cùng (Final Checkpoint)
    final_val_res = evaluator.evaluate(model, val_dataset)
    torch.save({
        'iter': args.iters,
        'branch': args.branch,
        'seed': args.seed,
        'model_state': model.state_dict(),
        'metrics': final_val_res
    }, run_dir / "final_model.pth")

    print("\n" + "=" * 60)
    print(f"HOÀN TẤT THỰC NGHIỆM NHÁNH {args.branch} (SEED {args.seed})")
    print(f"-> Tổng thời gian chạy: {total_gpu_time_s:.1f}s ({gpu_hours} GPU hours)")
    print(f"-> Best Iteration trên Val: {best_iter} (Hybrid PSNR: {best_hybrid_psnr:.2f} dB)")
    print(evaluator.format_table(best_metrics, title=f"KẾT QUẢ CHECKPOINT TỐT NHẤT (BEST VAL) - NHÁNH {args.branch} SEED {args.seed}"))
    print(evaluator.format_table(final_val_res, title=f"KẾT QUẢ CHECKPOINT CUỐI CÙNG (FINAL) - NHÁNH {args.branch} SEED {args.seed}"))

    summary_data = {
        'branch': args.branch,
        'seed': args.seed,
        'total_iters': args.iters,
        'gpu_hours': gpu_hours,
        'best_val_iter': best_iter,
        'best_val_metrics': best_metrics,
        'final_val_metrics': final_val_res,
        'history': history
    }
    with open(run_dir / "summary.json", "w", encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    # Cập nhật kết quả vào CSV tổng
    csv_file = REPO_ROOT / "results" / "pilot_v2_results.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    header = "branch,seed,checkpoint_type,iter,gpu_hours,psnr_noise,psnr_rain,psnr_haze,psnr_hybrid,psnr_mean,ssim_noise,ssim_rain,ssim_haze,ssim_hybrid,ssim_mean\n"
    if not csv_file.exists():
        with open(csv_file, "w", encoding='utf-8') as f:
            f.write(header)

    with open(csv_file, "a", encoding='utf-8') as f:
        # Ghi Best Val
        f.write(f"{args.branch},{args.seed},best_val,{best_iter},{gpu_hours},"
                f"{best_metrics['psnr_noise']},{best_metrics['psnr_rain']},{best_metrics['psnr_haze']},{best_metrics['psnr_hybrid']},{best_metrics['psnr_mean']},"
                f"{best_metrics['ssim_noise']},{best_metrics['ssim_rain']},{best_metrics['ssim_haze']},{best_metrics['ssim_hybrid']},{best_metrics['ssim_mean']}\n")
        # Ghi Final
        f.write(f"{args.branch},{args.seed},final,{args.iters},{gpu_hours},"
                f"{final_val_res['psnr_noise']},{final_val_res['psnr_rain']},{final_val_res['psnr_haze']},{final_val_res['psnr_hybrid']},{final_val_res['psnr_mean']},"
                f"{final_val_res['ssim_noise']},{final_val_res['ssim_rain']},{final_val_res['ssim_haze']},{final_val_res['ssim_hybrid']},{final_val_res['ssim_mean']}\n")

    print(f"Đã cập nhật bảng tổng kết tại: {csv_file}")
    print("=" * 60)


if __name__ == '__main__':
    run_experiment()
