"""Huấn luyện Thử nghiệm Đột phá: PHASE-HOPFIELD-IR (Pilot v3).
Mã thực nghiệm: PHASE-HOPFIELD-PILOT-V3

Các nhánh đối chứng khoa học chuẩn mực:
- Nhánh A: PromptIR Baseline (NeurIPS 2023)
- Nhánh B (Đề xuất): PhaseHopfieldIR (FrFT + Giao hoán tử Lie + Hopfield Attractor Nobel 2024)
- Nhánh C (Đối chứng tham số): ParamMatchedPhaseHopfieldIR (Khóa 100% cùng số lượng tham số, miền không gian, không Lie, không Hopfield)

Cải tiến khắc phục triệt để lỗi ngủ đông (Lazy Training):
1. Energy-Scaled Coupling: Tỷ lệ ghép khởi tạo tanh(gate) = 0.15 kích hoạt gradient mạnh mẽ từ step 1.
2. Dual Learning Rate Optimizer: lr_novel = 1e-3 (cao gấp 10 lần) so với lr_base = 1e-4 để khối mới thích nghi và hội tụ nhanh.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import time
import json
import math
import random
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Thiết lập đường dẫn import
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.leakage_checker import verify_zero_leakage
from src.evaluation.unified_evaluator import UnifiedEvaluator, AiOIRTestHybridOODDataset, forward_padded
from src.models.phase_hopfield_promptir import PhaseHopfieldIR, ParamMatchedPhaseHopfieldIR


def set_seed(seed: int = 42):
    """Khóa toàn bộ hạt giống ngẫu nhiên để bảo đảm khả năng tái lập 100%."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AiOIRDatasetV2(Dataset):
    """Dataset sinh suy thoái trực tiếp xác định (deterministic) từ ảnh sạch của train_v2."""
    def __init__(self, clean_dir: Path, patch_size: int = 128, length: int = 1500 * 4, seed: int = 42):
        self.clean_dir = Path(clean_dir)
        self.patch_size = patch_size
        self.length = length
        self.seed = seed
        self.image_files = sorted(list(self.clean_dir.glob('*.png')) + list(self.clean_dir.glob('*.jpg')))
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh sạch tại: {clean_dir}")

        self.images_cache = []
        for p in self.image_files:
            with Image.open(p) as img:
                arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
                self.images_cache.append(arr)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx * 7919)
        img_idx = rng.randint(0, len(self.images_cache))
        clean_img = self.images_cache[img_idx]

        h, w, _ = clean_img.shape
        ps = self.patch_size
        if h < ps or w < ps:
            raise ValueError(f"Kích thước ảnh ({h}x{w}) nhỏ hơn patch size {ps}!")

        top = rng.randint(0, h - ps + 1)
        left = rng.randint(0, w - ps + 1)
        patch = clean_img[top:top+ps, left:left+ps, :].copy()

        # Augmentation xác định
        if rng.rand() > 0.5:
            patch = np.fliplr(patch)
        if rng.rand() > 0.5:
            patch = np.flipud(patch)
        rot_k = rng.randint(0, 4)
        if rot_k > 0:
            patch = np.rot90(patch, rot_k)

        task_id = rng.randint(0, 4)
        clean_patch = patch.copy()
        deg_patch = patch.copy()

        if task_id == 0:
            # Nhiễu Gaussian
            sigma = rng.uniform(15.0, 50.0) / 255.0
            noise = rng.normal(0, sigma, deg_patch.shape).astype(np.float32)
            deg_patch = np.clip(deg_patch + noise, 0.0, 1.0)
        elif task_id == 1:
            # Mưa tổng hợp
            num_streaks = rng.randint(40, 120)
            rain_layer = np.zeros_like(deg_patch)
            for _ in range(num_streaks):
                rx = rng.randint(0, ps - 1)
                ry = rng.randint(0, ps - 1)
                length = rng.randint(10, 25)
                angle = rng.uniform(-0.3, 0.3)
                alpha = rng.uniform(0.3, 0.7)
                for l in range(length):
                    cur_y = int(ry + l)
                    cur_x = int(rx + l * angle)
                    if 0 <= cur_y < ps and 0 <= cur_x < ps:
                        rain_layer[cur_y, cur_x, :] = alpha
            deg_patch = np.clip(deg_patch + rain_layer, 0.0, 1.0)
        elif task_id == 2:
            # Sương mù (Koschmieder)
            t = rng.uniform(0.2, 0.8)
            A = rng.uniform(0.7, 1.0)
            deg_patch = np.clip(deg_patch * t + A * (1.0 - t), 0.0, 1.0)
        else:
            # Suy thoái hỗn hợp (Hybrid-OOD)
            t = rng.uniform(0.3, 0.7)
            A = rng.uniform(0.7, 1.0)
            hazy = deg_patch * t + A * (1.0 - t)
            sigma = rng.uniform(15.0, 45.0) / 255.0
            noise = rng.normal(0, sigma, deg_patch.shape).astype(np.float32)
            deg_patch = np.clip(hazy + noise, 0.0, 1.0)

        deg_t = torch.from_numpy(deg_patch).permute(2, 0, 1).float()
        clean_t = torch.from_numpy(clean_patch).permute(2, 0, 1).float()
        return deg_t, clean_t


class AiOIRValDatasetV2(Dataset):
    """Validation dataset với 4 tác vụ cố định (10 ảnh mỗi tác vụ)."""
    def __init__(self, clean_dir: Path, samples_per_task: int = 10, seed: int = 42, img_size: int = 256):
        self.samples = []
        files = sorted(list(clean_dir.glob('*.png')) + list(clean_dir.glob('*.jpg')))
        if len(files) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh tại {clean_dir}")

        for task_idx, task_name in enumerate(['denoise', 'derain', 'dehaze', 'hybrid']):
            rng = np.random.RandomState(seed + task_idx * 1337)
            for i in range(samples_per_task):
                f = files[(task_idx * samples_per_task + i) % len(files)]
                with Image.open(f) as img:
                    img_np = np.array(img.convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

                clean_img = img_np.copy()
                deg_img = img_np.copy()

                if task_name == 'denoise':
                    sigma = 35.0 / 255.0
                    noise = rng.normal(0, sigma, deg_img.shape).astype(np.float32)
                    deg_img = np.clip(deg_img + noise, 0.0, 1.0)
                elif task_name == 'derain':
                    rain_layer = np.zeros_like(deg_img)
                    for _ in range(80):
                        rx = rng.randint(0, img_size - 1)
                        ry = rng.randint(0, img_size - 1)
                        length = rng.randint(15, 30)
                        for l in range(length):
                            cy = int(ry + l)
                            cx = int(rx + l * 0.15)
                            if 0 <= cy < img_size and 0 <= cx < img_size:
                                rain_layer[cy, cx, :] = 0.5
                    deg_img = np.clip(deg_img + rain_layer, 0.0, 1.0)
                elif task_name == 'dehaze':
                    t = 0.5
                    A = 0.9
                    deg_img = np.clip(deg_img * t + A * (1.0 - t), 0.0, 1.0)
                elif task_name == 'hybrid':
                    t = 0.5
                    A = 0.9
                    hazy = deg_img * t + A * (1.0 - t)
                    sigma = 25.0 / 255.0
                    noise = rng.normal(0, sigma, hazy.shape).astype(np.float32)
                    deg_img = np.clip(hazy + noise, 0.0, 1.0)

                deg_t = torch.from_numpy(deg_img).permute(2, 0, 1).float()
                clean_t = torch.from_numpy(clean_img).permute(2, 0, 1).float()
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
            ckpt = torch.load(init_weights, map_location='cpu', weights_only=False)
            state = ckpt.get('state_dict', ckpt.get('params', ckpt))
            clean_state = {k.replace('net.', '').replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(clean_state, strict=False)

    elif branch == 'B':
        model = PhaseHopfieldIR(decoder=True)
        if init_weights and Path(init_weights).exists():
            model.load_pretrained_promptir(init_weights)

    elif branch == 'C':
        model = ParamMatchedPhaseHopfieldIR(decoder=True)
        if init_weights and Path(init_weights).exists():
            model.load_pretrained_promptir(init_weights)

    else:
        raise ValueError(f"Không nhận diện được nhánh '{branch}'. Hợp lệ: A, B, C, D.")

    return model.to(device)


def run_experiment():
    parser = argparse.ArgumentParser(description="Huấn luyện PhaseHopfieldIR Pilot v3 đối chứng sòng phẳng")
    parser.add_argument('--branch', type=str, required=True, choices=['A', 'B', 'C', 'D'], help="Nhánh thực nghiệm (A, B, C, D)")
    parser.add_argument('--seed', type=int, default=42, help="Hạt giống ngẫu nhiên (seed: 42, 123, 2024)")
    parser.add_argument('--iters', '--max_iter', dest='iters', type=int, default=None, help="Số iterations huấn luyện")
    parser.add_argument('--smoke_test', action='store_true', help="Chạy thử nghiệm nhanh")
    parser.add_argument('--batch_size', type=int, default=4, help="Kích thước batch")
    parser.add_argument('--lr', type=float, default=1e-4, help="Tốc độ học cho pretrained backbone")
    parser.add_argument('--lr_novel', type=float, default=1e-3, help="Tốc độ học cao hơn cho khối mới (Phase-Hopfield)")
    parser.add_argument('--lambda_hop', type=float, default=0.001, help="Hệ số điều chuẩn thế năng Lyapunov Hopfield")
    parser.add_argument('--val_freq', type=int, default=250, help="Tần suất đánh giá validation")
    parser.add_argument('--init_weights', type=str, default="baselines/weights/promptir_all.ckpt", help="Đường dẫn pretrained")
    parser.add_argument('--train_dir', type=str, default="data/train_v2/clean", help="Thư mục ảnh train sạch")
    parser.add_argument('--val_dir', type=str, default="data/val_v2/clean", help="Thư mục ảnh val sạch")
    parser.add_argument('--test_dir', type=str, default="data/test", help="Thư mục ảnh test")
    parser.add_argument('--patch_size', type=int, default=128, help="Kích thước crop patch huấn luyện (mặc định 128)")
    parser.add_argument('--output_dir', type=str, default="runs/phase_hopfield_v3", help="Thư mục lưu log và checkpoints")
    parser.add_argument('--resume', type=str, default=None, help="Đường dẫn checkpoint để tiếp tục huấn luyện")
    parser.add_argument('--save_every', type=int, default=250, help="Tần suất lưu checkpoint")
    args = parser.parse_args()

    if args.iters is None:
        args.iters = 2 if args.smoke_test else 1500

    if args.smoke_test:
        args.output_dir = "runs/smoke_phase_hopfield"
        args.val_freq = args.iters
        print(f"\n[CHẾ ĐỘ SMOKE TEST KÍCH HOẠT] iters={args.iters}, ghi vào {args.output_dir}, KHÔNG bao giờ ghi vào results/.")
    elif args.iters <= 50:
        args.val_freq = args.iters

    # 1. Khóa seed
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = (REPO_ROOT / args.train_dir).resolve()
    val_path = (REPO_ROOT / args.val_dir).resolve()
    test_path = (REPO_ROOT / args.test_dir).resolve()

    # 2. KHẲNG ĐỊNH CHỐNG RÒ RỈ DỮ LIỆU FAIL-CLOSED
    print("=" * 75)
    print(f"BẮT ĐẦU THỰC NGHIỆM PHASE-HOPFIELD-IR — NHÁNH {args.branch} (SEED {args.seed}) TRÊN {device}")
    print("=" * 75)
    verify_zero_leakage([train_path], [val_path], [test_path], near_dup_threshold=4, verbose=True)

    # 3. Tạo thư mục lưu kết quả & Ghi nhận Git Commit HEAD
    run_dir = (REPO_ROOT / args.output_dir / f"branch_{args.branch}_seed_{args.seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        commit_hash = "UNKNOWN_OR_NO_GIT"
    (run_dir / "git_commit.txt").write_text(commit_hash, encoding='utf-8')
    print(f">> [AUDIT KHOA HỌC] Git Commit HEAD: {commit_hash}", flush=True)

    # 4. Khởi tạo mô hình
    weights_path = (REPO_ROOT / args.init_weights).resolve()
    model = build_model(args.branch, str(weights_path), device)

    # Đếm số lượng tham số
    n_params = sum(p.numel() for p in model.parameters())
    print(f">> Tổng số tham số mô hình Nhánh {args.branch}: {n_params:,}")

    # 5. Chuẩn bị DataLoader xác định & Checksum
    train_dataset = AiOIRDatasetV2(train_path, patch_size=args.patch_size, length=max(50, args.iters * args.batch_size), seed=args.seed)

    import inspect
    data_code_str = inspect.getsource(AiOIRDatasetV2)
    data_code_hash = hashlib.sha256(data_code_str.encode('utf-8')).hexdigest()[:16]
    sorted_image_sigs = [f"{p.name}:{p.stat().st_size}" for p in train_dataset.image_files]
    clean_images_hash = hashlib.sha256("||".join(sorted_image_sigs).encode('utf-8')).hexdigest()[:16]

    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed)
    num_workers = 2 if (device.type == 'cuda' and os.name != 'nt') else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=train_gen,
        worker_init_fn=seed_worker,
        drop_last=True
    )

    batch_hasher = hashlib.sha256()
    preview_count = min(50, len(train_loader))
    loader_iter = iter(train_loader)
    for _ in range(preview_count):
        d_b, c_b = next(loader_iter)
        batch_hasher.update(d_b.numpy().tobytes()[:256])
        batch_hasher.update(c_b.numpy().tobytes()[:256])
    batch_checksum = batch_hasher.hexdigest()[:16]

    content_lock_metadata = {
        'seed': args.seed,
        'iters': args.iters,
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
        'data_code_hash': data_code_hash,
        'clean_images_hash': clean_images_hash,
        'num_clean_images': len(train_dataset.image_files),
        'batch_checksum_50': batch_checksum,
        'git_commit': commit_hash[:7]
    }

    checksum_filename = f"batch_checksum_seed_{args.seed}_patch_{args.patch_size}_batch_{args.batch_size}_iter_{args.iters}.json"
    checksum_file = (REPO_ROOT / args.output_dir / checksum_filename).resolve()
    checksum_file.parent.mkdir(parents=True, exist_ok=True)

    if checksum_file.exists():
        with open(checksum_file, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)
        for key in ['seed', 'iters', 'batch_size', 'patch_size', 'data_code_hash', 'clean_images_hash', 'num_clean_images', 'batch_checksum_50']:
            if content_lock_metadata.get(key) != ref_data.get(key):
                raise AssertionError(f"Mâu thuẫn khóa nội dung batch giữa các nhánh ở trường '{key}'!")
        print(f"[XÁC MINH ĐỒNG BỘ BATCH THÀNH CÔNG] Checksum khớp 100% với {checksum_file.name}!", flush=True)
    else:
        if args.branch in ['B', 'C']:
            raise FileNotFoundError(f"Nhánh {args.branch} bắt buộc phải có file checksum từ Nhánh A chạy trước: {checksum_file.name}")
        with open(checksum_file, 'w', encoding='utf-8') as f:
            json.dump(content_lock_metadata, f, indent=2)
        print(f"[LƯU CHECKSUM THAM CHIẾU] Đã lưu checksum chuẩn Nhánh {args.branch} -> {checksum_file.name}", flush=True)

    train_gen.manual_seed(args.seed)

    # 6. Khởi tạo Evaluator
    evaluator = UnifiedEvaluator(device=device, use_uint8=True)
    val_samples = 1 if args.smoke_test else 10
    ood_samples = 1 if args.smoke_test else 30
    eval_val_size = 128 if args.smoke_test else 256
    eval_ood_size = 128 if args.smoke_test else 512
    val_dataset = AiOIRValDatasetV2(val_path, samples_per_task=val_samples, seed=args.seed, img_size=eval_val_size)

    test_clean_dirs = [
        test_path / 'denoise' / 'bsd68',
        test_path / 'denoise' / 'kodak24',
        test_path / 'dehaze' / 'target'
    ]
    test_hybrid_ood_dataset = AiOIRTestHybridOODDataset(test_clean_dirs, max_images=ood_samples, seed=args.seed, max_size=eval_ood_size)

    # ĐO LƯỜNG TẠI ITERATION 0 (TRƯỚC FINE-TUNE)
    print("\n--- [ITERATION 0: ĐO ĐẠC MỨC NỀN TRƯỚC HUẤN LUYỆN] ---", flush=True)
    val_res_0 = evaluator.evaluate(model, val_dataset)
    ood_res_0 = evaluator.evaluate(model, test_hybrid_ood_dataset)
    print(f"Iter 0 (Pre-train) | Val Hybrid PSNR: {val_res_0.get('psnr_hybrid', 'N/A')} dB | Held-out Hybrid-OOD PSNR: {ood_res_0.get('psnr_mean', 'N/A')} dB", flush=True)

    if args.branch == 'D':
        out_summary = {
            'branch': 'D',
            'model': 'PromptIR_ZeroShot',
            'seed': args.seed,
            'val_metrics': val_res_0,
            'test_hybrid_ood_metrics': ood_res_0
        }
        with open(run_dir / "summary.json", "w", encoding='utf-8') as f:
            json.dump(out_summary, f, indent=2)
        print(f"Đã lưu kết quả Nhánh D vào: {run_dir / 'summary.json'}")
        return

    # TÁCH PARAMETER GROUPS CHO DUAL LEARNING RATE
    base_params = []
    novel_params = []
    novel_keywords = ['fc_order', 'head_A', 'head_B', 'conv_down', 'conv_up', 'conv_in', 'conv_out', 'M', 'step_size', 'gate']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in novel_keywords):
            novel_params.append(param)
        else:
            base_params.append(param)

    if args.branch in ['B', 'C']:
        param_groups = [
            {'params': base_params, 'lr': args.lr, 'weight_decay': 1e-4},
            {'params': novel_params, 'lr': args.lr_novel, 'weight_decay': 1e-4}
        ]
        print(f">> Khởi tạo DUAL LEARNING RATE: lr_base = {args.lr:.2e}, lr_novel = {args.lr_novel:.2e}")
        print(f">> Số tham số base: {sum(p.numel() for p in base_params):,}, Số tham số khối mới: {sum(p.numel() for p in novel_params):,}")
    else:
        param_groups = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': 1e-4}]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_hybrid_psnr = -1.0
    best_iter = -1
    best_metrics = {}
    history = []
    start_iter = 0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n[TIẾP TỤC HUẤN LUYỆN] Đang nạp checkpoint từ: {resume_path}")
            ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            if 'scaler_state' in ckpt and device.type == 'cuda':
                scaler.load_state_dict(ckpt['scaler_state'])
            start_iter = ckpt.get('iteration', 0)
            best_hybrid_psnr = ckpt.get('best_hybrid_psnr', -1.0)
            history = ckpt.get('history', [])
            print(f">> Đã khôi phục thành công từ iteration {start_iter} (Best Hybrid PSNR: {best_hybrid_psnr:.2f} dB)")

    data_iterator = iter(train_loader)
    print(f"\n[BẮT ĐẦU CHU KỲ HUẤN LUYỆN] Từ iter {start_iter} đến {args.iters}...")
    start_train_time = time.time()

    for cur_iter in range(start_iter + 1, args.iters + 1):
        model.train()
        try:
            deg_batch, clean_batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            deg_batch, clean_batch = next(data_iterator)

        deg_batch = deg_batch.to(device, non_blocking=True)
        clean_batch = clean_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            restored = model(deg_batch)
            loss_char = torch.sqrt((restored - clean_batch) ** 2 + 1e-6).mean()
            if args.branch == 'B' and hasattr(model, 'get_hopfield_energy_loss') and args.lambda_hop > 0:
                loss_hop = model.get_hopfield_energy_loss()
                loss_total = loss_char + args.lambda_hop * loss_hop
            else:
                loss_total = loss_char

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if cur_iter % 50 == 0 or cur_iter == args.iters:
            ratio_val = model.get_new_block_ratio() if hasattr(model, 'get_new_block_ratio') else 0.0
            energy_val = model.get_hopfield_energy() if hasattr(model, 'get_hopfield_energy') else 0.0
            lie_val = model.get_lie_norm() if hasattr(model, 'get_lie_norm') else 0.0
            print(f"Iter [{cur_iter:4d}/{args.iters}] | Loss: {loss_total.item():.4f} | Ratio(New/Base): {ratio_val:.4f} | Hopfield Energy: {energy_val:.2f} | Lie Norm: {lie_val:.4f}", flush=True)

        # Lưu checkpoint định kỳ phục vụ resume
        if cur_iter % args.save_every == 0 or cur_iter == args.iters:
            ckpt_data = {
                'iteration': cur_iter,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict() if device.type == 'cuda' else None,
                'best_hybrid_psnr': best_hybrid_psnr,
                'history': history,
                'args': vars(args)
            }
            torch.save(ckpt_data, run_dir / f"checkpoint_iter_{cur_iter}.pth")
            torch.save(ckpt_data, run_dir / "latest_checkpoint.pth")

        # Validation định kỳ
        if cur_iter % args.val_freq == 0 or cur_iter == args.iters:
            print(f"\n--- [ĐÁNH GIÁ TẠI ITER {cur_iter}] ---", flush=True)
            val_metrics = evaluator.evaluate(model, val_dataset)
            psnr_hybrid = val_metrics.get('psnr_hybrid', 0.0)
            ratio_val = model.get_new_block_ratio() if hasattr(model, 'get_new_block_ratio') else 0.0

            log_entry = {
                'iter': cur_iter,
                'loss': float(loss_total.item()),
                'val_metrics': val_metrics,
                'new_block_ratio': ratio_val
            }
            history.append(log_entry)

            print(f">> Val Denoise: {val_metrics.get('psnr_denoise', 0.0):.2f} dB | Dehaze: {val_metrics.get('psnr_dehaze', 0.0):.2f} dB | Derain: {val_metrics.get('psnr_derain', 0.0):.2f} dB | Hybrid: {psnr_hybrid:.2f} dB", flush=True)
            print(f">> Tỉ lệ năng lượng Ratio(New/Base): {ratio_val:.4f}", flush=True)

            if psnr_hybrid > best_hybrid_psnr:
                best_hybrid_psnr = psnr_hybrid
                best_iter = cur_iter
                best_metrics = val_metrics
                torch.save(model.state_dict(), run_dir / "best_model.pth")
                print(f"[KỶ LỤC MỚI] Đã lưu best_model.pth tại iter {cur_iter} (Hybrid PSNR: {psnr_hybrid:.2f} dB)", flush=True)

    total_gpu_hours = (time.time() - start_train_time) / 3600.0
    print(f"\n[HOÀN TẤT HUẤN LUYỆN] Tổng thời gian thực thi: {total_gpu_hours * 60.0:.2f} phút ({total_gpu_hours:.4f} GPU-hours)")

    # Đánh giá Primary Endpoint trên Test Held-out Hybrid-OOD
    print("\n======================================================================")
    print("[ĐÁNH GIÁ PRIMARY ENDPOINT: HELD-OUT TEST HYBRID-OOD]")
    print("======================================================================")
    best_weights_path = run_dir / "best_model.pth"
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))

    test_hybrid_ood_metrics = evaluator.evaluate(model, test_hybrid_ood_dataset)
    print(f">> Kết quả Held-out Test Hybrid-OOD PSNR: {test_hybrid_ood_metrics.get('psnr_mean', 0.0):.2f} dB (SSIM: {test_hybrid_ood_metrics.get('ssim_mean', 0.0):.4f})")

    final_summary = {
        'branch': args.branch,
        'seed': args.seed,
        'iters': args.iters,
        'best_val_iter': best_iter,
        'best_val_hybrid_psnr': best_hybrid_psnr,
        'val_metrics_at_best': best_metrics,
        'val_metrics_iter0': val_res_0,
        'test_hybrid_ood_metrics_iter0': ood_res_0,
        'test_hybrid_ood_metrics': test_hybrid_ood_metrics,
        'total_gpu_hours': round(total_gpu_hours, 4),
        'final_new_block_ratio': model.get_new_block_ratio() if hasattr(model, 'get_new_block_ratio') else 0.0,
        'final_prototype_diversity': model.get_prototype_diversity() if hasattr(model, 'get_prototype_diversity') else {'max_cos_sim': 0.0, 'mean_cos_sim': 0.0},
        'history': history
    }

    with open(run_dir / "summary.json", "w", encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2)
    print(f">> Đã lưu tóm tắt toàn bộ thực nghiệm vào: {run_dir / 'summary.json'}")

    # Cập nhật kết quả vào CSV tổng: Nếu là smoke_test thì lưu vào runs/smoke/, TUYỆT ĐỐI KHÔNG ghi vào results/
    if args.smoke_test:
        csv_file = REPO_ROOT / "runs" / "smoke_phase_hopfield" / "smoke_phase_hopfield_results.csv"
    else:
        csv_file = REPO_ROOT / "results" / "phase_hopfield_results.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    header = "branch,seed,checkpoint_type,iter,gpu_hours,val_psnr_noise,val_psnr_rain,val_psnr_haze,val_psnr_hybrid,val_psnr_mean,primary_endpoint_ood_psnr,new_block_ratio\n"
    if not csv_file.exists():
        with open(csv_file, "w", encoding='utf-8') as f:
            f.write(header)

    with open(csv_file, "a", encoding='utf-8') as f:
        f.write(f"{args.branch},{args.seed},best_val,{best_iter},{total_gpu_hours:.4f},"
                f"{best_metrics.get('psnr_denoise', 0.0)},{best_metrics.get('psnr_derain', 0.0)},{best_metrics.get('psnr_dehaze', 0.0)},{best_metrics.get('psnr_hybrid', 0.0)},{best_metrics.get('psnr_mean', 0.0)},"
                f"{test_hybrid_ood_metrics.get('psnr_mean', 0.0)},{final_summary['final_new_block_ratio']:.4f}\n")
        f.write(f"{args.branch},{args.seed},final,{args.iters},{total_gpu_hours:.4f},"
                f"{val_metrics.get('psnr_denoise', 0.0)},{val_metrics.get('psnr_derain', 0.0)},{val_metrics.get('psnr_dehaze', 0.0)},{val_metrics.get('psnr_hybrid', 0.0)},{val_metrics.get('psnr_mean', 0.0)},"
                f"{test_hybrid_ood_metrics.get('psnr_mean', 0.0)},{final_summary['final_new_block_ratio']:.4f}\n")

    print(f"Đã cập nhật bảng tổng kết tại: {csv_file}")
    print("=" * 75)


if __name__ == '__main__':
    run_experiment()
