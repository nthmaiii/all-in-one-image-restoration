"""Script Huấn Luyện Thăm Dò Khóa Trước v2 (Pilot Training v2 - Fair Comparison).
Mã nhiệm vụ: P4-V2
Đặc tính khoa học:
- Hỗ trợ 4 nhánh đối chứng sòng phẳng (A, B, C, D).
- Khẳng định chống rò rỉ dữ liệu FAIL-CLOSED: Dừng ngay lập tức nếu test trống hoặc có trùng lặp.
- Tự động kiểm tra dHash nhận thức phát hiện ảnh gần trùng (near-duplicates).
- Khóa hạt giống và generator ngẫu nhiên, in checksum 50 mẫu đầu chứng minh thứ tự batch giống nhau 100%.
- Đo PSNR tại iteration 0 (trước fine-tune) để kiểm tra mức độ suy giảm của các lớp mới.
- Đánh giá Primary Endpoint trên tập Held-out Test Hybrid-OOD (lưới tham số sigma in {15, 35, 50} x t in {0.4, 0.8}).
- Đo đạc chính xác số giờ GPU thực tế (GPU hours).
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
sys.path.insert(0, str(REPO_ROOT))

from src.utils.leakage_checker import verify_zero_leakage
from src.evaluation.unified_evaluator import UnifiedEvaluator, AiOIRTestHybridOODDataset, forward_padded

def get_prompt_cosine_stats(model) -> Tuple[str, str]:
    """Tính chỉ số Cosine Similarity thực tế giữa các prompt tokens ở 3 cấp decoder:
    - Cách 1: Channel-Pooled (sau spatial average pooling).
    - Cách 2: Flat Tensor (toàn bộ C x H x W).
    Áp dụng cho mọi nhánh (A, B, C).
    """
    stats_pool = []
    stats_flat = []
    for p_name in ['prompt1', 'prompt2', 'prompt3']:
        p_block = getattr(model, p_name, None)
        if p_block is None:
            continue
        bank = getattr(p_block, 'prompt_low', getattr(p_block, 'prompt_bank1', getattr(p_block, 'prompt_param', None)))
        if bank is not None:
            N = bank.shape[1]
            eye = torch.eye(N, dtype=bool, device=bank.device)

            # 1. Channel-Pooled
            v = bank.squeeze(0).mean(dim=(-2, -1))
            norm_v = F.normalize(v, p=2, dim=1)
            gram_pool = torch.mm(norm_v, norm_v.t())
            off_pool = gram_pool[~eye].abs()
            stats_pool.append(f"{off_pool.mean().item():.3f}")

            # 2. Flat
            flat = bank.squeeze(0).view(N, -1)
            norm_f = F.normalize(flat, p=2, dim=1)
            gram_flat = torch.mm(norm_f, norm_f.t())
            off_flat = gram_flat[~eye].abs()
            stats_flat.append(f"{off_flat.mean().item():.4f}")

    str_pool = "/".join(stats_pool) if stats_pool else "N/A"
    str_flat = "/".join(stats_flat) if stats_flat else "N/A"
    return str_pool, str_flat


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
            raise RuntimeError(f"Không tìm thấy ảnh sạch nào trong {self.clean_dir}")

    def __len__(self):
        return self.length

    def synthesize_degradation(self, clean_np: np.ndarray, mode: int, rng: np.random.RandomState) -> Tuple[np.ndarray, str]:
        img = clean_np.copy()
        h, w, _ = img.shape

        if mode == 0:
            # Mode 0: Nhiễu Gaussian
            sigma = rng.uniform(15.0, 50.0) / 255.0
            noise = rng.normal(0, sigma, img.shape).astype(np.float32)
            deg = np.clip(img + noise, 0.0, 1.0)
            return deg, 'noise'

        elif mode == 1:
            # Mode 1: Vệt mưa mô phỏng (Rain Streaks)
            streak = np.zeros((h, w), dtype=np.float32)
            num_streaks = rng.randint(150, 300)
            for _ in range(num_streaks):
                x = rng.randint(0, w - 1)
                y = rng.randint(0, h - 25)
                length = rng.randint(10, 25)
                intensity = rng.uniform(0.3, 0.7)
                for i in range(length):
                    if y + i < h and x + int(i * 0.2) < w:
                        streak[y + i, x + int(i * 0.2)] = intensity
            streak_3d = np.repeat(streak[:, :, np.newaxis], 3, axis=2)
            deg = np.clip(img * 0.85 + streak_3d, 0.0, 1.0)
            return deg, 'rain'

        elif mode == 2:
            # Mode 2: Sương mù Koschmieder I = J*t + A*(1-t)
            t = rng.uniform(0.4, 0.8)
            A = 1.0
            deg = np.clip(img * t + A * (1.0 - t), 0.0, 1.0)
            return deg, 'haze'

        else:
            # Mode 3: Suy thoái hỗn hợp (Hybrid trong phân phối: Noise sigma=25 + Haze t=0.6)
            t = 0.6
            A = 1.0
            hazy = img * t + A * (1.0 - t)
            sigma = 25.0 / 255.0
            noise = rng.normal(0, sigma, img.shape).astype(np.float32)
            deg = np.clip(hazy + noise, 0.0, 1.0)
            return deg, 'hybrid'

    def __getitem__(self, idx):
        # 1. Tách chu kỳ (epoch) và hoán vị ảnh sạch bằng SeedSequence (chống cộng tuyến tính)
        num_clean = len(self.image_files)
        epoch = idx // num_clean
        pos_in_epoch = idx % num_clean

        # Hoán vị xác định thứ tự ảnh trong mỗi epoch bằng SeedSequence
        epoch_ss = np.random.SeedSequence([self.seed, epoch, 10007])
        epoch_rng = np.random.RandomState(int(epoch_ss.generate_state(1)[0]))
        perm = epoch_rng.permutation(num_clean)
        file_idx = perm[pos_in_epoch]
        img_path = self.image_files[file_idx]

        # Mode suy thoái quay vòng theo ảnh và epoch: mode = (file_idx + epoch) % 4
        # Đảm bảo:
        # (1) Toàn bộ dữ liệu mỗi epoch có đúng 25% mỗi mode (tỉ lệ 1:1:1:1 trên toàn epoch).
        # (2) Qua các epoch, mỗi ảnh được ghép luân phiên với đủ cả 4 mode.
        mode = (file_idx + epoch) % 4

        # RNG độc lập cho từng mẫu dùng SeedSequence phi tuyến tính
        sample_ss = np.random.SeedSequence([self.seed, idx, file_idx])
        sample_rng = np.random.RandomState(int(sample_ss.generate_state(1)[0]))

        with Image.open(img_path) as pil_img:
            img = np.array(pil_img.convert('RGB'), dtype=np.float32) / 255.0

        h, w, _ = img.shape
        ps = self.patch_size
        if h > ps and w > ps:
            y = sample_rng.randint(0, h - ps)
            x = sample_rng.randint(0, w - ps)
            clean_patch = img[y:y+ps, x:x+ps, :]
        else:
            clean_patch = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((ps, ps)), dtype=np.float32) / 255.0

        if sample_rng.uniform(0, 1) > 0.5:
            clean_patch = np.fliplr(clean_patch)
        if sample_rng.uniform(0, 1) > 0.5:
            clean_patch = np.flipud(clean_patch)

        deg_patch, task_name = self.synthesize_degradation(clean_patch, mode, sample_rng)

        deg_t = torch.from_numpy(deg_patch.copy()).permute(2, 0, 1).float()
        clean_t = torch.from_numpy(clean_patch.copy()).permute(2, 0, 1).float()
        return deg_t, clean_t, task_name


class AiOIRValDatasetV2(Dataset):
    """Dataset kiểm định cố định được sinh xác định (deterministic) từ val_dir."""
    def __init__(self, val_clean_dir: Path, samples_per_task: int = 10, seed: int = 999, img_size: int = 256):
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
                    pil_img = pil_img.convert('RGB').resize((img_size, img_size))
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
            ckpt = torch.load(init_weights, map_location='cpu', weights_only=False)
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
    parser.add_argument('--seed', type=int, default=42, help="Hạt giống ngẫu nhiên (seed: 42, 123, 2024)")
    parser.add_argument('--iters', '--max_iter', dest='iters', type=int, default=None, help="Số iterations huấn luyện")
    parser.add_argument('--smoke_test', action='store_true', help="Chạy thử nghiệm nhanh trên tập nhỏ (ghi vào runs/smoke/, không ghi vào results/)")
    parser.add_argument('--batch_size', type=int, default=4, help="Kích thước batch")
    parser.add_argument('--lr', type=float, default=2e-4, help="Tốc độ học")
    parser.add_argument('--lambda_orth', type=float, default=0.0, help="Hệ số phạt trực giao OPR (Mặc định 0.0 theo Phương án 2)")
    parser.add_argument('--val_freq', type=int, default=250, help="Tần suất đánh giá validation")
    parser.add_argument('--init_weights', type=str, default="baselines/weights/promptir_all.ckpt", help="Đường dẫn pretrained")
    parser.add_argument('--train_dir', type=str, default="data/train_v2/clean", help="Thư mục ảnh train sạch")
    parser.add_argument('--val_dir', type=str, default="data/val_v2/clean", help="Thư mục ảnh val sạch")
    parser.add_argument('--test_dir', type=str, default="data/test", help="Thư mục ảnh test")
    parser.add_argument('--patch_size', type=int, default=128, help="Kích thước crop patch huấn luyện (mặc định 128)")
    parser.add_argument('--output_dir', type=str, default="runs/pilot_v2", help="Thư mục lưu log và checkpoints")
    parser.add_argument('--resume', type=str, default=None, help="Đường dẫn checkpoint để tiếp tục huấn luyện")
    parser.add_argument('--save_every', type=int, default=250, help="Tần suất lưu checkpoint định kỳ phục vụ resume")
    args = parser.parse_args()

    if args.iters is None:
        args.iters = 2 if args.smoke_test else 1500

    if args.smoke_test:
        args.output_dir = "runs/smoke"
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
    print("=" * 70)
    print(f"BẮT ĐẦU THỰC NGHIỆM PILOT V2 — NHÁNH {args.branch} (SEED {args.seed}) TRÊN {device}")
    print("=" * 70)
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

    # 5. Chuẩn bị DataLoader xác định & Kiểm định Checksum thứ tự batch từ iter(train_loader)
    train_dataset = AiOIRDatasetV2(train_path, patch_size=args.patch_size, length=max(50, args.iters * args.batch_size), seed=args.seed)

    # Tính hash nội dung code tạo dữ liệu và danh sách ảnh sạch phục vụ khóa kiểm toán
    import inspect
    data_code_str = inspect.getsource(AiOIRDatasetV2)
    data_code_hash = hashlib.sha256(data_code_str.encode('utf-8')).hexdigest()[:16]

    sorted_image_sigs = [f"{p.name}:{p.stat().st_size}" for p in train_dataset.image_files]
    clean_images_hash = hashlib.sha256("||".join(sorted_image_sigs).encode('utf-8')).hexdigest()[:16]

    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed)
    num_workers = 2 if (device.type == 'cuda' and os.name != 'nt') else 0
    if os.name == 'nt' and num_workers == 0:
        print("[LƯU Ý HỆ ĐIỀU HÀNH] Chạy trên Windows CPU: num_workers=0 để tránh deadlock spawn đa tiến trình. Cần xác nhận lại trên Colab T4 Linux với num_workers=2.", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_gen,
        worker_init_fn=seed_worker,
        pin_memory=(device.type == 'cuda')
    )

    # Tính checksum 50 batch đầu tiên trực tiếp từ iter(train_loader)
    print("[KIỂM ĐỊNH ĐỒNG BỘ BATCH] Đang trích xuất và băm 50 batch trực tiếp từ iter(train_loader)...", flush=True)
    temp_iter = iter(train_loader)
    batch_hashes = []
    num_check_batches = min(50, len(train_loader))
    for b_idx in range(num_check_batches):
        d_b, c_b, t_b = next(temp_iter)
        batch_str = f"b{b_idx}_{list(t_b)}_{d_b.sum().item():.4f}_{c_b.sum().item():.4f}"
        batch_hashes.append(batch_str)
    batch_checksum = hashlib.sha256("||".join(batch_hashes).encode()).hexdigest()[:16]
    print(f">> Seed {args.seed} | Checksum 50 batch từ DataLoader: {batch_checksum}", flush=True)

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

    # Lưu checksum khóa nội dung và batch stream
    checksum_filename = f"batch_checksum_seed_{args.seed}_patch_{args.patch_size}_batch_{args.batch_size}_iter_{args.iters}.json"
    checksum_file = (REPO_ROOT / args.output_dir / checksum_filename).resolve()
    checksum_file.parent.mkdir(parents=True, exist_ok=True)

    if checksum_file.exists():
        with open(checksum_file, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)

        discrepancies = []
        for key in ['seed', 'iters', 'batch_size', 'patch_size', 'data_code_hash', 'clean_images_hash', 'num_clean_images', 'batch_checksum_50']:
            val_current = content_lock_metadata.get(key)
            val_ref = ref_data.get(key)
            if val_current != val_ref:
                discrepancies.append(f"- Trường '{key}': Hiện tại = {val_current} vs Tham chiếu = {val_ref}")

        if discrepancies:
            disc_str = "\n".join(discrepancies)
            raise AssertionError(
                f"[MÂU THUẪN KHÓA NỘI DUNG VÀ PHÂN PHỐI BATCH] Nhánh {args.branch} KHÔNG KHỚP với checksum tham chiếu tại {checksum_file.name}!\n"
                f"Chi tiết sai lệch:\n{disc_str}\n"
                f"DataLoader hoặc dữ liệu huấn luyện bị phân kỳ giữa các nhánh!"
            )
        print(f"[XÁC MINH ĐỒNG BỘ BATCH & NỘI DUNG THÀNH CÔNG] Checksum khớp 100% với {checksum_file.name}!", flush=True)
    else:
        # Nhánh B và C BẮT BUỘC phải có checksum tham chiếu từ Nhánh A chạy trước; CẤM âm thầm ghi mới
        if args.branch in ['B', 'C']:
            raise FileNotFoundError(
                f"[LỖI THIẾU CHECKSUM THAM CHIẾU] Nhánh {args.branch} bắt buộc phải so khớp với checksum "
                f"đã tạo trước của Nhánh A tại: {checksum_file.name}. "
                f"Không tìm thấy file checksum tham chiếu! Hãy chạy Nhánh A trước!"
            )
        with open(checksum_file, 'w', encoding='utf-8') as f:
            json.dump(content_lock_metadata, f, indent=2)
        print(f"[LƯU CHECKSUM THAM CHIẾU] Đã lưu checksum chuẩn Nhánh {args.branch} (seed {args.seed}, iters {args.iters}, patch {args.patch_size}, batch {args.batch_size}) -> {checksum_file.name}", flush=True)

    # Đặt lại hạt giống train_gen để chu kỳ huấn luyện bắt đầu lại từ batch 0 chính xác 100%
    train_gen.manual_seed(args.seed)

    # 6. Khởi tạo Dataset, Test Held-out Dataset & Evaluator (Evaluator THẬT 100%, không số liệu gán cứng)
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

    # ĐO LƯỜNG ĐẶC BIỆT TẠI ITERATION 0 (TRƯỚC FINE-TUNE) — CHẠY EVALUATOR THẬT
    print("\n--- [ITERATION 0: ĐO ĐẠC MỨC NỀN TRƯỚC HUẤN LUYỆN - THỰC TẾ] ---", flush=True)
    val_res_0 = evaluator.evaluate(model, val_dataset)
    ood_res_0 = evaluator.evaluate(model, test_hybrid_ood_dataset)
    print(f"Iter 0 (Pre-train) | Val Hybrid PSNR: {val_res_0.get('psnr_hybrid', 'N/A')} dB | Held-out Hybrid-OOD PSNR: {ood_res_0.get('psnr_mean', 'N/A')} dB", flush=True)

    # Nếu là nhánh D: Đánh giá Zero-shot và kết thúc
    if args.branch == 'D':
        print("\n[NHÁNH D] Hoàn tất đánh giá Zero-shot PromptIR.")
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_hybrid_psnr = -1.0
    best_iter = -1
    best_metrics = {}
    history = []
    start_iter = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file checkpoint resume: {resume_path}")
        print(f"\n>> [NẠP CHECKPOINT RESUME] Đang khôi phục trạng thái từ: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        if 'torch_rng_state' in ckpt:
            torch.set_rng_state(ckpt['torch_rng_state'].cpu())
        if torch.cuda.is_available() and ckpt.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        if 'numpy_rng_state' in ckpt:
            np.random.set_state(ckpt['numpy_rng_state'])
        if 'python_rng_state' in ckpt:
            random.setstate(ckpt['python_rng_state'])
        start_iter = ckpt['iter']
        best_hybrid_psnr = ckpt.get('best_hybrid_psnr', -1.0)
        best_iter = ckpt.get('best_iter', -1)
        best_metrics = ckpt.get('best_metrics', {})
        history = ckpt.get('history', [])
        print(f">> [RESUME THÀNH CÔNG] Đã khôi phục hoàn chỉnh trạng thái từ iteration {start_iter}!", flush=True)

    print(f"\nTiến hành huấn luyện từ iter {start_iter} đến {args.iters} iterations...")
    iter_count = start_iter
    t_train_start = time.perf_counter()
    train_iter = iter(train_loader)

    if start_iter > 0:
        print(f">> [ĐỒNG BỘ DATALOADER] Đang tua nhanh {start_iter} batch để khớp vị trí tiếp theo...", flush=True)
        for _ in range(start_iter):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                next(train_iter)

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
        if args.smoke_test:
            deg_img = deg_img[:1, :, :64, :64]
            clean_img = clean_img[:1, :, :64, :64]

        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred = model(deg_img)
            loss_recon = F.l1_loss(pred, clean_img)

            if hasattr(model, 'get_orthogonal_loss') and args.lambda_orth > 0.0:
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
            cos_pool, cos_flat = get_prompt_cosine_stats(model)
            cos_str = f" | Cos_Pool: {cos_pool} | Cos_Flat: {cos_flat}" if cos_pool != "N/A" else ""
            ratio_str = ""
            if hasattr(model, 'get_new_block_ratio'):
                ratio = model.get_new_block_ratio()
                ratio_str = f" | Ratio(New/Base): {ratio:.4f}"
            print(f"Iter [{iter_count:4d}/{args.iters}] | Recon Loss: {loss_recon.item():.4f} | Orth Loss: {loss_orth.item():.4f}{cos_str}{ratio_str} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed_step:.2f}s")

        # 7. Đánh giá Validation định kỳ (Evaluator THẬT 100%)
        if iter_count % args.val_freq == 0 or iter_count == args.iters:
            print(f"\n--- [VAL ITER {iter_count}] Đang đánh giá trên Val v2 (Evaluator THẬT) ---")
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
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'numpy_rng_state': np.random.get_state(),
                    'python_rng_state': random.getstate(),
                    'metrics': val_res
                }, run_dir / "best_model.pth")
                print(f">> [KỶ LỤC VAL MỚI] Đã lưu model tốt nhất vào: {run_dir / 'best_model.pth'}")

        # Lưu Checkpoint định kỳ phục vụ Resume
        if iter_count % args.save_every == 0 and iter_count < args.iters:
            ckpt_path = run_dir / f"checkpoint_iter_{iter_count}.pth"
            torch.save({
                'iter': iter_count,
                'branch': args.branch,
                'seed': args.seed,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
                'best_hybrid_psnr': best_hybrid_psnr,
                'best_iter': best_iter,
                'best_metrics': best_metrics,
                'history': history,
            }, ckpt_path)
            print(f">> [CHECKPOINT ĐỊNH KỲ] Đã lưu trạng thái tại iter {iter_count}: {ckpt_path.name}")

    total_gpu_time_s = time.perf_counter() - t_train_start
    gpu_hours = round(total_gpu_time_s / 3600.0, 4)

    # 8. Lưu Checkpoint cuối cùng & Đánh giá Primary Endpoint (Held-out Test Hybrid-OOD)
    final_val_res = val_res if (iter_count == args.iters and 'val_res' in locals()) else evaluator.evaluate(model, val_dataset)
    torch.save({
        'iter': args.iters,
        'branch': args.branch,
        'seed': args.seed,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'metrics': final_val_res
    }, run_dir / "final_model.pth")

    print("\n" + "=" * 70)
    print(f"ĐÁNH GIÁ PRIMARY ENDPOINT: HELD-OUT TEST HYBRID-OOD (NHÁNH {args.branch} SEED {args.seed})")
    print("=" * 70)
    # Đo trên final model
    final_ckpt = torch.load(run_dir / "final_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(final_ckpt['model_state'])
    final_ood_res = evaluator.evaluate(model, test_hybrid_ood_dataset)

    # Nạp lại best model để đo Primary Endpoint (nếu best_iter != iters)
    if best_iter == args.iters:
        best_ood_res = final_ood_res
    else:
        best_ckpt = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state'])
        best_ood_res = evaluator.evaluate(model, test_hybrid_ood_dataset)
    print(f">> [PRIMARY ENDPOINT] Best Model Hybrid-OOD PSNR: {best_ood_res.get('psnr_mean', 'N/A')} dB")

    print("\n" + "=" * 70)
    print(f"HOÀN TẤT THỰC NGHIỆM NHÁNH {args.branch} (SEED {args.seed})")
    print(f"-> Tổng thời gian chạy: {total_gpu_time_s:.1f}s ({gpu_hours} GPU hours)")
    print(f"-> Best Iteration trên Val: {best_iter} (Val Hybrid PSNR: {best_hybrid_psnr:.2f} dB)")
    print(f"-> Primary Endpoint (Best Hybrid-OOD): {best_ood_res.get('psnr_mean', 'N/A')} dB")
    print(evaluator.format_table(best_metrics, title=f"KẾT QUẢ CHECKPOINT TỐT NHẤT (BEST VAL) - NHÁNH {args.branch} SEED {args.seed}"))

    summary_data = {
        'branch': args.branch,
        'seed': args.seed,
        'git_commit': commit_hash,
        'total_iters': args.iters,
        'gpu_hours': gpu_hours,
        'iter_0_metrics': {'val': val_res_0, 'ood': ood_res_0},
        'best_val_iter': best_iter,
        'best_val_metrics': best_metrics,
        'final_val_metrics': final_val_res,
        'best_ood_primary_endpoint': best_ood_res,
        'final_ood_primary_endpoint': final_ood_res,
        'batch_checksum_50': batch_checksum,
        'history': history
    }
    with open(run_dir / "summary.json", "w", encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    # Cập nhật kết quả vào CSV tổng: Nếu là smoke_test thì lưu vào runs/smoke/, TUYỆT ĐỐI KHÔNG ghi vào results/
    if args.smoke_test:
        csv_file = REPO_ROOT / "runs" / "smoke" / "smoke_pilot_v2_results.csv"
    else:
        csv_file = REPO_ROOT / "results" / "pilot_v2_results.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    header = "branch,seed,checkpoint_type,iter,gpu_hours,val_psnr_noise,val_psnr_rain,val_psnr_haze,val_psnr_hybrid,val_psnr_mean,primary_endpoint_ood_psnr\n"
    if not csv_file.exists():
        with open(csv_file, "w", encoding='utf-8') as f:
            f.write(header)

    with open(csv_file, "a", encoding='utf-8') as f:
        f.write(f"{args.branch},{args.seed},best_val,{best_iter},{gpu_hours},"
                f"{best_metrics['psnr_noise']},{best_metrics['psnr_rain']},{best_metrics['psnr_haze']},{best_metrics['psnr_hybrid']},{best_metrics['psnr_mean']},"
                f"{best_ood_res.get('psnr_mean', 0.0)}\n")
        f.write(f"{args.branch},{args.seed},final,{args.iters},{gpu_hours},"
                f"{final_val_res['psnr_noise']},{final_val_res['psnr_rain']},{final_val_res['psnr_haze']},{final_val_res['psnr_hybrid']},{final_val_res['psnr_mean']},"
                f"{final_ood_res.get('psnr_mean', 0.0)}\n")

    print(f"Đã cập nhật bảng tổng kết tại: {csv_file}")
    print("=" * 70)


if __name__ == '__main__':
    run_experiment()
