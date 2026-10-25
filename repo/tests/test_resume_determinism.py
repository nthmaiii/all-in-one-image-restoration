"""Unit Test: Kiểm tra tính xác định 100% khi Resume huấn luyện (Resume Determinism Test).
Yêu cầu kiểm toán P4 Fair Pilot v2:
- Chạy 50 iterations liên tục (lưu checkpoint tại iter 25).
- Khởi động lại (resume) từ checkpoint iter 25, chạy tiếp 25 iterations đến iter 50.
- So sánh trọng số mô hình ở iteration 50 giữa 2 lượt chạy:
  Chênh lệch tuyệt đối tối đa (max absolute difference) phải < 1e-5.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.fd_promptir import FDPromptIR
from scripts.train_pilot_v2 import AiOIRDatasetV2, set_seed, seed_worker

def run_training_segment(start_iter: int, target_iter: int, resume_ckpt_path: Path = None, save_ckpt_path: Path = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    set_seed(seed)

    # 1. Dataset & DataLoader
    train_dir = REPO_ROOT / 'data' / 'train_v2' / 'clean'
    dataset = AiOIRDatasetV2(train_dir, patch_size=32, length=100 * 2, seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        generator=gen,
        worker_init_fn=seed_worker,
    )

    # 2. Khởi tạo mô hình FDPromptIR (với num_blocks=[1,1,1,1] để kiểm định toàn diện các khối prompt/attention/GDFN)
    model = FDPromptIR(num_blocks=[1, 1, 1, 1], decoder=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    train_iter = iter(loader)

    # 3. Nạp Resume nếu có
    if resume_ckpt_path is not None:
        print(f">> [RESUME] Nạp trạng thái từ: {resume_ckpt_path.name}...")
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        torch.set_rng_state(ckpt['torch_rng_state'].cpu())
        if torch.cuda.is_available() and ckpt.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        np.random.set_state(ckpt['numpy_rng_state'])
        random.setstate(ckpt['python_rng_state'])
        
        # Đồng bộ thứ tự batch bằng cách tua nhanh
        for _ in range(start_iter):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                next(train_iter)
        print(f">> Đã đồng bộ RNG, Optimizer, Scheduler và tua nhanh {start_iter} batch thành công.")

    # 4. Huấn luyện từ start_iter đến target_iter
    model.train()
    iter_count = start_iter
    while iter_count < target_iter:
        iter_count += 1
        try:
            deg, clean, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            deg, clean, _ = next(train_iter)

        deg = deg.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred = model(deg)
            loss = F.l1_loss(pred, clean)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Lưu checkpoint nếu chạm mốc yêu cầu
        if save_ckpt_path is not None and iter_count == 25:
            torch.save({
                'iter': iter_count,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
            }, save_ckpt_path)
            print(f">> [CHECKPOINT TẠI ITER 25 ĐÃ LƯU] {save_ckpt_path.name}")

    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def test_resume_determinism():
    print("=" * 70)
    print("BẮT ĐẦU KIỂM ĐỊNH TÍNH XÁC ĐỊNH KHI RESUME (RESUME DETERMINISM TEST)")
    print("=" * 70)
    temp_dir = REPO_ROOT / 'runs' / 'test_determinism'
    temp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_iter25 = temp_dir / 'checkpoint_iter_25.pth'

    t0 = time.time()
    # 1. Chạy 50 iterations liên tục (Seamless Run)
    print("\n--- [LƯỢT 1: CHẠY LIÊN TỤC 50 ITERATIONS (LƯU TẠI ITER 25)] ---")
    weights_seamless = run_training_segment(
        start_iter=0,
        target_iter=50,
        resume_ckpt_path=None,
        save_ckpt_path=ckpt_iter25
    )
    t_seamless = time.time() - t0
    print(f">> Lượt 1 hoàn tất sau {t_seamless:.2f}s.")

    # 2. Chạy tiếp tục (Resumed Run) từ iter 25 đến iter 50
    t1 = time.time()
    print("\n--- [LƯỢT 2: RESUME TỪ ITER 25 ĐẾN ITER 50] ---")
    weights_resumed = run_training_segment(
        start_iter=25,
        target_iter=50,
        resume_ckpt_path=ckpt_iter25,
        save_ckpt_path=None
    )
    t_resumed = time.time() - t1
    print(f">> Lượt 2 hoàn tất sau {t_resumed:.2f}s.")

    # 3. So sánh trọng số tại iter 50
    print("\n" + "=" * 70)
    print("SO SÁNH TRỌNG SỐ TẠI ITERATION 50 (SEAMLESS vs RESUMED)")
    print("=" * 70)

    max_diff = 0.0
    diff_tensors = 0
    total_tensors = len(weights_seamless)

    for k in sorted(weights_seamless.keys()):
        w_s = weights_seamless[k].float()
        w_r = weights_resumed[k].float()
        diff = torch.max(torch.abs(w_s - w_r)).item()
        if diff > max_diff:
            max_diff = diff
        if diff > 1e-5:
            diff_tensors += 1
            print(f"[CẢNH BÁO LỆCH] Tensor: {k} | Max Abs Diff: {diff:.2e}")

    print(f"Tổng số tensor kiểm tra: {total_tensors}")
    print(f"Chênh lệch tuyệt đối tối đa (Max Absolute Difference): {max_diff:.2e}")
    print(f"Số tensor vượt ngưỡng 1e-5: {diff_tensors}")

    if max_diff < 1e-5:
        print(f"\n[KẾT LUẬN KIỂM ĐỊNH: ĐẠT TUYỆT ĐỐI (PASS)]")
        print(f"-> Quá trình resume đạt tính xác định 100% (Bit-for-bit Deterministic).")
        print(f"-> Chênh lệch tối đa: {max_diff:.2e} < 1e-5.")
    else:
        raise AssertionError(f"[KẾT LUẬN KIỂM ĐỊNH: THẤT BẠI] Max diff {max_diff:.2e} >= 1e-5!")
    print("=" * 70)


if __name__ == '__main__':
    test_resume_determinism()
