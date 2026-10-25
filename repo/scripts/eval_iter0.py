"""Script Đo Đạc PSNR Chuẩn Mực Tại Iteration 0 (Eval Iteration 0 Baseline).
Mục đích:
- Đo đạc chính xác mức nền (baseline floor) của Model A và Model D (Zero-shot PromptIR)
  trên tập `val_v2` và tập giữ lại `test_hybrid_ood`.
- Kiểm chứng thực nghiệm trên GPU Colab rằng Model B (FD-PromptIR) và Model C (Param-Matched)
  sau khi nạp pretrained PromptIR + Zero-Init có PSNR trùng khớp 100% với Model A tại Iteration 0.
- Thay thế hoàn toàn các con số ước đoán trước đây bằng dữ liệu thực nghiệm có log kiểm chứng.
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import csv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'baselines' / 'promptir'))

from baselines.promptir.net.model import PromptIR
from src.models.fd_promptir import FDPromptIR
from src.models.param_matched_promptir import ParamMatchedPromptIR
from src.evaluation.unified_evaluator import UnifiedEvaluator, AiOIRTestHybridOODDataset
from scripts.train_pilot_v2 import AiOIRValDatasetV2

def evaluate_iteration_0():
    parser = argparse.ArgumentParser(description="Đo đạc PSNR tại Iteration 0")
    parser.add_argument('--weights', type=str, default="baselines/weights/promptir_all.ckpt")
    parser.add_argument('--val_dir', type=str, default="data/val_v2/clean")
    parser.add_argument('--test_dir', type=str, default="data/test")
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--samples_per_task', type=int, default=10)
    parser.add_argument('--ood_images', type=int, default=30)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print(f"ĐO ĐẠC MỨC NỀN PSNR THỰC TẾ TẠI ITERATION 0 TRÊN {device}")
    print("=" * 75)

    weights_path = (REPO_ROOT / args.weights).resolve()
    val_path = (REPO_ROOT / args.val_dir).resolve()
    test_path = (REPO_ROOT / args.test_dir).resolve()

    if not weights_path.exists():
        print(f"[CẢNH BÁO] Không tìm thấy file trọng số tại {weights_path}!")
        print("Trên Colab, hãy tải weights trước: !bash scripts/download_datasets.sh")
        return

    # 1. Khởi tạo Evaluator & Datasets
    evaluator = UnifiedEvaluator(device=device, use_uint8=True)
    val_dataset = AiOIRValDatasetV2(val_path, samples_per_task=args.samples_per_task, seed=42)

    test_clean_dirs = [
        test_path / 'denoise' / 'bsd68',
        test_path / 'denoise' / 'kodak24',
        test_path / 'dehaze' / 'target'
    ]
    ood_dataset = AiOIRTestHybridOODDataset(test_clean_dirs, max_images=args.ood_images, seed=42)

    # 2. Khởi tạo & nạp trọng số 3 mô hình
    print("\n1. Khởi tạo Model A (PromptIR Zero-Shot / Pretrained Baseline)...")
    model_a = PromptIR(decoder=True).to(device)
    ckpt = torch.load(str(weights_path), map_location='cpu')
    state = ckpt.get('state_dict', ckpt.get('params', ckpt))
    clean_state = {k.replace('net.', '').replace('module.', ''): v for k, v in state.items()}
    model_a.load_state_dict(clean_state, strict=False)

    print("\n2. Khởi tạo Model B (FD-PromptIR với Zero-Init)...")
    model_b = FDPromptIR(decoder=True).to(device)
    model_b.load_pretrained_promptir(str(weights_path))

    print("\n3. Khởi tạo Model C (Param-Matched với Zero-Init)...")
    model_c = ParamMatchedPromptIR(decoder=True).to(device)
    model_c.load_pretrained_promptir(str(weights_path))

    models = {
        'A (PromptIR Pretrained / Nhánh D)': model_a,
        'B (FD-PromptIR Zero-Init)': model_b,
        'C (Param-Matched Zero-Init)': model_c
    }

    results = []

    print("\n" + "=" * 75)
    print("TIẾN HÀNH ĐÁNH GIÁ TRÊN TẬP VAL VÀ HELD-OUT TEST HYBRID-OOD...")
    print("=" * 75)

    for name, model in models.items():
        print(f"\n--> Đang đánh giá mô hình: {name}...")
        val_res = evaluator.evaluate(model, val_dataset)
        ood_res = evaluator.evaluate(model, ood_dataset)

        rec = {
            'model_name': name,
            'val_noise_psnr': round(val_res.get('psnr_noise', 0.0), 4),
            'val_rain_psnr': round(val_res.get('psnr_rain', 0.0), 4),
            'val_haze_psnr': round(val_res.get('psnr_haze', 0.0), 4),
            'val_hybrid_psnr': round(val_res.get('psnr_hybrid', 0.0), 4),
            'val_mean_psnr': round(val_res.get('psnr_mean', 0.0), 4),
            'heldout_hybrid_ood_psnr': round(ood_res.get('psnr_mean', 0.0), 4)
        }
        results.append(rec)

    # 3. In bảng so sánh
    print("\n" + "=" * 85)
    print("BẢNG KẾT QUẢ ĐO ĐẠC PSNR TẠI ITERATION 0 [ĐÃ XÁC NHẬN BẰNG ĐO ĐẠC]")
    print("=" * 85)
    print(f"{'Mô hình':<35} | {'Val Noise':<9} | {'Val Rain':<9} | {'Val Haze':<9} | {'Val Hybrid':<10} | {'Hybrid-OOD (Primary)':<20}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_name']:<35} | {r['val_noise_psnr']:<9.2f} | {r['val_rain_psnr']:<9.2f} | {r['val_haze_psnr']:<9.2f} | {r['val_hybrid_psnr']:<10.2f} | {r['heldout_hybrid_ood_psnr']:<20.2f}")
    print("=" * 85)

    # 4. Kiểm tra đẳng cấu toán học (Assert B == A và C == A)
    diff_ba_ood = abs(results[1]['heldout_hybrid_ood_psnr'] - results[0]['heldout_hybrid_ood_psnr'])
    diff_ca_ood = abs(results[2]['heldout_hybrid_ood_psnr'] - results[0]['heldout_hybrid_ood_psnr'])
    print(f"\nKiểm tra độ chênh lệch so với Model A tại Iteration 0:")
    print(f"-> Độ lệch B - A (Hybrid-OOD): {diff_ba_ood:.6f} dB")
    print(f"-> Độ lệch C - A (Hybrid-OOD): {diff_ca_ood:.6f} dB")
    assert diff_ba_ood < 1e-3, f"Model B không trùng khớp Model A tại iter 0 (delta={diff_ba_ood:.4f} dB)!"
    assert diff_ca_ood < 1e-3, f"Model C không trùng khớp Model A tại iter 0 (delta={diff_ca_ood:.4f} dB)!"
    print(">> [XÁC NHẬN BẢO TOÀN ĐỒNG NHẤT THÀNH CÔNG] Model B, C và A hoàn toàn trùng khớp tại iter 0.")

    # 5. Lưu kết quả CSV
    out_csv = (REPO_ROOT / args.output_dir / "iter0_baseline_measured.csv").resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nĐã lưu kết quả đo đạc chính thức vào: {out_csv}")

if __name__ == '__main__':
    evaluate_iteration_0()
