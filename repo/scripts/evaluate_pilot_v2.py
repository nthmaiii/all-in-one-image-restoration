"""Script Đánh Giá Độc Lập Cho Pilot v2 (Standalone Unified Evaluator).
Mã nhiệm vụ: P4-V2
Cho phép nạp bất kỳ checkpoint nào (A, B, C, D) và đánh giá trên tập Val v2 hoặc Test.
"""

import sys
import argparse
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.unified_evaluator import UnifiedEvaluator
from scripts.train_pilot_v2 import AiOIRValDatasetV2, build_model

def evaluate_checkpoint():
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint bằng UnifiedEvaluator")
    parser.add_argument('--checkpoint', type=str, required=True, help="Đường dẫn file .pth hoặc .ckpt")
    parser.add_argument('--branch', type=str, required=True, choices=['A', 'B', 'C', 'D'], help="Nhánh kiến trúc (A, B, C, D)")
    parser.add_argument('--val_dir', type=str, default="data/val_v2/clean", help="Thư mục ảnh val sạch")
    parser.add_argument('--seed', type=int, default=42, help="Seed tạo tập val")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    print(f"Đang nạp mô hình Nhánh {args.branch}...")
    model = build_model(args.branch, init_weights="", device=device)

    # Nạp weights từ checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
        clean_state = {k.replace('net.', '').replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    print("Mô hình đã nạp thành công.")

    val_path = (REPO_ROOT / args.val_dir).resolve()
    val_dataset = AiOIRValDatasetV2(val_path, samples_per_task=10, seed=args.seed)

    evaluator = UnifiedEvaluator(device=device, use_uint8=True)
    print("\nTiến hành đánh giá thống nhất trên 4 tác vụ (Noise, Rain, Haze, Hybrid)...")
    res = evaluator.evaluate(model, val_dataset)

    print("\n" + "=" * 60)
    print(evaluator.format_table(res, title=f"KẾT QUẢ ĐÁNH GIÁ: {ckpt_path.name} (Nhánh {args.branch})"))
    print("=" * 60)

if __name__ == '__main__':
    evaluate_checkpoint()
