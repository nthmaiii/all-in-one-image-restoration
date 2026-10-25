"""Script Đo Lường Cosine Similarity Trực Tiếp Trên Trọng Số Pretrained PromptIR.
Mã nhiệm vụ: Khảo sát cấu trúc hình học của Prompt Tokens trong checkpoint thực tế.
Mục tiêu:
- Nạp checkpoint PromptIR pretrained (promptir_all.ckpt).
- Đo ma trận Gram Cosine giữa 5 prompt tokens tại từng tầng decoder (L1, L2, L3).
- Thực hiện song song 2 phương pháp:
  + Cách 1: Flat Tensor (Không gian toàn phần C x H x W).
  + Cách 2: Channel-Pooled (Không gian biểu diễn kênh C sau Spatial Pooling).
- In ma trận chi tiết 5x5, Mean |cos|, Max |cos| cho từng tầng.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parents[1]

def format_matrix_5x5(mat: np.ndarray) -> str:
    """Định dạng ma trận 5x5 thành chuỗi dễ đọc."""
    header = "       |   Token 0 |   Token 1 |   Token 2 |   Token 3 |   Token 4 |"
    sep =    "-------+-----------+-----------+-----------+-----------+-----------+"
    lines = [header, sep]
    for i in range(5):
        row_str = f"Token {i} |"
        for j in range(5):
            val = mat[i, j]
            row_str += f"   {val:8.4f} |"
        lines.append(row_str)
    return "\n".join(lines)


def analyze_checkpoint(ckpt_path: str):
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"LỖI: Không tìm thấy checkpoint tại: {ckpt_file}")
        print("Vui lòng tải checkpoint về hoặc chạy script này trên Google Colab:")
        print(f"  python scripts/measure_prompt_cosine.py --checkpoint baselines/weights/promptir_all.ckpt")
        sys.exit(1)

    print("=" * 75)
    print(f"ĐO LƯỜNG HÌNH HỌC PROMPT TOKENS: {ckpt_file.name}")
    print("=" * 75)

    ckpt = torch.load(str(ckpt_file), map_location='cpu')
    state = ckpt.get('state_dict', ckpt.get('params', ckpt))

    levels = [
        ('Level 1 (Decoder Level 1)', 'prompt1.prompt_param', 64, 64),
        ('Level 2 (Decoder Level 2)', 'prompt2.prompt_param', 128, 32),
        ('Level 3 (Decoder Level 3)', 'prompt3.prompt_param', 320, 16),
    ]

    for lvl_name, param_key, C, S in levels:
        print(f"\n>>> {lvl_name} [C={C}, H=W={S}] <<<")

        # Tìm trọng số trong state_dict
        matched_k = None
        for k in state.keys():
            clean_k = k.replace('net.', '').replace('module.', '')
            if clean_k == param_key or param_key in clean_k:
                matched_k = k
                break

        if matched_k is None:
            print(f"  [CẢNH BÁO] Không tìm thấy key '{param_key}' trong checkpoint.")
            continue

        param = state[matched_k].float()  # (1, 5, C, S, S)
        N = param.shape[1]
        D = C * S * S

        # ----------------------------------------------------
        # Cách 1: Flat Cosine (Toàn bộ C x H x W)
        # ----------------------------------------------------
        flat = param.squeeze(0).view(N, -1)  # (5, D)
        norm_flat = F.normalize(flat, p=2, dim=1)
        gram_flat = torch.mm(norm_flat, norm_flat.t()).numpy()

        off_flat = gram_flat[~np.eye(N, dtype=bool)]
        mean_cos_flat = np.mean(np.abs(off_flat))
        max_cos_flat = np.max(np.abs(off_flat))

        print(f"\n1. Phương Pháp FLAT (D = {D:,} chiều):")
        print(f"   - Mean |cos(u_i, u_j)|: {mean_cos_flat:.6f}")
        print(f"   - Max  |cos(u_i, u_j)|: {max_cos_flat:.6f}")
        print("   - Ma trận Gram Cosine (Flat):")
        print(format_matrix_5x5(gram_flat))

        # ----------------------------------------------------
        # Cách 2: Channel-Pooled Cosine (Spatial Average Pooling)
        # ----------------------------------------------------
        v = param.squeeze(0).mean(dim=(-2, -1))  # (5, C)
        norm_v = F.normalize(v, p=2, dim=1)
        gram_pool = torch.mm(norm_v, norm_v.t()).numpy()

        off_pool = gram_pool[~np.eye(N, dtype=bool)]
        mean_cos_pool = np.mean(np.abs(off_pool))
        max_cos_pool = np.max(np.abs(off_pool))

        print(f"\n2. Phương Pháp CHANNEL-POOLED (C = {C} chiều kênh):")
        print(f"   - Mean |cos(v_i, v_j)|: {mean_cos_pool:.4f}")
        print(f"   - Max  |cos(v_i, v_j)|: {max_cos_pool:.4f}")
        print("   - Ma trận Gram Cosine (Channel-Pooled):")
        print(format_matrix_5x5(gram_pool))

    print("\n" + "=" * 75)
    print("HOÀN TẤT PHÉP ĐO TRỰC TIẾP TRÊN CHECKPOINT PRETRAINED.")
    print("=" * 75)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Đo ma trận cosine similarity trên PromptIR")
    parser.add_argument('--checkpoint', type=str, default="baselines/weights/promptir_all.ckpt",
                        help="Đường dẫn file trọng số promptir_all.ckpt")
    args = parser.parse_args()
    analyze_checkpoint(args.checkpoint)
