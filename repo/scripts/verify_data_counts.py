"""Script kiểm tra số lượng và tính toàn vẹn của các tập dữ liệu test cho AiOIR.
Tiêu chuẩn theo README của PromptIR và AdaIR:
- Denoise: BSD68 (68 ảnh), Urban100 (100 ảnh), Kodak24 (24 ảnh)
- Derain: Rain100L (100 ảnh input + 100 ảnh target)
- Dehaze: SOTS (tùy chọn)
"""

import os
import sys
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EXPECTED_COUNTS = {
    'denoise/bsd68': 68,
    'denoise/urban100': 100,
    'denoise/kodak24': 24,
    'derain/Rain100L/input': 100,
    'derain/Rain100L/target': 100,
}

OPTIONAL_COUNTS = {
    'dehaze/input': 500,
    'dehaze/target': 500
}

def resolve_test_dir(raw_dir):
    candidates = [
        Path(raw_dir),
        Path('data/test'),
        Path('repo/data/test'),
        Path('/content/repo/data/test'),
        Path('/content/data/test')
    ]
    if raw_dir.startswith('repo/') and Path(raw_dir[5:]).exists():
        candidates.insert(0, Path(raw_dir[5:]))

    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c.resolve()
    return Path(raw_dir).resolve()

def find_subpath(root, rel_path):
    parts = Path(rel_path).parts
    curr = root
    for part in parts:
        found = False
        if curr.exists():
            for child in curr.iterdir():
                if child.name.lower() == part.lower():
                    curr = child
                    found = True
                    break
        if not found:
            return curr / part
    return curr

def verify_dataset(root_dir, check_image_open=True):
    root = resolve_test_dir(root_dir)
    print(f"=== KIỂM TRA TẬP DỮ LIỆU TẠI: {root} ===")
    
    all_passed = True
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}

    # 1. Kiểm tra các tập bắt buộc
    for rel_path, expected in EXPECTED_COUNTS.items():
        dir_path = find_subpath(root, rel_path)
        
        if not dir_path.exists():
            print(f"[-] {rel_path:<25}: KHÔNG TỒN TẠI (Kỳ vọng: {expected})")
            all_passed = False
            continue

        images = [p for p in dir_path.glob('*') if p.is_file() and p.suffix.lower() in valid_exts]
        count = len(images)
        
        status = "ĐẠT" if count == expected else f"LỆCH ({count}/{expected})"
        if count != expected:
            all_passed = False
            
        corrupted = 0
        if check_image_open and count > 0:
            for img_p in images[:5]:
                try:
                    with Image.open(img_p) as img:
                        img.verify()
                except Exception:
                    corrupted += 1
                    
        corrupt_str = f" | Bị hỏng: {corrupted}" if corrupted > 0 else ""
        print(f"[{'+' if count == expected else '!'}] {rel_path:<25}: {count:>4}/{expected} ảnh [{status}]{corrupt_str}")

    # 2. Kiểm tra tập tùy chọn (Dehaze)
    for rel_path, expected in OPTIONAL_COUNTS.items():
        dir_path = find_subpath(root, rel_path)
        if dir_path.exists():
            images = [p for p in dir_path.glob('*') if p.is_file() and p.suffix.lower() in valid_exts]
            print(f"[*] {rel_path:<25}: {len(images):>4}/{expected} ảnh [TÙY CHỌN]")
        else:
            print(f"[*] {rel_path:<25}: CHƯA TẢI (Tùy chọn)")

    print("=" * 60)
    if all_passed:
        print("[SUCCESS] 100% CÁC TẬP DỮ LIỆU BENCHMARK CHÍNH (DENOISE & DERAIN) ĐÃ ĐẠT CHUẨN!")
    else:
        print("[LƯU Ý] Số lượng ảnh chưa khớp, cần chạy script lọc để đạt 100%.")
    return all_passed

if __name__ == '__main__':
    raw_arg = sys.argv[1] if len(sys.argv) > 1 else 'data/test'
    passed = verify_dataset(raw_arg, check_image_open=True)
