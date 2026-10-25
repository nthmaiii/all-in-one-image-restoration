"""Script kiểm tra số lượng và tính toàn vẹn của các tập dữ liệu test cho AiOIR.
Tiêu chuẩn theo README của PromptIR và AdaIR:
- Denoise: BSD68 (68 ảnh), Urban100 (100 ảnh), Kodak24 (24 ảnh)
- Derain: Rain100L (100 ảnh input + 100 ảnh target)
- Dehaze: SOTS-Outdoor (500 ảnh input + 500 ảnh target)
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
    'dehaze/input': 500,
    'dehaze/target': 500
}

def verify_dataset(root_dir, check_image_open=True):
    root = Path(root_dir)
    print(f"=== KIỂM TRA TẬP DỮ LIỆU TẠI: {root.resolve()} ===")
    
    all_passed = True
    results = {}
    
    for rel_path, expected in EXPECTED_COUNTS.items():
        dir_path = root / rel_path
        if not dir_path.exists():
            print(f"[-] {rel_path:<25}: KHÔNG TỒN TẠI (Kỳ vọng: {expected})")
            results[rel_path] = {'status': 'MISSING', 'count': 0, 'expected': expected}
            all_passed = False
            continue
            
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        images = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
        count = len(images)
        
        status = "ĐẠT" if count == expected else f"LỆCH ({count}/{expected})"
        if count != expected:
            all_passed = False
            
        corrupted = 0
        if check_image_open and count > 0:
            for img_p in images[:10]: # Kiểm tra 10 ảnh đầu tiên xem có đọc được không
                try:
                    with Image.open(img_p) as img:
                        img.verify()
                except Exception as e:
                    corrupted += 1
                    
        corrupt_str = f" | Bị hỏng: {corrupted}" if corrupted > 0 else ""
        print(f"[{'+' if count == expected else '!'}] {rel_path:<25}: {count:>4}/{expected} ảnh [{status}]{corrupt_str}")
        results[rel_path] = {'status': status, 'count': count, 'expected': expected, 'corrupted': corrupted}

    print("=" * 60)
    if all_passed:
        print("[SUCCESS] 100% CÁC TẬP DỮ LIỆU ĐÃ KHỚP CHUẨN!")
    else:
        print("[WARNING] Một số tập dữ liệu chưa đủ hoặc chưa khớp số lượng!")
    return all_passed, results

if __name__ == '__main__':
    default_dir = 'repo/data/test' if Path('repo/data/test').exists() else 'data/test'
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    passed, _ = verify_dataset(data_dir, check_image_open=True)
    sys.exit(0 if passed else 1)
