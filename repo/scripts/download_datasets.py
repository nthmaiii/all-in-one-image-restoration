"""Script tự động tải, bung nén và lọc chuẩn xác các tập dữ liệu test cho AiOIR.
Tách biệt chính xác:
- Urban100: đúng 100 ảnh (img_001 -> img_100).
- Rain100L: đúng 100 cặp ảnh test khớp nhau.
"""

import os
import sys
import re
import shutil
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def get_base_dir():
    if Path('scripts').exists() and Path('baselines').exists():
        return Path('.').resolve()
    elif Path('repo').exists():
        return Path('repo').resolve()
    return Path('.').resolve()

BASE_DIR = get_base_dir()

def filter_urban100(urban_dir):
    urban_dir = Path(urban_dir)
    if not urban_dir.exists():
        return
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    all_imgs = sorted([f for f in urban_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts])
    
    # Ưu tiên các file img_001 đến img_100
    matched = [f for f in all_imgs if re.match(r'^img_?\d+', f.name.lower())]
    if len(matched) >= 100:
        keep_set = set(matched[:100])
    else:
        keep_set = set(all_imgs[:100])
        
    for f in all_imgs:
        if f not in keep_set:
            f.unlink(missing_ok=True)
            
    print(f"-> Urban100: Đã chuẩn hóa chính xác {len(list(urban_dir.glob('*.png')))}/100 ảnh.")

def organize_and_filter_rain100l(rain_root):
    rain_root = Path(rain_root)
    input_dir = rain_root / 'input'
    target_dir = rain_root / 'target'
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    # 1. Tìm các cặp tương ứng bằng stem chuẩn hóa
    in_files = [f for f in input_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
    tar_files = [f for f in target_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]

    in_map = {f.name.lower().replace('norain-', '').replace('rain-', '').replace('.png', '').replace('.jpg', ''): f for f in in_files}
    tar_map = {f.name.lower().replace('norain-', '').replace('rain-', '').replace('.png', '').replace('.jpg', ''): f for f in tar_files}

    common_keys = sorted(
        list(set(in_map.keys()) & set(tar_map.keys())),
        key=lambda k: int(re.search(r'\d+', k).group(0)) if re.search(r'\d+', k) else k
    )[:100]

    keep_in = set(in_map[k] for k in common_keys)
    keep_tar = set(tar_map[k] for k in common_keys)

    for f in in_files:
        if f not in keep_in:
            f.unlink(missing_ok=True)

    for f in tar_files:
        if f not in keep_tar:
            f.unlink(missing_ok=True)

    print(f"-> Rain100L: Đã chuẩn hóa chính xác {len(list(input_dir.glob('*.png')))} ảnh input và {len(list(target_dir.glob('*.png')))} ảnh target.")

def main():
    print(f"=== P0-T3: BẮT ĐẦU CHUẨN HÓA SỐ LƯỢNG ẢNH TEST ({BASE_DIR}) ===")
    test_root = BASE_DIR / 'data' / 'test'

    # 1. BSD68 & Kodak24
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    bsd_dir = test_root / 'denoise' / 'bsd68'
    kodak_dir = test_root / 'denoise' / 'kodak24'
    bsd_count = sum(1 for f in bsd_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts)
    kodak_count = sum(1 for f in kodak_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts)
    print(f"-> BSD68: {bsd_count}/68 ảnh.")
    print(f"-> Kodak24: {kodak_count}/24 ảnh.")

    # 2. Chuẩn hóa Urban100 & Rain100L
    filter_urban100(test_root / 'denoise' / 'urban100')
    organize_and_filter_rain100l(test_root / 'derain' / 'Rain100L')

    print("\nHoàn tất chuẩn hóa số lượng ảnh.")

if __name__ == '__main__':
    main()
