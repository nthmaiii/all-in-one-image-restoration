"""Script tự động tải, bung nén và lọc chuẩn xác các tập dữ liệu test cho AiOIR.
Tách biệt chính xác:
- Urban100: đúng 100 ảnh (img_001 -> img_100).
- Rain100L: đúng 100 cặp ảnh test từ Rain100L.zip (loại bỏ hoàn toàn RainTrainL).
"""

import os
import sys
import re
import shutil
import zipfile
import subprocess
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
    
    matched = [f for f in all_imgs if re.match(r'^img_?\d+', f.name.lower())]
    if len(matched) >= 100:
        keep_set = set(matched[:100])
    else:
        keep_set = set(all_imgs[:100])
        
    for f in all_imgs:
        if f not in keep_set:
            f.unlink(missing_ok=True)
            
    print(f"-> Urban100: Đã chuẩn hóa chính xác {len(list(urban_dir.glob('*.png')))}/100 ảnh.")

def setup_clean_rain100l(rain_root):
    rain_root = Path(rain_root)
    zip_path = rain_root.parent / 'Rain100L.zip'
    
    in_dir = rain_root / 'input'
    tar_dir = rain_root / 'target'
    
    # Kiểm tra xem đã có đúng 100 cặp chuẩn chưa
    if in_dir.exists() and tar_dir.exists():
        in_c = len(list(in_dir.glob('*.png')))
        tar_c = len(list(tar_dir.glob('*.png')))
        if in_c == 100 and tar_c == 100:
            print(f"-> Rain100L: Đã sẵn sàng chuẩn 100 cặp ({in_c} input, {tar_c} target).")
            return

    print("Thiết lập lại thư mục Rain100L test chuẩn...")
    if rain_root.exists():
        shutil.rmtree(rain_root, ignore_errors=True)
    rain_root.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    tar_dir.mkdir(parents=True, exist_ok=True)

    # Tải file Rain100L.zip trực tiếp (53MB) nếu chưa có
    if not zip_path.exists() or zip_path.stat().st_size < 1024*1024:
        print("Đang tải Rain100L.zip (chính thức 53MB)...")
        cmd = ['gdown', '--id', '1-3Y7TLZ5EpYYZmFLkRKjV508kcBeibjU', '-O', str(zip_path)]
        subprocess.run(cmd, check=True)

    print("Đang giải nén Rain100L.zip...")
    with zipfile.ZipFile(str(zip_path), 'r') as z:
        z.extractall(str(rain_root))

    # Di chuyển đúng input và target
    for f in list(rain_root.rglob('*.png')):
        if f.parent == in_dir or f.parent == tar_dir:
            continue
        fname = f.name.lower()
        if 'norain' in fname or 'clean' in fname:
            shutil.move(str(f), str(tar_dir / f.name))
        elif 'rain' in fname:
            shutil.move(str(f), str(in_dir / f.name))

    # Xóa các thư mục con rác
    for sub in list(rain_root.iterdir()):
        if sub.is_dir() and sub.name not in ['input', 'target']:
            shutil.rmtree(sub, ignore_errors=True)

    print(f"-> Rain100L: Đã hoàn tất {len(list(in_dir.glob('*.png')))} ảnh input, {len(list(tar_dir.glob('*.png')))} ảnh target.")

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

    # 2. Chuẩn hóa Urban100
    filter_urban100(test_root / 'denoise' / 'urban100')

    # 3. Chuẩn hóa Rain100L
    setup_clean_rain100l(test_root / 'derain' / 'Rain100L')

    print("\nHoàn tất chuẩn hóa số lượng ảnh.")

if __name__ == '__main__':
    main()
