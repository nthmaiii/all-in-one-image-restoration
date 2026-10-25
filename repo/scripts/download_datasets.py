"""Script tự động tải, bung nén và lọc chuẩn xác các tập dữ liệu test cho AiOIR.
Tách biệt chính xác:
- Urban100: 100 ảnh (img_001 -> img_100), loại bỏ các tập super-resolution khác (Set5, Set14, Manga...).
- Rain100L: 100 cặp ảnh test (rain-001 -> rain-100 và norain-001 -> norain-100), loại bỏ tập train.
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

def unpack_and_flatten_zips(directory, valid_exts={'.png', '.jpg', '.jpeg', '.bmp'}):
    directory = Path(directory)
    if not directory.exists():
        return
    zip_files = list(directory.glob('*.zip')) + list(directory.glob('*/*.zip'))
    for zf in zip_files:
        print(f"Bung nén {zf.name}...")
        try:
            with zipfile.ZipFile(str(zf), 'r') as z:
                z.extractall(str(directory))
            zf.unlink(missing_ok=True)
        except Exception as e:
            print(f"Lỗi bung nén {zf}: {e}")

    for img_file in list(directory.rglob('*')):
        if img_file.is_file() and img_file.suffix.lower() in valid_exts:
            if img_file.parent.name not in ['input', 'target'] and img_file.parent != directory:
                target_file = directory / img_file.name
                if not target_file.exists():
                    shutil.move(str(img_file), str(target_file))

    for sub in list(directory.iterdir()):
        if sub.is_dir() and sub.name not in ['input', 'target']:
            shutil.rmtree(str(sub), ignore_errors=True)

def filter_urban100(urban_dir):
    """Giữ lại đúng 100 ảnh của Urban100 (img_001.png đến img_100.png), loại bỏ Set5, Set14, Manga109..."""
    urban_dir = Path(urban_dir)
    unpack_and_flatten_zips(urban_dir)
    
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    all_imgs = [f for f in urban_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
    
    kept = 0
    removed = 0
    for f in all_imgs:
        if re.match(r'^img_?\d+\.(png|jpg)$', f.name.lower()):
            m = re.search(r'\d+', f.name)
            if m and (1 <= int(m.group(0)) <= 100):
                kept += 1
                continue
        f.unlink(missing_ok=True)
        removed += 1
    print(f"-> Urban100: Đã lọc chính xác {kept} ảnh chuẩn (loại bỏ {removed} ảnh từ các tập khác).")

def organize_and_filter_rain100l(rain_root):
    """Tổ chức và lọc đúng 100 cặp ảnh test của Rain100L (chỉ số 1 -> 100)."""
    rain_root = Path(rain_root)
    unpack_and_flatten_zips(rain_root)
    
    input_dir = rain_root / 'input'
    target_dir = rain_root / 'target'
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    all_imgs = [f for f in rain_root.rglob('*') if f.is_file() and f.suffix.lower() in valid_exts]
    
    for f in all_imgs:
        if f.parent == input_dir or f.parent == target_dir:
            continue
        lower_name = f.name.lower()
        if 'rain' in lower_name and 'norain' not in lower_name and 'clean' not in lower_name:
            dest = input_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
        elif 'norain' in lower_name or 'clean' in lower_name or 'target' in lower_name or 'gt' in str(f.parent).lower():
            dest = target_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))

    # Lọc chỉ giữ lại index 1 đến 100 cho tập test
    for d, tag in [(input_dir, 'input'), (target_dir, 'target')]:
        imgs = [f for f in d.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
        kept = 0
        removed = 0
        for f in imgs:
            m = re.search(r'\d+', f.name)
            if m and (1 <= int(m.group(0)) <= 100):
                kept += 1
                continue
            f.unlink(missing_ok=True)
            removed += 1
        print(f"-> Rain100L ({tag}): Đã lọc chính xác {kept} ảnh test (loại bỏ {removed} ảnh train).")

def main():
    print(f"=== P0-T3: BẮT ĐẦU LỌC DỮ LIỆU CHUẨN XÁC THEO BENCHMARK ({BASE_DIR}) ===")
    test_root = BASE_DIR / 'data' / 'test'

    # 1. BSD68 & Kodak24
    bsd_dir = test_root / 'denoise' / 'bsd68'
    kodak_dir = test_root / 'denoise' / 'kodak24'
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    bsd_count = sum(1 for f in bsd_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts)
    kodak_count = sum(1 for f in kodak_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts)
    print(f"-> BSD68: {bsd_count} ảnh.")
    print(f"-> Kodak24: {kodak_count} ảnh.")

    # 2. Lọc Urban100
    filter_urban100(test_root / 'denoise' / 'urban100')

    # 3. Lọc Rain100L
    organize_and_filter_rain100l(test_root / 'derain' / 'Rain100L')

    print("\nHoàn tất chuẩn hóa số lượng ảnh.")

if __name__ == '__main__':
    main()
