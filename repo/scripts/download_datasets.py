"""Script tự động tải và tổ chức các tập dữ liệu và checkpoint cho All-in-One Image Restoration (AiOIR).
Sử dụng các Google Drive ID chính thức đã xác minh, tương thích tối đa với môi trường Google Colab.
"""

import os
import sys
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

def unpack_and_flatten_zips(directory, valid_exts={'.png', '.jpg', '.jpeg', '.bmp'}):
    directory = Path(directory)
    if not directory.exists():
        return
    # Giải nén mọi file zip tìm thấy trong thư mục
    zip_files = list(directory.glob('*.zip')) + list(directory.glob('*/*.zip'))
    for zf in zip_files:
        print(f"Bung nén {zf.name} vào {directory}...")
        try:
            with zipfile.ZipFile(str(zf), 'r') as z:
                z.extractall(str(directory))
            zf.unlink(missing_ok=True)
        except Exception as e:
            print(f"Lỗi bung nén {zf}: {e}")

    # Đưa các ảnh nằm trong thư mục con lồng nhau lên thư mục chính (trừ input/target)
    for img_file in list(directory.rglob('*')):
        if img_file.is_file() and img_file.suffix.lower() in valid_exts:
            if img_file.parent.name not in ['input', 'target'] and img_file.parent != directory:
                target_file = directory / img_file.name
                if not target_file.exists():
                    shutil.move(str(img_file), str(target_file))

    # Xóa các thư mục con rỗng không phải input/target
    for sub in list(directory.iterdir()):
        if sub.is_dir() and sub.name not in ['input', 'target']:
            shutil.rmtree(str(sub), ignore_errors=True)

def organize_rain100l(rain_root):
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

def count_images_in_dir(d_path):
    if not d_path.exists():
        return 0
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    return sum(1 for f in d_path.rglob('*') if f.is_file() and f.suffix.lower() in valid_exts)

def main():
    print(f"=== P0-T3: BẮT ĐẦU TỰ ĐỘNG BUNG NÉN & SẮP XẾP DỮ LIỆU ({BASE_DIR}) ===")

    test_root = BASE_DIR / 'data' / 'test'
    test_root.mkdir(parents=True, exist_ok=True)

    # 1. BSD68
    bsd_dir = test_root / 'denoise' / 'bsd68'
    bsd_zip = test_root / 'denoise' / 'cbsd68.zip'
    if bsd_zip.exists():
        unpack_and_flatten_zips(bsd_dir)
    print(f"-> BSD68: {count_images_in_dir(bsd_dir)} ảnh.")

    # 2. Kodak24
    kodak_dir = test_root / 'denoise' / 'kodak24'
    kodak_zip = test_root / 'denoise' / 'kodak24.zip'
    if kodak_zip.exists():
        unpack_and_flatten_zips(kodak_dir)
    print(f"-> Kodak24: {count_images_in_dir(kodak_dir)} ảnh.")

    # 3. Urban100
    urban_dir = test_root / 'denoise' / 'urban100'
    unpack_and_flatten_zips(urban_dir)
    print(f"-> Urban100: {count_images_in_dir(urban_dir)} ảnh.")

    # 4. Rain100L
    rain_dir = test_root / 'derain' / 'Rain100L'
    organize_rain100l(rain_dir)
    in_count = count_images_in_dir(rain_dir / 'input')
    tar_count = count_images_in_dir(rain_dir / 'target')
    print(f"-> Rain100L: {in_count} ảnh input, {tar_count} ảnh target.")

    print("\nHoàn tất bung nén và sắp xếp dữ liệu.")

if __name__ == '__main__':
    main()
