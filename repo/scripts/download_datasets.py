"""Script tự động tải, bung nén và lọc chuẩn xác các tập dữ liệu test & checkpoint cho AiOIR.
Tách biệt chính xác:
- Urban100: đúng 100 ảnh (img_001 -> img_100).
- Rain100L: đúng 100 cặp ảnh test từ Rain100L.zip (loại bỏ hoàn toàn RainTrainL).
- Checkpoint: PromptIR All-in-One Checkpoint (promptir_all.ckpt).
- Benchmark: BSD68 (68 ảnh), Kodak24 (24 ảnh).
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
    if Path('/content/repo').exists():
        return Path('/content/repo')
    for p in [Path('.'), Path('repo'), Path('..')]:
        if (p / 'scripts').exists() and (p / 'baselines').exists():
            return p.resolve()
    return Path('.').resolve()

BASE_DIR = get_base_dir()

def download_file(file_id, dest_path, desc):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        print(f"[ĐÃ CÓ] {desc} tại {dest_path}")
        return True

    print(f"\n[ĐANG TẢI] {desc}...")
    cmd = ['gdown', '--id', file_id, '-O', str(dest_path), '--fuzzy']
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Lỗi tải {desc}: {e}")
        return False

def download_folder(folder_id, dest_dir, desc):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[ĐÃ CÓ] {desc} tại {dest_dir}")
        return True

    print(f"\n[ĐANG TẢI THƯ MỤC] {desc}...")
    cmd = ['gdown', '--folder', f"https://drive.google.com/drive/folders/{folder_id}", '-O', str(dest_dir)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Lỗi tải thư mục {desc}: {e}")
        return False

def unpack_and_flatten_zips(directory, valid_exts={'.png', '.jpg', '.jpeg', '.bmp'}):
    directory = Path(directory)
    if not directory.exists():
        return
    zip_files = list(directory.glob('*.zip')) + list(directory.glob('*/*.zip'))
    for zf in zip_files:
        print(f"Bung nén {zf.name} vào {directory}...")
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

    if not zip_path.exists() or zip_path.stat().st_size < 1024 * 1024:
        ok = download_file('1-3Y7TLZ5EpYYZmFLkRKjV508kcBeibjU', zip_path, 'Rain100L.zip (chính thức 53MB)')
        if not ok or not zip_path.exists():
            print(f"[-] Không thể tải {zip_path.name}. Bỏ qua bước giải nén.")
            return

    print("Đang giải nén Rain100L.zip...")
    with zipfile.ZipFile(str(zip_path), 'r') as z:
        z.extractall(str(rain_root))

    for f in list(rain_root.rglob('*.png')):
        if f.parent == in_dir or f.parent == tar_dir:
            continue
        fname = f.name.lower()
        if 'norain' in fname or 'clean' in fname:
            shutil.move(str(f), str(tar_dir / f.name))
        elif 'rain' in fname:
            shutil.move(str(f), str(in_dir / f.name))

    for sub in list(rain_root.iterdir()):
        if sub.is_dir() and sub.name not in ['input', 'target']:
            shutil.rmtree(sub, ignore_errors=True)

    # Khớp chính xác 100 cặp bằng regex integer index
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    in_files = [f for f in in_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]
    tar_files = [f for f in tar_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts]

    in_map = {}
    for f in in_files:
        m = re.search(r'\d+', f.name)
        if m:
            in_map[int(m.group(0))] = f

    tar_map = {}
    for f in tar_files:
        m = re.search(r'\d+', f.name)
        if m:
            tar_map[int(m.group(0))] = f

    common_keys = sorted(list(set(in_map.keys()) & set(tar_map.keys())))[:100]
    keep_in = set(in_map[k] for k in common_keys)
    keep_tar = set(tar_map[k] for k in common_keys)

    for f in in_files:
        if f not in keep_in:
            f.unlink(missing_ok=True)
    for f in tar_files:
        if f not in keep_tar:
            f.unlink(missing_ok=True)

    print(f"-> Rain100L: Đã hoàn tất {len(list(in_dir.glob('*.png')))} ảnh input, {len(list(tar_dir.glob('*.png')))} ảnh target.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tải tài nguyên dữ liệu và checkpoint cho AiOIR")
    parser.add_argument('--weights_only', action='store_true', help='Chỉ tải checkpoint')
    parser.add_argument('--data_only', action='store_true', help='Chỉ tải dữ liệu test')
    args = parser.parse_args()

    print(f"=== P0-T3: BẮT ĐẦU TẢI DỮ LIỆU & CHECKPOINT (Thư mục gốc: {BASE_DIR}) ===")
    test_root = BASE_DIR / 'data' / 'test'
    weights_root = BASE_DIR / 'baselines' / 'weights'
    weights_root.mkdir(parents=True, exist_ok=True)

    if not args.data_only:
        # 1. Tải PromptIR Checkpoint
        promptir_ckpt = weights_root / 'promptir_all.ckpt'
        download_file('1j-b5Od70pGF7oaCqKAfUzmf-N-xEAjYl', promptir_ckpt, 'PromptIR All-in-One Checkpoint (All.ckpt, ~140MB)')

        # 2. Tải AdaIR Checkpoint Folder (tùy chọn)
        adair_dir = weights_root / 'adair'
        download_folder('1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH', adair_dir, 'AdaIR Pretrained Checkpoints Folder')

    if not args.weights_only:
        # 3. Kodak24
        kodak_dir = test_root / 'denoise' / 'kodak24'
        kodak_zip = test_root / 'denoise' / 'kodak24.zip'
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        kodak_count = sum(1 for f in kodak_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts) if kodak_dir.exists() else 0
        if kodak_count < 24:
            download_file('1F4_mv4oTXhiG-zyG9DI4OO05KqvEKhs9', kodak_zip, 'Kodak24 Denoising Testset (24 ảnh)')
            kodak_dir.mkdir(parents=True, exist_ok=True)
            if kodak_zip.exists():
                with zipfile.ZipFile(str(kodak_zip), 'r') as z:
                    z.extractall(str(kodak_dir))
                unpack_and_flatten_zips(kodak_dir)

        # 4. CBSD68
        bsd_dir = test_root / 'denoise' / 'bsd68'
        bsd_zip = test_root / 'denoise' / 'cbsd68.zip'
        bsd_count = sum(1 for f in bsd_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts) if bsd_dir.exists() else 0
        if bsd_count < 68:
            download_file('1lxXQ_buMll_JVWxKpk5fp0jduW5F_MHe', bsd_zip, 'CBSD68 Denoising Testset (68 ảnh)')
            bsd_dir.mkdir(parents=True, exist_ok=True)
            if bsd_zip.exists():
                with zipfile.ZipFile(str(bsd_zip), 'r') as z:
                    z.extractall(str(bsd_dir))
                unpack_and_flatten_zips(bsd_dir)

        # 5. Urban100
        urban_dir = test_root / 'denoise' / 'urban100'
        urban_count = sum(1 for f in urban_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts) if urban_dir.exists() else 0
        if urban_count < 100:
            download_folder('1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u', urban_dir, 'Urban100 Denoising Testset (100 ảnh)')
            unpack_and_flatten_zips(urban_dir)
            filter_urban100(urban_dir)

        # 6. Rain100L
        setup_clean_rain100l(test_root / 'derain' / 'Rain100L')

        # Báo cáo tổng kết số lượng
        print("\nHoàn tất quá trình kiểm tra và tải tài nguyên.")
        bsd_count = sum(1 for f in bsd_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts) if bsd_dir.exists() else 0
        kodak_count = sum(1 for f in kodak_dir.glob('*') if f.is_file() and f.suffix.lower() in valid_exts) if kodak_dir.exists() else 0
        urban_count = len(list(urban_dir.glob('*.png'))) if urban_dir.exists() else 0
        rain_in_c = len(list((test_root / 'derain' / 'Rain100L' / 'input').glob('*.png'))) if (test_root / 'derain' / 'Rain100L' / 'input').exists() else 0
        rain_tar_c = len(list((test_root / 'derain' / 'Rain100L' / 'target').glob('*.png'))) if (test_root / 'derain' / 'Rain100L' / 'target').exists() else 0

        print(f"-> BSD68   : {bsd_count}/68 ảnh")
        print(f"-> Kodak24 : {kodak_count}/24 ảnh")
        print(f"-> Urban100: {urban_count}/100 ảnh")
        print(f"-> Rain100L: {rain_in_c}/100 ảnh input, {rain_tar_c}/100 ảnh target")

if __name__ == '__main__':
    main()
