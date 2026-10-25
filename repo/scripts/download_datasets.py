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
    # Tự động phát hiện thư mục gốc
    if Path('scripts').exists() and Path('baselines').exists():
        return Path('.').resolve()
    elif Path('repo').exists():
        return Path('repo').resolve()
    return Path('.').resolve()

BASE_DIR = get_base_dir()

GDRIVE_RESOURCES = {
    # 1. Trọng số mô hình đã huấn luyện (Pretrained Weights)
    'promptir_weights': {
        'id': '1j-b5Od70pGF7oaCqKAfUzmf-N-xEAjYl',
        'rel_target': 'baselines/weights/promptir_all.ckpt',
        'is_folder': False,
        'desc': 'PromptIR All-in-One Checkpoint (All.ckpt, ~140MB)'
    },
    'adair_weights': {
        'id': '1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH',
        'rel_target': 'baselines/weights/adair',
        'is_folder': True,
        'desc': 'AdaIR Pretrained Checkpoints Folder'
    },
    # 2. Tập dữ liệu kiểm chuẩn (Benchmark Test Sets)
    'kodak24': {
        'id': '1F4_mv4oTXhiG-zyG9DI4OO05KqvEKhs9',
        'rel_target': 'data/test/denoise/kodak24.zip',
        'is_folder': False,
        'unzip_to': 'data/test/denoise/kodak24',
        'expected_count': 24,
        'desc': 'Kodak24 Denoising Testset (24 ảnh)'
    },
    'cbsd68': {
        'id': '1lxXQ_buMll_JVWxKpk5fp0jduW5F_MHe',
        'rel_target': 'data/test/denoise/cbsd68.zip',
        'is_folder': False,
        'unzip_to': 'data/test/denoise/bsd68',
        'expected_count': 68,
        'desc': 'CBSD68 Denoising Testset (68 ảnh)'
    },
    'urban100': {
        'id': '1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u',
        'rel_target': 'data/test/denoise/urban100',
        'is_folder': True,
        'expected_count': 100,
        'desc': 'Urban100 Denoising Testset (100 ảnh)'
    },
    'rain100l': {
        'id': '1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f',
        'rel_target': 'data/test/derain/Rain100L',
        'is_folder': True,
        'desc': 'Rain100L Deraining Testset (100 cặp)'
    }
}

def count_images_in_dir(d_path):
    if not d_path.exists():
        return 0
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    return sum(1 for f in d_path.rglob('*') if f.is_file() and f.suffix.lower() in valid_exts)

def extract_and_organize_zip(zip_file, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Đang giải nén {zip_file} -> {dest_dir}...")
    try:
        with zipfile.ZipFile(str(zip_file), 'r') as z:
            z.extractall(str(dest_dir))
        
        # Nếu giải nén tạo thư mục con lồng nhau, đưa toàn bộ file ảnh lên thư mục gốc của dest_dir
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        for img_file in list(dest_dir.rglob('*')):
            if img_file.is_file() and img_file.suffix.lower() in valid_exts:
                target_file = dest_dir / img_file.name
                if target_file.resolve() != img_file.resolve():
                    shutil.move(str(img_file), str(target_file))
        
        # Xóa các thư mục rỗng lồng nhau
        for sub in list(dest_dir.iterdir()):
            if sub.is_dir() and sub.name not in ['input', 'target']:
                shutil.rmtree(str(sub), ignore_errors=True)

        print(f"-> Giải nén thành công, hiện có {count_images_in_dir(dest_dir)} ảnh trong {dest_dir}.")
    except Exception as e:
        print(f"Lỗi giải nén {zip_file}: {e}")

def organize_rain100l(rain_root):
    """Tự động tổ chức cấu trúc input và target cho Rain100L nếu chưa có."""
    rain_root = Path(rain_root)
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
            shutil.move(str(f), str(input_dir / f.name))
        elif 'norain' in lower_name or 'clean' in lower_name or 'target' in lower_name or 'gt' in str(f.parent).lower():
            shutil.move(str(f), str(target_dir / f.name))

def download_item(item_key, item_cfg):
    target = (BASE_DIR / item_cfg['rel_target']).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    # 1. Nếu là file zip, kiểm tra xem đã giải nén chưa
    if 'unzip_to' in item_cfg:
        unzip_path = (BASE_DIR / item_cfg['unzip_to']).resolve()
        img_count = count_images_in_dir(unzip_path)
        expected = item_cfg.get('expected_count', 1)
        if img_count >= expected:
            print(f"[ĐÃ CÓ ĐỦ] {item_cfg['desc']} ({img_count}/{expected} ảnh) tại {unzip_path}")
            return
        elif target.exists() and target.is_file() and target.stat().st_size > 1024:
            # File zip đã có, chỉ cần giải nén
            extract_and_organize_zip(target, unzip_path)
            return

    # 2. Nếu là thư mục dữ liệu thông thường
    if item_cfg.get('is_folder', False) and 'weights' not in item_key:
        img_count = count_images_in_dir(target)
        expected = item_cfg.get('expected_count', 1)
        if img_count >= expected:
            print(f"[ĐÃ CÓ ĐỦ] {item_cfg['desc']} ({img_count} ảnh) tại {target}")
            if 'rain100l' in item_key:
                organize_rain100l(target)
            return

    # 3. Nếu là file checkpoint trọng số
    if 'weights' in item_key:
        if target.exists() and target.is_file() and target.stat().st_size > 1024*1024:
            print(f"[ĐÃ CÓ] {item_cfg['desc']} tại {target}")
            return
        if target.exists() and target.is_dir() and any(target.iterdir()):
            print(f"[ĐÃ CÓ] {item_cfg['desc']} tại {target}")
            return

    # 4. Tiến hành tải bằng gdown
    print(f"\n[ĐANG TẢI] {item_cfg['desc']}...")
    cmd = ['gdown']
    if item_cfg.get('is_folder', False):
        cmd.extend(['--folder', f"https://drive.google.com/drive/folders/{item_cfg['id']}"])
    else:
        cmd.extend(['--id', item_cfg['id']])
    cmd.extend(['-O', str(target)])

    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Lỗi tải {item_key}: {e}")
        return

    # Giải nén sau khi tải xong nếu có cấu hình
    if 'unzip_to' in item_cfg and target.suffix == '.zip' and target.exists():
        unzip_path = (BASE_DIR / item_cfg['unzip_to']).resolve()
        extract_and_organize_zip(target, unzip_path)
    
    if 'rain100l' in item_key and target.exists():
        organize_rain100l(target)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tải tài nguyên dữ liệu và checkpoint cho AiOIR")
    parser.add_argument('--weights_only', action='store_true')
    parser.add_argument('--data_only', action='store_true')
    args = parser.parse_args()

    print(f"=== P0-T3: BẮT ĐẦU TẢI DỮ LIỆU & CHECKPOINT (Thư mục gốc: {BASE_DIR}) ===")

    # 1. Sao chép ảnh BSD68 nội bộ nếu có
    local_bsd = Path('CNN GAN/DnCNN/testsets/BSD68')
    target_bsd = (BASE_DIR / 'data/test/denoise/bsd68').resolve()
    if local_bsd.exists() and len(list(local_bsd.glob('*.png'))) == 68:
        target_bsd.mkdir(parents=True, exist_ok=True)
        if count_images_in_dir(target_bsd) < 68:
            print(f"Sao chép 68 ảnh BSD68 từ {local_bsd} -> {target_bsd}...")
            for p in local_bsd.glob('*.png'):
                shutil.copy(p, target_bsd / p.name)

    # 2. Xử lý các tài nguyên Google Drive
    for k, cfg in GDRIVE_RESOURCES.items():
        if 'weights' in k and args.data_only:
            continue
        if 'weights' not in k and args.weights_only:
            continue
        download_item(k, cfg)

    # 3. Đồng bộ lại sang cả thư mục repo/ lồng nhau nếu có
    nested_data = Path('repo/data/test')
    main_data = (BASE_DIR / 'data/test').resolve()
    if nested_data.resolve() != main_data and main_data.exists():
        nested_data.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(main_data, nested_data, dirs_exist_ok=True)
        except Exception:
            pass

    print("\nHoàn tất quá trình kiểm tra, tải và giải nén tài nguyên.")

if __name__ == '__main__':
    main()
