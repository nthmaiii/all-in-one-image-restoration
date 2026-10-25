"""Script tự động tải và tổ chức các tập dữ liệu và checkpoint cho All-in-One Image Restoration (AiOIR).
Sử dụng các Google Drive ID chính thức đã xác minh, tương thích tối đa với môi trường Google Colab.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

GDRIVE_RESOURCES = {
    # 1. Trọng số mô hình đã huấn luyện (Pretrained Weights)
    'promptir_weights': {
        'id': '1j-b5Od70pGF7oaCqKAfUzmf-N-xEAjYl',
        'target': 'repo/baselines/weights/promptir_all.ckpt',
        'is_folder': False,
        'desc': 'PromptIR All-in-One Checkpoint (All.ckpt, ~140MB)'
    },
    'adair_weights': {
        'id': '1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH',
        'target': 'repo/baselines/weights/adair',
        'is_folder': True,
        'desc': 'AdaIR Pretrained Checkpoints Folder'
    },
    # 2. Tập dữ liệu kiểm chuẩn (Benchmark Test Sets)
    'kodak24': {
        'id': '1F4_mv4oTXhiG-zyG9DI4OO05KqvEKhs9',
        'target': 'repo/data/test/denoise/kodak24.zip',
        'is_folder': False,
        'unzip_to': 'repo/data/test/denoise/kodak24',
        'desc': 'Kodak24 Denoising Testset (24 ảnh)'
    },
    'cbsd68': {
        'id': '1lxXQ_buMll_JVWxKpk5fp0jduW5F_MHe',
        'target': 'repo/data/test/denoise/cbsd68.zip',
        'is_folder': False,
        'unzip_to': 'repo/data/test/denoise/bsd68',
        'desc': 'CBSD68 Denoising Testset (68 ảnh)'
    },
    'urban100': {
        'id': '1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u',
        'target': 'repo/data/test/denoise/urban100',
        'is_folder': True,
        'desc': 'Urban100 Denoising Testset (100 ảnh)'
    },
    'rain100l': {
        'id': '1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f',
        'target': 'repo/data/test/derain/Rain100L',
        'is_folder': True,
        'desc': 'Rain100L Deraining Testset (100 cặp)'
    }
}

def download_item(item_key, item_cfg):
    target = Path(item_cfg['target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    
    if target.exists():
        if target.is_dir() and any(target.iterdir()):
            print(f"[ĐÃ CÓ] {item_cfg['desc']} tại {target}")
            return
        elif target.is_file() and target.stat().st_size > 1024:
            print(f"[ĐÃ CÓ] {item_cfg['desc']} tại {target}")
            return

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

    # Tự động giải nén nếu là file zip
    if 'unzip_to' in item_cfg and target.suffix == '.zip' and target.exists():
        import zipfile
        unzip_path = Path(item_cfg['unzip_to'])
        unzip_path.mkdir(parents=True, exist_ok=True)
        print(f"Đang giải nén {target} -> {unzip_path}...")
        try:
            with zipfile.ZipFile(str(target), 'r') as z:
                z.extractall(str(unzip_path))
            print(f"Giải nén thành công.")
        except Exception as e:
            print(f"Lỗi giải nén {target}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tải tài nguyên dữ liệu và checkpoint cho AiOIR")
    parser.add_argument('--weights_only', action='store_true')
    parser.add_argument('--data_only', action='store_true')
    args = parser.parse_args()

    print("=== P0-T3: BẮT ĐẦU TẢI DỮ LIỆU & CHECKPOINT TỰ ĐỘNG ===")

    # 1. BSD68 nội bộ nếu có
    local_bsd = Path('CNN GAN/DnCNN/testsets/BSD68')
    target_bsd = Path('repo/data/test/denoise/bsd68')
    if local_bsd.exists() and len(list(local_bsd.glob('*.png'))) == 68:
        target_bsd.mkdir(parents=True, exist_ok=True)
        if len(list(target_bsd.glob('*.png'))) < 68:
            print(f"Sao chép 68 ảnh BSD68 từ thư mục nội bộ...")
            for p in local_bsd.glob('*.png'):
                shutil.copy(p, target_bsd / p.name)

    # 2. Xử lý các tài nguyên Drive
    for k, cfg in GDRIVE_RESOURCES.items():
        if 'weights' in k and args.data_only:
            continue
        if 'weights' not in k and args.weights_only:
            continue
        download_item(k, cfg)

    print("\nHoàn tất quá trình kiểm tra và tải tài nguyên.")

if __name__ == '__main__':
    main()
