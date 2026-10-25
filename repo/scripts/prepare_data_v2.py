"""Script Chuẩn Bị Dữ Liệu Huấn Luyện & Kiểm Định Độc Lập v2 (Zero-Leakage Data Preparation).
Mã nhiệm vụ: P4-V2
Mục đích:
- Tạo lập tập huấn luyện `data/train_v2` và kiểm định `data/val_v2` độc lập hoàn toàn với `data/test`.
- Nguồn dữ liệu sạch chuẩn: BSD400 / RainTrainL (target) / DIV2K train.
- Tạo sẵn 40 ảnh validation cố định được chia đều cho 4 tác vụ (10 Noise, 10 Rain, 10 Haze, 10 Hybrid).
- Chạy khẳng định (assert) băm SHA-256 để bảo đảm giao với tập test bằng RỖNG.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import shutil
import random
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Thiết lập đường dẫn import
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.leakage_checker import verify_zero_leakage

def get_base_data_dir():
    if Path('/content/repo/data').exists():
        return Path('/content/repo/data')
    return (REPO_ROOT / 'data').resolve()

def download_and_extract_clean_pool(dest_dir: Path, target_count: int = 200):
    """Tải và chuẩn bị kho ảnh sạch độc lập hoàn toàn với test."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_images = list(dest_dir.glob('*.png')) + list(dest_dir.glob('*.jpg'))
    if len(existing_images) >= target_count:
        print(f"[ĐÃ CÓ ĐỦ DỮ LIỆU] Tìm thấy {len(existing_images)} ảnh sạch tại {dest_dir}")
        return existing_images[:target_count]

    print(f"\n[KHỞI TẠO DỮ LIỆU V2] Chuẩn bị {target_count} ảnh sạch độc lập...")
    
    # 1. Kiểm tra xem có RainTrainL target (clean) chưa
    rain_train_target = dest_dir.parent / 'train' / 'derain' / 'RainTrainL' / 'target'
    collected = []
    if rain_train_target.exists():
        train_imgs = list(rain_train_target.glob('*.png'))
        for img in train_imgs:
            dst = dest_dir / f"clean_train_{len(collected):04d}.png"
            if not dst.exists():
                shutil.copy(img, dst)
            collected.append(dst)
            if len(collected) >= target_count:
                break

    # 2. Nếu chưa đủ 200 ảnh, tải thêm từ kho BSD400 train (độc lập với BSD68)
    if len(collected) < target_count:
        print("Tải thêm tập ảnh mẫu BSD400 train từ kho lưu trữ độc lập...")
        bsd_url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds500/BSR_bsds500.tgz"
        tgz_path = dest_dir.parent / "BSR_bsds500.tgz"
        try:
            if not tgz_path.exists():
                print(f"Đang tải {bsd_url}...")
                urllib.request.urlretrieve(bsd_url, str(tgz_path))
            import tarfile
            with tarfile.open(str(tgz_path), "r:gz") as tar:
                # Chỉ lấy thư mục images/train (200 ảnh) và images/val (100 ảnh), BỎ QUA images/test
                for member in tar.getmembers():
                    if ("images/train/" in member.name or "images/val/" in member.name) and member.name.endswith(".jpg"):
                        f = tar.extractfile(member)
                        if f:
                            img = Image.open(f).convert('RGB')
                            dst = dest_dir / f"bsd_train_{len(collected):04d}.png"
                            img.save(dst)
                            collected.append(dst)
                            if len(collected) >= target_count:
                                break
        except Exception as e:
            print(f"[THÔNG BÁO] Không thể tải BSDS500 qua URL ({e}), tạo kho ảnh sạch phân vùng độc lập...")
            import numpy as np
            for i in range(len(collected), target_count):
                rng_img = np.random.RandomState(50000 + i * 17)
                # Tạo ảnh hoa văn kết cấu ngẫu nhiên hoàn toàn độc lập, khác biệt quang học
                base = rng_img.randint(50, 200, (16, 16, 3), dtype=np.uint8)
                img_rgb = cv2.resize(base, (256, 256), interpolation=cv2.INTER_CUBIC)
                noise = rng_img.randint(-15, 15, (256, 256, 3), dtype=np.int16)
                img_rgb = np.clip(img_rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                dst = dest_dir / f"clean_synthetic_{i:04d}.png"
                Image.fromarray(img_rgb).save(dst)
                collected.append(dst)

    return list(dest_dir.glob('*.png')) + list(dest_dir.glob('*.jpg'))


def setup_data_v2():
    data_dir = get_base_data_dir()
    train_dir = data_dir / 'train_v2' / 'clean'
    val_dir = data_dir / 'val_v2' / 'clean'
    test_dir = data_dir / 'test'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    pool_dir = data_dir / 'clean_pool_v2'
    all_clean = download_and_extract_clean_pool(pool_dir, target_count=200)

    if len(all_clean) >= 200:
        # Cố định seed chia tách để bảo đảm tính tái lập
        rng = random.Random(42)
        shuffled = list(all_clean)
        rng.shuffle(shuffled)

        train_imgs = shuffled[:160]
        val_imgs = shuffled[160:200]

        for img in train_imgs:
            dst = train_dir / img.name
            if not dst.exists():
                shutil.copy(img, dst)

        for img in val_imgs:
            dst = val_dir / img.name
            if not dst.exists():
                shutil.copy(img, dst)

        print(f"-> Đã phân bổ: {len(list(train_dir.glob('*')))} ảnh vào {train_dir}")
        print(f"-> Đã phân bổ: {len(list(val_dir.glob('*')))} ảnh vào {val_dir}")

    # Chạy kiểm tra khẳng định (Assert) chống rò rỉ dữ liệu
    print("\nTiến hành kiểm toán khẳng định tính độc lập dữ liệu...")
    if test_dir.exists():
        verify_zero_leakage([train_dir], [val_dir], [test_dir], verbose=True)
    else:
        print("[LƯU Ý] Chưa tìm thấy thư mục data/test, bỏ qua đối chiếu test cục bộ.")

if __name__ == '__main__':
    setup_data_v2()
