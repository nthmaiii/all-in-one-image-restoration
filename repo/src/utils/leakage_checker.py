"""Module kiểm tra và khẳng định tính độc lập tuyệt đối giữa các tập dữ liệu (Zero Data Leakage Assertion).
Sử dụng mã băm SHA-256 nội dung nhị phân kết hợp mã băm mảng điểm ảnh (pixel array hash)
để ngăn chặn hoàn toàn việc rò rỉ dữ liệu giữa Train/Val và Test.
"""

import sys
import hashlib
from pathlib import Path
from typing import List, Union, Set, Dict, Tuple
from PIL import Image
import numpy as np

def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Tính mã băm SHA-256 từ byte nhị phân của tệp."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_pixel_hash(file_path: Union[str, Path]) -> str:
    """Tính mã băm SHA-256 từ mảng điểm ảnh (chống trùng lặp khác định dạng nén/metadata)."""
    try:
        with Image.open(file_path) as img:
            arr = np.array(img.convert('RGB'))
            return hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception:
        return compute_file_hash(file_path)

def scan_images(directory: Union[str, Path], valid_exts=('.png', '.jpg', '.jpeg', '.bmp')) -> List[Path]:
    """Quét toàn bộ tệp ảnh hợp lệ trong thư mục và thư mục con."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([
        f for f in directory.rglob('*')
        if f.is_file() and f.suffix.lower() in valid_exts
    ])

def verify_zero_leakage(
    train_dirs: Union[List[Union[str, Path]], Union[str, Path]],
    val_dirs: Union[List[Union[str, Path]], Union[str, Path]],
    test_dirs: Union[List[Union[str, Path]], Union[str, Path]],
    verbose: bool = True
) -> bool:
    """Kiểm tra khẳng định (Assertion) rằng không có sự giao thoa dữ liệu giữa Train, Val và Test.
    Nếu phát hiện trùng lặp, lập tức ném ngoại lệ AssertionError kèm danh sách tệp vi phạm.
    """
    if isinstance(train_dirs, (str, Path)):
        train_dirs = [train_dirs]
    if isinstance(val_dirs, (str, Path)):
        val_dirs = [val_dirs]
    if isinstance(test_dirs, (str, Path)):
        test_dirs = [test_dirs]

    # Thu thập ảnh
    train_files = []
    for d in train_dirs:
        train_files.extend(scan_images(d))
    
    val_files = []
    for d in val_dirs:
        val_files.extend(scan_images(d))
        
    test_files = []
    for d in test_dirs:
        test_files.extend(scan_images(d))

    if verbose:
        print("=" * 60)
        print("KIỂM TRA CHỐNG RÒ RỈ DỮ LIỆU TỰ ĐỘNG (ZERO-LEAKAGE AUDIT)")
        print(f"-> Số ảnh Train tìm thấy: {len(train_files)}")
        print(f"-> Số ảnh Val tìm thấy:   {len(val_files)}")
        print(f"-> Số ảnh Test tìm thấy:  {len(test_files)}")

    if len(train_files) == 0:
        raise AssertionError("LỖI: Thư mục Train không chứa bất kỳ ảnh hợp lệ nào!")
    if len(test_files) == 0:
        if verbose:
            print("[CẢNH BÁO] Thư mục Test trống hoặc chưa tải, không thể đối chiếu test.")
        return True

    # Băm ảnh Train & Val
    # Lưu dict: hash -> path
    train_hashes: Dict[str, Path] = {}
    for p in train_files:
        h = compute_pixel_hash(p)
        train_hashes[h] = p

    val_hashes: Dict[str, Path] = {}
    for p in val_files:
        h = compute_pixel_hash(p)
        val_hashes[h] = p

    test_hashes: Dict[str, Path] = {}
    for p in test_files:
        h = compute_pixel_hash(p)
        test_hashes[h] = p

    # 1. Kiểm tra Train vs Test
    train_test_overlap = set(train_hashes.keys()) & set(test_hashes.keys())
    if train_test_overlap:
        msg = f"\n[CHÍ MẠNG] PHÁT HIỆN RÒ RỈ DỮ LIỆU GIỮA TRAIN VÀ TEST ({len(train_test_overlap)} ảnh)!\n"
        for h in list(train_test_overlap)[:10]:
            msg += f" - Trùng lặp: Train [{train_hashes[h]}] <==> Test [{test_hashes[h]}]\n"
        raise AssertionError(msg)

    # 2. Kiểm tra Val vs Test
    val_test_overlap = set(val_hashes.keys()) & set(test_hashes.keys())
    if val_test_overlap:
        msg = f"\n[CHÍ MẠNG] PHÁT HIỆN RÒ RỈ DỮ LIỆU GIỮA VAL VÀ TEST ({len(val_test_overlap)} ảnh)!\n"
        for h in list(val_test_overlap)[:10]:
            msg += f" - Trùng lặp: Val [{val_hashes[h]}] <==> Test [{test_hashes[h]}]\n"
        raise AssertionError(msg)

    # 3. Kiểm tra Train vs Val
    train_val_overlap = set(train_hashes.keys()) & set(val_hashes.keys())
    if train_val_overlap:
        msg = f"\n[LỖI NGHIÊM TRỌNG] PHÁT HIỆN TRÙNG LẶP GIỮA TRAIN VÀ VAL ({len(train_val_overlap)} ảnh)!\n"
        for h in list(train_val_overlap)[:10]:
            msg += f" - Trùng lặp: Train [{train_hashes[h]}] <==> Val [{val_hashes[h]}]\n"
        raise AssertionError(msg)

    if verbose:
        print("[XÁC NHẬN AN TOÀN] Khẳng định giao giữa Train/Val và Test bằng RỖNG.")
        print("Tập dữ liệu hoàn toàn độc lập, đủ điều kiện khoa học để huấn luyện.")
        print("=" * 60)

    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Kiểm tra chống rò rỉ dữ liệu")
    parser.add_argument('--train_dir', type=str, required=True, help="Đường dẫn thư mục train")
    parser.add_argument('--val_dir', type=str, default="", help="Đường dẫn thư mục val")
    parser.add_argument('--test_dir', type=str, required=True, help="Đường dẫn thư mục test")
    args = parser.parse_args()

    val_d = [args.val_dir] if args.val_dir else []
    try:
        verify_zero_leakage([args.train_dir], val_d, [args.test_dir], verbose=True)
        print("KIỂM TRA THÀNH CÔNG: KHÔNG CÓ RÒ RỈ DỮ LIỆU.")
    except AssertionError as e:
        print(e)
        sys.exit(1)
