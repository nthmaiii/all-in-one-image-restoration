"""Module kiểm tra và khẳng định tính độc lập tuyệt đối giữa các tập dữ liệu (Zero Data Leakage Assertion).
Sử dụng song song:
1. Mã băm mật mã SHA-256 nội dung nhị phân và mảng điểm ảnh (Cryptographic Pixel Hash).
2. Mã băm sai phân nhận thức (Difference Perceptual Hash - dHash 64-bit) để phát hiện ảnh gần trùng (near-duplicates).
Nguyên tắc: FAIL-CLOSED (Nếu thư mục Test trống hoặc không tồn tại -> DỪNG NGAY TIẾN TRÌNH).
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

def compute_dhash(file_path: Union[str, Path], hash_size: int = 8) -> int:
    """Tính mã băm sai phân nhận thức (Difference Perceptual Hash - dHash, 64 bits).
    Chống trùng lặp thị giác (ảnh bị đổi định dạng nén JPEG/PNG, hoặc resize nhẹ).
    """
    try:
        with Image.open(file_path) as img:
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(img, dtype=np.int32)
            diff = pixels[:, 1:] > pixels[:, :-1]
            hash_val = 0
            for bit in diff.flatten():
                hash_val = (hash_val << 1) | int(bit)
            return hash_val
    except Exception:
        return 0

def hamming_distance(h1: int, h2: int) -> int:
    """Khoảng cách Hamming giữa hai mã băm dHash 64-bit."""
    return bin(h1 ^ h2).count('1')

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
    near_dup_threshold: int = 4,
    verbose: bool = True
) -> bool:
    """Kiểm tra khẳng định (Assertion) rằng không có sự giao thoa dữ liệu giữa Train, Val và Test.
    Nguyên tắc FAIL-CLOSED:
    - Nếu thư mục Test không có ảnh hợp lệ -> Raise AssertionError ngay lập tức!
    - Nếu có trùng lặp SHA-256 hoặc gần trùng dHash (Hamming <= near_dup_threshold) -> Raise AssertionError!
    """
    if isinstance(train_dirs, (str, Path)):
        train_dirs = [train_dirs]
    if isinstance(val_dirs, (str, Path)):
        val_dirs = [val_dirs]
    if isinstance(test_dirs, (str, Path)):
        test_dirs = [test_dirs]

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
        print("=" * 70)
        print("KIỂM TRA CHỐNG RÒ RỈ DỮ LIỆU TỰ ĐỘNG (ZERO-LEAKAGE FAIL-CLOSED AUDIT)")
        print(f"-> Số ảnh Train tìm thấy: {len(train_files)}")
        print(f"-> Số ảnh Val tìm thấy:   {len(val_files)}")
        print(f"-> Số ảnh Test tìm thấy:  {len(test_files)}")

    # 1. FAIL-CLOSED Assertion: Test không được rỗng
    if len(train_files) == 0:
        raise AssertionError("LỖI CHÍ MẠNG (FAIL-CLOSED): Thư mục Train không chứa bất kỳ ảnh hợp lệ nào!")
    if len(test_files) == 0:
        raise AssertionError(
            "LỖI CHÍ MẠNG (FAIL-CLOSED): Thư mục Test không tồn tại hoặc không chứa bất kỳ ảnh hợp lệ nào! "
            "Để bảo đảm tính liêm chính khoa học và chống rò rỉ dữ liệu, quy trình bị DỪNG NGAY LẬP TỨC. "
            "Vui lòng tải hoặc kiểm tra thư mục test trước khi tiến hành."
        )

    # 2. Băm mật mã SHA-256
    train_sha: Dict[str, Path] = {}
    train_dhash: Dict[Path, int] = {}
    for p in train_files:
        h = compute_pixel_hash(p)
        train_sha[h] = p
        train_dhash[p] = compute_dhash(p)

    val_sha: Dict[str, Path] = {}
    val_dhash: Dict[Path, int] = {}
    for p in val_files:
        h = compute_pixel_hash(p)
        val_sha[h] = p
        val_dhash[p] = compute_dhash(p)

    test_sha: Dict[str, Path] = {}
    test_dhash: Dict[Path, int] = {}
    for p in test_files:
        h = compute_pixel_hash(p)
        test_sha[h] = p
        test_dhash[p] = compute_dhash(p)

    # A. Trùng lặp SHA-256 Train vs Test
    train_test_sha = set(train_sha.keys()) & set(test_sha.keys())
    if train_test_sha:
        msg = f"\n[CHÍ MẠNG] PHÁT HIỆN RÒ RỈ DỮ LIỆU CHÍNH XÁC (SHA-256) GIỮA TRAIN VÀ TEST ({len(train_test_sha)} ảnh)!\n"
        for h in list(train_test_sha)[:10]:
            msg += f" - Trùng lặp: Train [{train_sha[h]}] <==> Test [{test_sha[h]}]\n"
        raise AssertionError(msg)

    # B. Trùng lặp SHA-256 Val vs Test
    val_test_sha = set(val_sha.keys()) & set(test_sha.keys())
    if val_test_sha:
        msg = f"\n[CHÍ MẠNG] PHÁT HIỆN RÒ RỈ DỮ LIỆU CHÍNH XÁC (SHA-256) GIỮA VAL VÀ TEST ({len(val_test_sha)} ảnh)!\n"
        for h in list(val_test_sha)[:10]:
            msg += f" - Trùng lặp: Val [{val_sha[h]}] <==> Test [{test_sha[h]}]\n"
        raise AssertionError(msg)

    # C. Trùng lặp SHA-256 Train vs Val
    train_val_sha = set(train_sha.keys()) & set(val_sha.keys())
    if train_val_sha:
        msg = f"\n[LỖI NGHIÊM TRỌNG] PHÁT HIỆN TRÙNG LẶP GIỮA TRAIN VÀ VAL ({len(train_val_sha)} ảnh)!\n"
        for h in list(train_val_sha)[:10]:
            msg += f" - Trùng lặp: Train [{train_sha[h]}] <==> Val [{val_sha[h]}]\n"
        raise AssertionError(msg)

    # 3. Kiểm tra mã băm nhận thức dHash (Perceptual Hash & Near-Duplicates)
    near_dups_train_test = []
    for p_tr, h_tr in train_dhash.items():
        for p_te, h_te in test_dhash.items():
            dist = hamming_distance(h_tr, h_te)
            if dist <= near_dup_threshold:
                near_dups_train_test.append((p_tr, p_te, dist))

    near_dups_val_test = []
    for p_v, h_v in val_dhash.items():
        for p_te, h_te in test_dhash.items():
            dist = hamming_distance(h_v, h_te)
            if dist <= near_dup_threshold:
                near_dups_val_test.append((p_v, p_te, dist))

    if verbose:
        print(f"-> Kiểm tra dHash nhận thức (Ngưỡng Hamming <= {near_dup_threshold}/64 bits):")
        print(f"   + Cặp gần trùng Train vs Test: {len(near_dups_train_test)}")
        print(f"   + Cặp gần trùng Val vs Test:   {len(near_dups_val_test)}")

    if len(near_dups_train_test) > 0:
        msg = f"\n[CHÍ MẠNG] PHÁT HIỆN {len(near_dups_train_test)} ẢNH GẦN TRÙNG (NEAR-DUPLICATES) GIỮA TRAIN VÀ TEST!\n"
        for p_tr, p_te, d in near_dups_train_test[:10]:
            msg += f" - Gần trùng (Hamming dist={d}): Train [{p_tr.name}] vs Test [{p_te.name}]\n"
        raise AssertionError(msg)

    if len(near_dups_val_test) > 0:
        msg = f"\n[CHÍ MẠNG] PHÁT HIỆN {len(near_dups_val_test)} ẢNH GẦN TRÙNG (NEAR-DUPLICATES) GIỮA VAL VÀ TEST!\n"
        for p_v, p_te, d in near_dups_val_test[:10]:
            msg += f" - Gần trùng (Hamming dist={d}): Val [{p_v.name}] vs Test [{p_te.name}]\n"
        raise AssertionError(msg)

    if verbose:
        print("[XÁC NHẬN AN TOÀN TUYỆT ĐỐI]")
        print("1. Mã băm SHA-256: Giao giữa Train/Val và Test bằng RỖNG (0 trùng lặp).")
        print(f"2. Mã băm dHash (Hamming <= {near_dup_threshold}): 0 cặp gần trùng giữa Train/Val và Test.")
        print("Tập dữ liệu hoàn toàn độc lập, đủ điều kiện khoa học để huấn luyện.")
        print("=" * 70)

    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Kiểm tra chống rò rỉ dữ liệu")
    parser.add_argument('--train_dir', type=str, required=True, help="Đường dẫn thư mục train")
    parser.add_argument('--val_dir', type=str, default="", help="Đường dẫn thư mục val")
    parser.add_argument('--test_dir', type=str, required=True, help="Đường dẫn thư mục test")
    parser.add_argument('--threshold', type=int, default=4, help="Ngưỡng khoảng cách Hamming dHash")
    args = parser.parse_args()

    val_d = [args.val_dir] if args.val_dir else []
    try:
        verify_zero_leakage([args.train_dir], val_d, [args.test_dir], near_dup_threshold=args.threshold, verbose=True)
        print("KIỂM TRA THÀNH CÔNG: KHÔNG CÓ RÒ RỈ DỮ LIỆU.")
    except AssertionError as e:
        print(e)
        sys.exit(1)
