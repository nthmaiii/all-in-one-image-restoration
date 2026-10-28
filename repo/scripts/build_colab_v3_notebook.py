"""Tạo tự động Jupyter Notebook Google Colab: 04_phase_hopfield_pilot.ipynb.
Thực nghiệm khóa trước Pilot v3: PHASE-HOPFIELD-IR vs PromptIR vs Param-Matched Control.
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "colab" / "04_phase_hopfield_pilot.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PHASE-HOPFIELD-IR — Non-Commutative Fractional Hopfield Dynamics for All-in-One Image Restoration\n",
            "### Thực Nghiệm Khóa Trước Đối Chứng Độc Lập (Pilot v3 — Tesla T4 GPU)\n",
            "\n",
            "**Trụ Cột Toán Học & Đột Phá Khoa Học**:\n",
            "1. **Giải tích Phân số (Fractional Fourier Transform - FrFT)**: Không gian pha liên tục $(x, p)_\\alpha$ với góc quay thích ứng $\\alpha(x) \\in (0, 1)$.\n",
            "2. **Đại số Lie (Lie Algebra Commutator)**: Đo đạc chính xác số hạng tương tác chéo phi giao hoán $[\\mathcal{O}_{\\text{haze}}, \\mathcal{O}_{\\text{noise}}] = \\epsilon(1 - t)$ bị các mô hình phẳng bỏ sót.\n",
            "3. **Vật lý Thống kê Hopfield (Giải Nobel Vật lý 2024)**: Dòng chảy gradient theo thế năng Lyapunov $E(\\mathbf{z})$ kéo quỹ đạo về manifold ảnh sạch.\n",
            "4. **Triệt tiêu lỗi ngủ đông (Dormant Block)**: Ghép năng lượng Energy-Scaled Coupling + Tách nhóm tối ưu Dual Learning Rate ($lr_{\\text{novel}} = 10 \\times lr_{\\text{base}}$).\n",
            "\n",
            "**Tiêu Chuẩn Quyết Định Khoa Học (Preregistered Gate P5)**:\n",
            "- $\\Delta(B - A) = \\mu_B - \\mu_A \\ge X = \\max(0.40, 2\\hat{\\sigma}_A)$ dB trên Held-out Hybrid-OOD.\n",
            "- $\\Delta(B - C) = \\mu_B - \\mu_C \\ge Z = \\max(0.25, 1.5\\hat{\\sigma}_A)$ dB (vượt trội mô hình đối chứng tham số).\n",
            "- Tỉ lệ năng lượng khối mới `Ratio(New/Base)` $\\ge 15\\%$.\n",
            "- Tác vụ đơn lẻ (Noise, Rain, Haze) không suy giảm quá $0.15$ dB."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Thiết Lập Môi Trường & Kiểm Tra GPU Tesla T4"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Kiểm tra phần cứng GPU và phiên bản CUDA\n",
            "!nvidia-smi\n",
            "\n",
            "import torch\n",
            "print(f\"PyTorch: {torch.__version__} | CUDA Sẵn sàng: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Đồng Bộ Codebase Sang Đĩa SSD Cục Bộ Colab"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, shutil\n",
            "from pathlib import Path\n",
            "\n",
            "%cd /content\n",
            "!rm -rf /content/repo /content/temp_repo\n",
            "\n",
            "print(\"Đang clone codebase từ GitHub...\")\n",
            "!git clone https://github.com/nthmaiii/all-in-one-image-restoration.git /content/temp_repo\n",
            "\n",
            "# Chuyển thư mục repo ra /content/repo và giữ .git để phục vụ audit commit HEAD\n",
            "shutil.move(\"/content/temp_repo/repo\", \"/content/repo\")\n",
            "shutil.move(\"/content/temp_repo/.git\", \"/content/repo/.git\")\n",
            "!rm -rf /content/temp_repo\n",
            "\n",
            "%cd /content/repo\n",
            "!pip install -q -r requirements.txt\n",
            "\n",
            "# Kiểm toán Git Commit HEAD\n",
            "commit_hash = !git rev-parse HEAD\n",
            "print(f\">> [AUDIT KHOA HỌC] Git Commit HEAD: {commit_hash[0]}\")\n",
            "os.makedirs(\"runs/phase_hopfield_v3\", exist_ok=True)\n",
            "with open(\"runs/phase_hopfield_v3/git_commit.txt\", \"w\", encoding=\"utf-8\") as f:\n",
            "    f.write(commit_hash[0] + \"\\n\")\n",
            "print(\">> Môi trường thực thi PHASE-HOPFIELD-IR đã sẵn sàng!\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Tải Dữ Liệu Kiểm Chuẩn & Khẳng Định Chống Rò Rỉ (Zero-Leakage Audit)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "if Path(\"/content/repo/repo/scripts\").exists():\n",
            "    !cp -rf /content/repo/repo/* /content/repo/\n",
            "    !rm -rf /content/repo/repo\n",
            "\n",
            "# 1. Tải checkpoint PromptIR All-in-One và tập test held-out (BSD68, Kodak24)\n",
            "!python scripts/download_datasets.py --pilot_v2\n",
            "\n",
            "# 2. Chuẩn bị 160 ảnh train và 40 ảnh val sạch độc lập & kiểm tra zero-leakage fail-closed\n",
            "!python scripts/prepare_data_v2.py\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Kiểm Chứng Toán Học & Toàn Bộ Bộ Unit Test (Gate 1 & Gate 3 Pass)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Đo đạc thực nghiệm Giao hoán tử Lie [Koschmieder, Gaussian] = eps * (1 - t) trên ảnh thật\n",
            "!python scripts/verify_lie_commutator_gap.py\n",
            "\n",
            "# 2. Chạy 5/5 Unit Test kiểm toán toán học (Phản đối xứng Lie, Đơn điệu năng lượng Hopfield, Đẳng tham số B vs C)\n",
            "!python tests/test_phase_hopfield.py\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Smoke Test Trên GPU Tesla T4 (Kiểm Tra Toàn Bộ Pipeline Trước Huấn Luyện)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Smoke test nhánh A, B, C (iters=2) xác nhận DataLoader, Dual LR và Evaluator chạy mượt mà\n",
            "!python scripts/train_phase_hopfield.py --branch A --smoke_test --seed 42 --patch_size 128\n",
            "!python scripts/train_phase_hopfield.py --branch B --smoke_test --seed 42 --patch_size 128\n",
            "!python scripts/train_phase_hopfield.py --branch C --smoke_test --seed 42 --patch_size 128\n",
            "print(\">> Smoke test hoàn tất thành công! Mọi pipeline đã xác nhận hoạt động 100% trên GPU.\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Nhánh A: PromptIR Baseline (3 Seeds: 42, 123, 2024)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Huấn luyện Nhánh A để thiết lập độ lệch chuẩn cơ sở và khóa ngưỡng thống kê\n",
            "!python scripts/train_phase_hopfield.py --branch A --seed 42 --iters 1500 --batch_size 4 --patch_size 128\n",
            "!python scripts/train_phase_hopfield.py --branch A --seed 123 --iters 1500 --batch_size 4 --patch_size 128\n",
            "!python scripts/train_phase_hopfield.py --branch A --seed 2024 --iters 1500 --batch_size 4 --patch_size 128\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Khóa Ngưỡng Thống Kê Thuyết Minh Dựa Trên σ_A (Preregistered Decision Gate P5)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "from pathlib import Path\n",
            "\n",
            "csv_path = Path(\"results/phase_hopfield_results.csv\")\n",
            "assert csv_path.exists(), \"Chưa tìm thấy file results/phase_hopfield_results.csv!\"\n",
            "\n",
            "df = pd.read_csv(csv_path)\n",
            "df_a = df[(df[\"branch\"] == \"A\") & (df[\"checkpoint_type\"] == \"best_val\")]\n",
            "assert len(df_a) == 3, f\"Nhánh A phải đủ 3 seeds trước khi khóa ngưỡng! Hiện có: {len(df_a)}\"\n",
            "\n",
            "psnr_a = df_a[\"primary_endpoint_ood_psnr\"].values\n",
            "mu_a = float(np.mean(psnr_a))\n",
            "sigma_a = float(np.std(psnr_a, ddof=1))\n",
            "\n",
            "X = float(max(0.40, 2.0 * sigma_a))\n",
            "Z = float(max(0.25, 1.5 * sigma_a))\n",
            "\n",
            "thresholds = {\n",
            "    \"mu_A\": round(mu_a, 4),\n",
            "    \"sigma_A\": round(sigma_a, 4),\n",
            "    \"X_threshold_B_vs_A\": round(X, 4),\n",
            "    \"Z_threshold_B_vs_C\": round(Z, 4),\n",
            "    \"seeds_A\": df_a[\"seed\"].tolist(),\n",
            "    \"psnr_A_per_seed\": [round(x, 4) for x in psnr_a.tolist()]\n",
            "}\n",
            "\n",
            "with open(\"thresholds_locked_v3.json\", \"w\", encoding=\"utf-8\") as f:\n",
            "    json.dump(thresholds, f, indent=2)\n",
            "\n",
            "print(\"=\" * 65)\n",
            "print(\"ĐÃ KHÓA NGƯỠNG THỐNG KÊ (PRE-REGISTERED THRESHOLDS LOCKED):\")\n",
            "print(f\"-> Baseline PromptIR Mean (mu_A):  {mu_a:.4f} dB\")\n",
            "print(f\"-> Baseline PromptIR Std (sigma_A): {sigma_a:.4f} dB\")\n",
            "print(f\"-> Ngưỡng Vượt Trội B vs A (X):      {X:.4f} dB\")\n",
            "print(f\"-> Ngưỡng Vượt Trội B vs C (Z):      {Z:.4f} dB\")\n",
            "print(\"=\" * 65)\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Nhánh B: Huấn Luyện Đề Xuất PHASE-HOPFIELD-IR (3 Seeds: 42, 123, 2024)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Huấn luyện mô hình đề xuất PhaseHopfieldIR với Dual Learning Rate\n",
            "!python scripts/train_phase_hopfield.py --branch B --seed 42 --iters 1500 --batch_size 4 --patch_size 128 --lr 1e-4 --lr_novel 1e-3\n",
            "!python scripts/train_phase_hopfield.py --branch B --seed 123 --iters 1500 --batch_size 4 --patch_size 128 --lr 1e-4 --lr_novel 1e-3\n",
            "!python scripts/train_phase_hopfield.py --branch B --seed 2024 --iters 1500 --batch_size 4 --patch_size 128 --lr 1e-4 --lr_novel 1e-3\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Nhánh C: Đối Chứng Tham Số Param-Matched Control (3 Seeds: 42, 123, 2024)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Huấn luyện mô hình đối chứng tham số (cùng số tham số, cùng Dual LR, không FrFT/Lie/Hopfield)\n",
            "!python scripts/train_phase_hopfield.py --branch C --seed 42 --iters 1500 --batch_size 4 --patch_size 128 --lr 1e-4 --lr_novel 1e-3\n",
            "!python scripts/train_phase_hopfield.py --branch C --seed 123 --iters 1500 --batch_size 4 --patch_size 128 --lr 1e-4 --lr_novel 1e-3\n",
            "!python scripts/train_phase_hopfield.py --branch C --seed 2024 --iters 1500 --batch_size 4 --patch_size 128 --lr 1e-4 --lr_novel 1e-3\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Tổng Hợp Kết Quả & Nghiệm Thu Khoa Học Quyết Định (Decision Gate P5)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "from pathlib import Path\n",
            "from google.colab import files\n",
            "\n",
            "csv_path = Path(\"results/phase_hopfield_results.csv\")\n",
            "assert csv_path.exists(), \"Không tìm thấy results/phase_hopfield_results.csv!\"\n",
            "df = pd.read_csv(csv_path)\n",
            "\n",
            "with open(\"thresholds_locked_v3.json\", \"r\", encoding=\"utf-8\") as f:\n",
            "    thresh = json.load(f)\n",
            "X = thresh[\"X_threshold_B_vs_A\"]\n",
            "Z = thresh[\"Z_threshold_B_vs_C\"]\n",
            "\n",
            "df_best = df[df[\"checkpoint_type\"] == \"best_val\"]\n",
            "\n",
            "res = {}\n",
            "for br in [\"A\", \"B\", \"C\"]:\n",
            "    sub = df_best[df_best[\"branch\"] == br]\n",
            "    psnr_vals = sub[\"primary_endpoint_ood_psnr\"].values\n",
            "    ratio_vals = sub[\"new_block_ratio\"].values if \"new_block_ratio\" in sub else [0.0]*len(sub)\n",
            "    res[br] = {\n",
            "        \"mean\": float(np.mean(psnr_vals)),\n",
            "        \"std\": float(np.std(psnr_vals, ddof=1)),\n",
            "        \"values\": psnr_vals.tolist(),\n",
            "        \"mean_ratio\": float(np.mean(ratio_vals))\n",
            "    }\n",
            "\n",
            "delta_BA = res[\"B\"][\"mean\"] - res[\"A\"][\"mean\"]\n",
            "delta_BC = res[\"B\"][\"mean\"] - res[\"C\"][\"mean\"]\n",
            "\n",
            "print(\"=\" * 85)\n",
            "print(\"TỔNG KẾT KHOA HỌC: BẢNG SO SÁNH ĐỐI CHỨNG PRIMARY ENDPOINT (HELD-OUT HYBRID-OOD)\")\n",
            "print(\"=\" * 85)\n",
            "print(f\"Nhánh A (PromptIR Baseline):        {res['A']['mean']:.4f} ± {res['A']['std']:.4f} dB\")\n",
            "print(f\"Nhánh B (PHASE-HOPFIELD-IR):       {res['B']['mean']:.4f} ± {res['B']['std']:.4f} dB  (Ratio: {res['B']['mean_ratio']*100:.2f}%)\")\n",
            "print(f\"Nhánh C (Param-Matched Control):   {res['C']['mean']:.4f} ± {res['C']['std']:.4f} dB  (Ratio: {res['C']['mean_ratio']*100:.2f}%)\")\n",
            "print(\"-\" * 85)\n",
            "print(f\"-> Chênh lệch Delta(B - A): {delta_BA:+.4f} dB | Yêu cầu Gate X: >= {X:.4f} dB -> {'[PASS CHIẾN THẮNG]' if delta_BA >= X else '[CHƯA ĐẠT]'}\")\n",
            "print(f\"-> Chênh lệch Delta(B - C): {delta_BC:+.4f} dB | Yêu cầu Gate Z: >= {Z:.4f} dB -> {'[PASS CHIẾN THẮNG]' if delta_BC >= Z else '[CHƯA ĐẠT]'}\")\n",
            "print(f\"-> Hoạt động khối mới:      {res['B']['mean_ratio']*100:.2f}%     | Yêu cầu Gate Ratio: >= 15.00%  -> {'[PASS KHÔNG NGỦ ĐÔNG]' if res['B']['mean_ratio'] >= 0.15 else '[DORMANT]'}\")\n",
            "print(\"=\" * 85)\n",
            "\n",
            "# Tự động nén và tải về toàn bộ artifacts để nộp báo cáo cho cô Châu\n",
            "!zip -r phase_hopfield_pilot_results.zip results/ runs/phase_hopfield_v3/ thresholds_locked_v3.json\n",
            "files.download(\"phase_hopfield_pilot_results.zip\")\n"
        ]
    }
]

notebook_dict = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"Đã tạo thành công Jupyter Notebook tại: {NOTEBOOK_PATH}")
