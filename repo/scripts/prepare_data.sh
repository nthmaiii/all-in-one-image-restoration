#!/bin/bash
# ==============================================================================
# Script chuẩn bị dữ liệu AiOIR trên Google Colab / Linux
# Sử dụng: bash scripts/prepare_data.sh [DATA_DIR]
# ==============================================================================

set -e

DATA_DIR=${1:-"data/test"}
echo "=== CHUẨN BỊ DỮ LIỆU TẠI: $DATA_DIR ==="

mkdir -p "$DATA_DIR/denoise/bsd68"
mkdir -p "$DATA_DIR/denoise/kodak24"
mkdir -p "$DATA_DIR/denoise/urban100"
mkdir -p "$DATA_DIR/derain/Rain100L/input"
mkdir -p "$DATA_DIR/derain/Rain100L/target"
mkdir -p "$DATA_DIR/dehaze/input"
mkdir -p "$DATA_DIR/dehaze/target"

# Chạy python script để tải và giải nén các tập test
python scripts/download_datasets.py "$DATA_DIR"

# Kiểm tra số lượng ảnh
python scripts/verify_data_counts.py "$DATA_DIR" || echo "Lưu ý: Cần đảm bảo Rain100L và SOTS đã được đồng bộ từ Google Drive."

echo "=== HOÀN TẤT CHUẨN BỊ DỮ LIỆU ==="
