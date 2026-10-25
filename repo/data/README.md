# Data Directory Specification (Quy Chuẩn Dữ Liệu)

> [!IMPORTANT]
> Thư mục này **TUYỆT ĐỐI KHÔNG CHỨA DỮ LIỆU THÔ HOẶC ẢNH**. Mọi dữ liệu hình ảnh được lưu trữ trên Google Drive và copy vào đĩa tạm `/content` của Google Colab khi chạy thí nghiệm.

## 1. Cấu Trúc Thư Mục Dữ Liệu Dự Kiến Tại `/content/data`

```text
/content/data/
├── train/
│   ├── denoise/       # WED, BSD400, Waterloo Exploration (ảnh clean để sinh nhiễu online)
│   ├── derain/        # Train100L (cặp ảnh rainy / clean)
│   └── dehaze/        # RESIDE-OTS (Outdoor Training Set: cặp hazy / clear)
└── test/
    ├── denoise/
    │   ├── bsd68/     # 68 ảnh kiểm chuẩn Grayscale/RGB Denoising (sigma 15, 25, 50)
    │   ├── urban100/  # 100 ảnh kiến trúc chi tiết cao
    │   └── kodak24/   # 24 ảnh tự nhiên chuẩn
    ├── derain/
    │   └── rain100l/  # 100 ảnh vệt mưa nhẹ kiểm chuẩn
    └── dehaze/
        └── sots/      # SOTS-Outdoor / SOTS-Indoor (500 ảnh kiểm chuẩn sương mù)
```

## 2. Nguồn Tải Dữ Liệu Chuẩn & Tham Chiếu Gốc
- **Denoising (BSD68 / Urban100 / Kodak24)**:
  - BSD68: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
  - Kodak24: http://r0k.us/graphics/kodak/
  - Urban100: https://github.com/jbhuang0604/SelfExSR
- **Deraining (Rain100L)**:
  - Rain100L: https://github.com/hongguangdi/Rain100L
- **Dehazing (RESIDE - SOTS)**:
  - RESIDE Benchmark: https://sites.google.com/view/reside-dehaze-datasets/reside-v0

## 3. Quy Cách Đóng Gói Phục Vụ Google Colab Free
- Toàn bộ dữ liệu sau khi chuẩn bị ở task P0-T3 sẽ được đóng gói thành một file lưu trữ duy nhất trên Google Drive:
  `MyDrive/nckh_coChau_data/aioir_data.tar.gz`
- Colab notebook sẽ giải nén file này vào `/content/data` trong vòng chưa đầy 1 phút, tránh nghẽn I/O qua Google Drive mount.
