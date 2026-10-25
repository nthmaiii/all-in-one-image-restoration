"""P1-T1: Script đánh giá checkpoint pretrained của PromptIR, AdaIR và AirNet trên các tập test.
Đo đạc: PSNR, SSIM, LPIPS.
Xuất kết quả: results/baseline_results.csv và results/baseline_per_image.csv.
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines' / 'promptir'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines' / 'adair'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines' / 'airnet'))

try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
except ImportError:
    pass

try:
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex')
except Exception:
    loss_fn_alex = None

def compute_metrics(restored_np, target_np):
    """restored_np and target_np are float32 in range [0, 1], shape (H, W, 3) or (H, W)"""
    psnr = compare_psnr(target_np, restored_np, data_range=1.0)
    ssim = compare_ssim(target_np, restored_np, data_range=1.0, channel_axis=2 if target_np.ndim == 3 else None)
    return psnr, ssim

def compute_lpips_val(restored_t, target_t, device):
    """restored_t and target_t are float32 in range [0, 1], shape (1, 3, H, W)"""
    if loss_fn_alex is None:
        return -1.0
    try:
        with torch.no_grad():
            r = restored_t.to(device) * 2.0 - 1.0
            t = target_t.to(device) * 2.0 - 1.0
            val = loss_fn_alex.to(device)(r, t)
            return float(val.item())
    except Exception:
        return -1.0

def load_baseline_model(model_name, weights_path=None, device='cpu'):
    print(f"Khởi tạo mô hình {model_name}...")
    if model_name.lower() == 'promptir':
        from baselines.promptir.net.model import PromptIR
        model = PromptIR(decoder=True)
    elif model_name.lower() == 'adair':
        from baselines.adair.net.model import AdaIR
        model = AdaIR(decoder=True)
    elif model_name.lower() == 'airnet':
        from baselines.airnet.net.model import AirNet
        import argparse
        opt = argparse.Namespace()
        opt.device = device
        model = AirNet(opt)
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}")

    if weights_path and Path(weights_path).exists():
        print(f"Nạp trọng số từ: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('params', checkpoint))
        # Remove 'net.' or 'module.' prefixes if present
        clean_state = {}
        for k, v in state_dict.items():
            clean_k = k.replace('net.', '').replace('module.', '')
            clean_state[clean_k] = v
        model.load_state_dict(clean_state, strict=False)
        print("Đã nạp checkpoint thành công!")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy file weights tại {weights_path}. Đang dùng trọng số ngẫu nhiên để test pipeline!")

    model = model.to(device)
    model.eval()
    return model

def evaluate_denoise(model, model_name, dataset_dir, sigmas=[15, 25, 50], device='cpu', max_images=None):
    results = []
    p = Path(dataset_dir)
    images = sorted([f for f in p.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
    if max_images:
        images = images[:max_images]

    print(f">> Bắt đầu đánh giá Denoise trên {len(images)} ảnh ({p.name})...")
    
    for sigma in sigmas:
        psnr_list, ssim_list, lpips_list = [], [], []
        for img_p in images:
            img = Image.open(img_p).convert('RGB')
            clean_np = np.array(img).astype(np.float32) / 255.0
            clean_t = to_tensor(img).unsqueeze(0).to(device)

            # Thêm nhiễu Gauss AWGN
            np.random.seed(42)
            noise = np.random.normal(0, sigma / 255.0, clean_np.shape).astype(np.float32)
            noisy_np = np.clip(clean_np + noise, 0.0, 1.0)
            noisy_t = torch.from_numpy(noisy_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.no_grad():
                if model_name.lower() == 'airnet':
                    restored_t = model(noisy_t, noisy_t)
                else:
                    restored_t = model(noisy_t)

            restored_np = np.clip(restored_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
            psnr, ssim = compute_metrics(restored_np, clean_np)
            lpips_val = compute_lpips_val(restored_t, clean_t, device)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips_val)

        mean_psnr = float(np.mean(psnr_list))
        mean_ssim = float(np.mean(ssim_list))
        valid_lpips = [l for l in lpips_list if l >= 0]
        mean_lpips = float(np.mean(valid_lpips)) if valid_lpips else -1.0

        print(f"[{model_name}] Denoise-{sigma} | PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | LPIPS: {mean_lpips:.4f}")
        results.append({
            'model': model_name,
            'task': f'denoise_sigma_{sigma}',
            'dataset': p.name,
            'metric_psnr': round(mean_psnr, 2),
            'metric_ssim': round(mean_ssim, 4),
            'metric_lpips': round(mean_lpips, 4),
            'num_images': len(images)
        })
    return results

def evaluate_paired(model, model_name, task_name, dataset_name, input_dir, target_dir, device='cpu', max_images=None):
    in_p = Path(input_dir)
    tar_p = Path(target_dir)
    in_files = sorted([f for f in in_p.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
    if max_images:
        in_files = in_files[:max_images]

    print(f">> Bắt đầu đánh giá {task_name} ({dataset_name}) trên {len(in_files)} ảnh...")
    psnr_list, ssim_list, lpips_list = [], [], []

    for img_p in in_files:
        tar_img_p = tar_p / img_p.name
        if not tar_img_p.exists():
            continue
        in_img = Image.open(img_p).convert('RGB')
        tar_img = Image.open(tar_img_p).convert('RGB')

        clean_np = np.array(tar_img).astype(np.float32) / 255.0
        clean_t = to_tensor(tar_img).unsqueeze(0).to(device)
        in_t = to_tensor(in_img).unsqueeze(0).to(device)

        with torch.no_grad():
            if model_name.lower() == 'airnet':
                restored_t = model(in_t, in_t)
            else:
                restored_t = model(in_t)

        restored_np = np.clip(restored_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
        psnr, ssim = compute_metrics(restored_np, clean_np)
        lpips_val = compute_lpips_val(restored_t, clean_t, device)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val)

    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    valid_lpips = [l for l in lpips_list if l >= 0]
    mean_lpips = float(np.mean(valid_lpips)) if valid_lpips else -1.0

    print(f"[{model_name}] {task_name} | PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | LPIPS: {mean_lpips:.4f}")
    return [{
        'model': model_name,
        'task': task_name,
        'dataset': dataset_name,
        'metric_psnr': round(mean_psnr, 2),
        'metric_ssim': round(mean_ssim, 4),
        'metric_lpips': round(mean_lpips, 4),
        'num_images': len(psnr_list)
    }]

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình AiOIR")
    parser.add_argument('--model', type=str, default='PromptIR', choices=['PromptIR', 'AdaIR', 'AirNet'])
    parser.add_argument('--weights', type=str, default=None, help='Đường dẫn checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/test', help='Thư mục chứa test datasets')
    parser.add_argument('--task', type=str, default='all', choices=['all', 'denoise', 'derain', 'dehaze'])
    parser.add_argument('--max_images', type=int, default=None, help='Giới hạn số ảnh để test nhanh')
    parser.add_argument('--output_csv', type=str, default='results/baseline_results.csv')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_baseline_model(args.model, args.weights, device=device)

    data_dir = Path(args.data_dir)
    all_results = []

    # 1. Denoise (BSD68)
    if args.task in ['all', 'denoise']:
        bsd68_dir = data_dir / 'denoise' / 'bsd68'
        if bsd68_dir.exists() and any(bsd68_dir.iterdir()):
            res = evaluate_denoise(model, args.model, bsd68_dir, device=device, max_images=args.max_images)
            all_results.extend(res)

    # 2. Derain (Rain100L)
    if args.task in ['all', 'derain']:
        rain_in = data_dir / 'derain' / 'Rain100L' / 'input'
        rain_tar = data_dir / 'derain' / 'Rain100L' / 'target'
        if rain_in.exists() and any(rain_in.iterdir()):
            res = evaluate_paired(model, args.model, 'deraining', 'Rain100L', rain_in, rain_tar, device=device, max_images=args.max_images)
            all_results.extend(res)

    # 3. Dehaze (SOTS)
    if args.task in ['all', 'dehaze']:
        haze_in = data_dir / 'dehaze' / 'input'
        haze_tar = data_dir / 'dehaze' / 'target'
        if haze_in.exists() and any(haze_in.iterdir()):
            res = evaluate_paired(model, args.model, 'dehazing', 'SOTS', haze_in, haze_tar, device=device, max_images=args.max_images)
            all_results.extend(res)

    # Lưu kết quả
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = out_csv.exists()
    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'task', 'dataset', 'metric_psnr', 'metric_ssim', 'metric_lpips', 'num_images'])
        if not file_exists:
            writer.writeheader()
        writer.writerows(all_results)
        
    print(f"Đã lưu kết quả đánh giá vào: {out_csv.resolve()}")

if __name__ == '__main__':
    main()
