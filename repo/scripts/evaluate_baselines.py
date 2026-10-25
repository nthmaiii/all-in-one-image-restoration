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
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex')
except Exception:
    loss_fn_alex = None

def forward_padded(model, x, factor=64):
    """Tự động pad ảnh thành bội số của 64 để tương thích với pixel_unshuffle (Restormer/PromptIR),
    sau đó unpad về kích thước gốc (H, W).
    """
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
    else:
        x_pad = x
    out = model(x_pad)
    return out[:, :, :h, :w]

def resolve_file_path(p_str, search_name=None):
    if not p_str and not search_name:
        return None
    candidates = []
    if p_str:
        p = Path(p_str)
        candidates.append(p)
        if str(p).startswith('repo/') or str(p).startswith('repo\\'):
            candidates.append(Path(str(p)[5:]))
        candidates.append(Path('repo') / p)
        candidates.append(Path('..') / p)
    
    for c in candidates:
        if c.exists():
            return c.resolve()

    if search_name:
        for root in [Path('.'), Path('..'), Path('/content'), Path('/content/repo')]:
            if root.exists():
                found = list(root.rglob(search_name))
                if found:
                    return found[0].resolve()
    return candidates[0] if candidates else None

def compute_metrics(restored_np, target_np):
    """restored_np and target_np are float32 in range [0, 1], shape (H, W, 3) or (H, W)"""
    try:
        from skimage.metrics import peak_signal_noise_ratio as compare_psnr
        from skimage.metrics import structural_similarity as compare_ssim
        psnr = compare_psnr(target_np, restored_np, data_range=1.0)
        ssim = compare_ssim(target_np, restored_np, data_range=1.0, channel_axis=2 if target_np.ndim == 3 else None)
        return float(psnr), float(ssim)
    except Exception:
        mse = np.mean((target_np.astype(np.float64) - restored_np.astype(np.float64)) ** 2)
        psnr = float('inf') if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        mu1 = np.mean(target_np)
        mu2 = np.mean(restored_np)
        sigma1 = np.var(target_np)
        sigma2 = np.var(restored_np)
        sigma12 = np.mean((target_np - mu1) * (restored_np - mu2))
        ssim = float(((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)))
        return psnr, ssim

def compute_lpips_val(restored_t, target_t, device):
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
    m_name = model_name.strip().lower()
    clean_model_name = 'PromptIR'
    if 'ada' in m_name:
        clean_model_name = 'AdaIR'
    elif 'air' in m_name:
        clean_model_name = 'AirNet'

    print(f"Khởi tạo mô hình {clean_model_name}...")
    if clean_model_name == 'PromptIR':
        from baselines.promptir.net.model import PromptIR
        model = PromptIR(decoder=True)
    elif clean_model_name == 'AdaIR':
        from baselines.adair.net.model import AdaIR
        model = AdaIR(decoder=True)
    elif clean_model_name == 'AirNet':
        from baselines.airnet.net.model import AirNet
        import argparse
        opt = argparse.Namespace()
        opt.device = device
        model = AirNet(opt)
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}")

    resolved_weights = resolve_file_path(weights_path, search_name=Path(weights_path).name if weights_path else None)
    if resolved_weights and resolved_weights.exists():
        print(f"Nạp trọng số từ: {resolved_weights}")
        checkpoint = torch.load(str(resolved_weights), map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint.get('params', checkpoint))
        clean_state = {}
        for k, v in state_dict.items():
            clean_k = k.replace('net.', '').replace('module.', '')
            clean_state[clean_k] = v
        model.load_state_dict(clean_state, strict=False)
        print("Đã nạp checkpoint thành công!")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy file weights tại {weights_path} (hoặc {resolved_weights}). Dùng trọng số ngẫu nhiên!")

    model = model.to(device)
    model.eval()
    return model, clean_model_name

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
            clean_t = to_tensor(img).unsqueeze(0)
            
            torch.manual_seed(42)
            noise = torch.randn_like(clean_t) * (sigma / 255.0)
            noisy_t = torch.clamp(clean_t + noise, 0.0, 1.0)

            with torch.no_grad():
                restored_t = forward_padded(model, noisy_t.to(device), factor=64)
                restored_t = torch.clamp(restored_t, 0.0, 1.0).cpu()

            restored_np = (restored_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round() / 255.0
            clean_np = (clean_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round() / 255.0

            psnr, ssim = compute_metrics(restored_np, clean_np)
            lp = compute_lpips_val(restored_t, clean_t, device)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            if lp >= 0:
                lpips_list.append(lp)

        mean_psnr = float(np.mean(psnr_list))
        mean_ssim = float(np.mean(ssim_list))
        mean_lpips = float(np.mean(lpips_list)) if lpips_list else -1.0

        print(f"[{model_name}] Denoise ({p.name}) sigma={sigma} | PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | LPIPS: {mean_lpips:.4f}")
        results.append({
            'model': model_name,
            'task': f'denoising_sigma_{sigma}',
            'dataset': p.name,
            'metric_psnr': round(mean_psnr, 2),
            'metric_ssim': round(mean_ssim, 4),
            'metric_lpips': round(mean_lpips, 4),
            'num_images': len(images)
        })
    return results

def evaluate_paired(model, model_name, task_name, dataset_name, input_dir, target_dir, device='cpu', max_images=None, use_y_channel=True):
    results = []
    in_p = Path(input_dir)
    tar_p = Path(target_dir)

    in_images = sorted([f for f in in_p.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
    if max_images:
        in_images = in_images[:max_images]

    print(f">> Bắt đầu đánh giá {task_name} ({dataset_name}) trên {len(in_images)} cặp ảnh (Y-channel={use_y_channel})...")
    psnr_list, ssim_list, lpips_list = [], [], []

    for img_file in in_images:
        tar_file = tar_p / img_file.name
        if not tar_file.exists():
            clean_stem = img_file.name.lower().replace('rain-', '').replace('norain-', '')
            matches = [f for f in tar_p.glob('*') if clean_stem in f.name.lower()]
            if matches:
                tar_file = matches[0]
            else:
                continue

        in_img = Image.open(img_file).convert('RGB')
        tar_img = Image.open(tar_file).convert('RGB')

        inp_t = to_tensor(in_img).unsqueeze(0)
        tar_t = to_tensor(tar_img).unsqueeze(0)

        with torch.no_grad():
            restored_t = forward_padded(model, inp_t.to(device), factor=64)
            restored_t = torch.clamp(restored_t, 0.0, 1.0).cpu()

        restored_np = (restored_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round() / 255.0
        tar_np = (tar_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round() / 255.0

        if use_y_channel:
            r_y = 0.257 * restored_np[:, :, 0] + 0.504 * restored_np[:, :, 1] + 0.098 * restored_np[:, :, 2] + 16.0 / 255.0
            t_y = 0.257 * tar_np[:, :, 0] + 0.504 * tar_np[:, :, 1] + 0.098 * tar_np[:, :, 2] + 16.0 / 255.0
            psnr, ssim = compute_metrics(r_y, t_y)
        else:
            psnr, ssim = compute_metrics(restored_np, tar_np)

        lp = compute_lpips_val(restored_t, tar_t, device)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        if lp >= 0:
            lpips_list.append(lp)

    if not psnr_list:
        print(f"[-] Không có ảnh hợp lệ nào trong {dataset_name}!")
        return []

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    valid_lpips = [x for x in lpips_list if x >= 0]
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
    parser.add_argument('--baseline', '--model', dest='model', type=str, default='PromptIR', help='Tên mô hình baseline (PromptIR, AdaIR, AirNet)')
    parser.add_argument('--weights', type=str, default=None, help='Đường dẫn checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/test', help='Thư mục chứa test datasets')
    parser.add_argument('--task', type=str, default='all', choices=['all', 'denoise', 'derain', 'dehaze'])
    parser.add_argument('--max_images', type=int, default=None, help='Giới hạn số ảnh để test nhanh')
    parser.add_argument('--output_csv', type=str, default='results/baseline_results.csv')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, clean_name = load_baseline_model(args.model, args.weights, device=device)

    data_dir = resolve_file_path(args.data_dir)
    if not data_dir or not data_dir.exists():
        for candidate in [Path('repo/data/test'), Path('data/test'), Path('/content/repo/data/test')]:
            if candidate.exists():
                data_dir = candidate
                break

    print(f"Sử dụng thư mục dữ liệu tại: {data_dir}")
    all_results = []

    # 1. Denoise (BSD68, Kodak24, Urban100)
    if args.task in ['all', 'denoise'] and data_dir and data_dir.exists():
        for sub_name in ['bsd68', 'kodak24', 'urban100']:
            sub_dir = data_dir / 'denoise' / sub_name
            if sub_dir.exists() and any(sub_dir.iterdir()):
                res = evaluate_denoise(model, clean_name, sub_dir, device=device, max_images=args.max_images)
                all_results.extend(res)

    # 2. Derain (Rain100L - Y-channel evaluation)
    if args.task in ['all', 'derain'] and data_dir and data_dir.exists():
        rain_in = data_dir / 'derain' / 'Rain100L' / 'input'
        rain_tar = data_dir / 'derain' / 'Rain100L' / 'target'
        if rain_in.exists() and any(rain_in.iterdir()):
            res = evaluate_paired(model, clean_name, 'deraining', 'Rain100L', rain_in, rain_tar, device=device, max_images=args.max_images, use_y_channel=True)
            all_results.extend(res)

    # 3. Dehaze (SOTS nếu có)
    if args.task in ['all', 'dehaze'] and data_dir and data_dir.exists():
        haze_in = data_dir / 'dehaze' / 'input'
        haze_tar = data_dir / 'dehaze' / 'target'
        if haze_in.exists() and any(haze_in.iterdir()):
            res = evaluate_paired(model, clean_name, 'dehazing', 'SOTS', haze_in, haze_tar, device=device, max_images=args.max_images, use_y_channel=False)
            all_results.extend(res)

    # Lưu kết quả vào cả 2 vị trí
    target_csvs = [Path(args.output_csv)]
    if str(args.output_csv).startswith('repo/'):
        target_csvs.append(Path(str(args.output_csv)[5:]))
    else:
        target_csvs.append(Path('repo') / args.output_csv)

    for out_csv in target_csvs:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = out_csv.exists()
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'task', 'dataset', 'metric_psnr', 'metric_ssim', 'metric_lpips', 'num_images'])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Đã lưu kết quả đánh giá vào: {out_csv.resolve()}")

if __name__ == '__main__':
    main()
