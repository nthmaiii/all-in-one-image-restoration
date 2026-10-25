"""P1-T2: Đo tham số (Params), FLOPs, độ trễ (latency), và bộ nhớ đỉnh (Peak VRAM).
Tối ưu hóa chạy tức thì trên cả CPU (local test) và GPU T4 (Colab benchmarking).
"""

import sys
import time
import csv
from pathlib import Path
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines' / 'promptir'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines' / 'adair'))

def profile_model(model_name, model, input_size=(3, 128, 128), num_runs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 1. Parameter count
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 2. FLOPs calculation
    gflops = -1.0
    if torch.cuda.is_available() or input_size[1] <= 128:
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(
                model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False
            )
            gflops = (macs * 2) / 1e9
        except Exception as e:
            print(f"Lưu ý ptflops ({model_name}): {e}")

    # 3. Latency & Peak VRAM
    dummy_input = torch.randn(1, *input_size, device=device)
    
    if torch.cuda.is_available():
        # Warmup trên GPU
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        latency_ms = ((time.time() - start_time) / num_runs) * 1000
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        # Chạy 1 lần trên CPU để kiểm thử logic
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        latency_ms = (time.time() - start_time) * 1000
        peak_vram_mb = 0.0

    print(f"[{model_name}] Params: {total_params:.2f}M | FLOPs ({input_size[1]}x{input_size[2]}): {gflops:.2f}G | Latency: {latency_ms:.1f}ms | VRAM: {peak_vram_mb:.1f}MB ({device})")
    return {
        'model': model_name,
        'input_size': f"{input_size[0]}x{input_size[1]}x{input_size[2]}",
        'params_m': round(total_params, 3),
        'gflops': round(gflops, 3) if gflops > 0 else 'N/A_CPU',
        'latency_ms': round(latency_ms, 2),
        'peak_vram_mb': round(peak_vram_mb, 2),
        'device': str(device)
    }

def main():
    print("=== P1-T2: BẮT ĐẦU ĐO ĐẠC HIỆU NĂNG MÔ HÌNH BASELINE ===")
    
    from baselines.promptir.net.model import PromptIR
    from baselines.adair.net.model import AdaIR
    
    results = []
    
    # 1. Profile PromptIR
    m_promptir = PromptIR(decoder=True)
    results.append(profile_model('PromptIR', m_promptir, input_size=(3, 128, 128)))
    if torch.cuda.is_available():
        results.append(profile_model('PromptIR', m_promptir, input_size=(3, 256, 256)))
    
    # 2. Profile AdaIR
    m_adair = AdaIR(decoder=True)
    results.append(profile_model('AdaIR', m_adair, input_size=(3, 128, 128)))
    if torch.cuda.is_available():
        results.append(profile_model('AdaIR', m_adair, input_size=(3, 256, 256)))
    
    out_csv = Path('repo/results/efficiency.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'input_size', 'params_m', 'gflops', 'latency_ms', 'peak_vram_mb', 'device'])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Đã lưu kết quả hiệu năng vào: {out_csv.resolve()}")

if __name__ == '__main__':
    main()
