"""Finite-only pilot v4 with paired-order grids and reproducible resume.

Run baseline first for each seed. Results live exclusively in the run directory;
legacy v1-v3 CSVs are not used. Default FP32 establishes the stability reference.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.unified_evaluator import UnifiedEvaluator
from src.interaction_data import InteractionTrainDataset, InteractionEvalDataset, dataset_manifest
from src.interaction_training import (
    set_seed, train_step, checkpoint_state, restore_checkpoint, evaluate_groups, evaluate_intervention,
)
from src.models.interaction_promptir import InteractionPromptIR, VARIANTS
from src.utils.leakage_checker import verify_zero_leakage


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=999)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_novel", type=float, default=2e-4)
    parser.add_argument("--residual_cap", type=float, default=0.20)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--val_freq", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--train_dir", default="data/train_v2/clean")
    parser.add_argument("--val_dir", default="data/val_v2/clean")
    parser.add_argument("--test_dirs", nargs="+", default=["data/test/denoise/bsd68", "data/test/denoise/kodak24"])
    parser.add_argument("--val_images", type=int, default=10)
    parser.add_argument("--test_images", type=int, default=30)
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--test_size", type=int, default=512)
    parser.add_argument("--init_weights", default="baselines/weights/promptir_all.ckpt")
    parser.add_argument("--output_dir", default="runs/interaction_v4")
    parser.add_argument("--resume", action="store_true", help="Resume this run's latest_checkpoint.pth")
    parser.add_argument("--stop_after", type=int, help="Pause after this absolute iteration, preserving the planned schedule")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--random_init", action="store_true", help="Only allowed for explicitly labeled smoke tests")
    args = parser.parse_args(argv)
    args.iters = args.iters if args.iters is not None else (2 if args.smoke_test else 1500)
    if args.smoke_test:
        args.output_dir = "runs/smoke_interaction_v4"
        args.val_images = args.test_images = 1
        args.val_size = args.test_size = 128
        args.val_freq = args.save_every = args.iters
    if min(args.iters, args.batch_size, args.patch_size, args.val_freq, args.save_every) < 1:
        parser.error("Iterations, sizes and frequencies must be positive")
    if args.patch_size % 8:
        parser.error("patch_size must be divisible by 8 for PromptIR")
    if args.lr <= 0 or args.lr_novel <= 0:
        parser.error("Learning rates must be positive")
    if args.random_init and not args.smoke_test:
        parser.error("--random_init is restricted to --smoke_test")
    if args.stop_after is not None and not 0 < args.stop_after <= args.iters:
        parser.error("stop_after must be in [1, iters]")
    return args


def run_experiment(args):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if args.precision == "amp" and device.type != "cuda":
        raise ValueError("AMP experiments require CUDA; use fp32 for CPU checks")
    set_seed(args.seed)
    # PromptIR bilinear interpolation / adaptive pooling backward can lack a
    # deterministic CUDA kernel. Request deterministic alternatives where they
    # exist, warn otherwise, and do not falsely promise bitwise T4 reproducibility.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    train_dir, val_dir = (REPO_ROOT / args.train_dir).resolve(), (REPO_ROOT / args.val_dir).resolve()
    test_dirs = [(REPO_ROOT / path).resolve() for path in args.test_dirs]
    weights_path = (REPO_ROOT / args.init_weights).resolve()
    if not args.random_init and not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint required: {weights_path}")
    # Require every requested test directory, in addition to the overlap audit.
    if any(not path.is_dir() for path in test_dirs):
        raise FileNotFoundError(f"Missing a requested test directory: {test_dirs}")
    verify_zero_leakage(train_dir, val_dir, test_dirs, near_dup_threshold=4)

    train_data = InteractionTrainDataset(train_dir, args.iters * args.batch_size, args.patch_size, args.seed)
    val_data = InteractionEvalDataset([val_dir], "validation", args.val_images, args.eval_seed, args.val_size)
    composition_data = InteractionEvalDataset(test_dirs, "composition", args.test_images, args.eval_seed, args.test_size)
    extrapolation_data = InteractionEvalDataset(test_dirs, "extrapolation", args.test_images, args.eval_seed, args.test_size)
    manifests = {name: dataset_manifest(dataset) for name, dataset in
                 (("train", train_data), ("validation", val_data),
                  ("composition", composition_data), ("extrapolation", extrapolation_data))}
    source_files = [Path(__file__), REPO_ROOT / "src/interaction_data.py", REPO_ROOT / "src/interaction_training.py",
                    REPO_ROOT / "src/models/interaction_promptir.py", REPO_ROOT / "src/evaluation/unified_evaluator.py",
                    REPO_ROOT / "src/utils/leakage_checker.py", REPO_ROOT / "baselines/promptir/net/model.py"]
    shared = {
        "version": "interaction-v4", "seed": args.seed, "eval_seed": args.eval_seed,
        "iters": args.iters, "batch_size": args.batch_size, "patch_size": args.patch_size,
        "lr": args.lr, "lr_novel": args.lr_novel, "residual_cap": args.residual_cap,
        "precision": args.precision, "device": device.type, "torch_version": str(torch.__version__),
        "deterministic_algorithms": "warn_only", "allow_tf32": False,
        "val_freq": args.val_freq, "smoke_test": args.smoke_test, "random_init": args.random_init,
        "pretrained_sha256": None if args.random_init else sha256_file(weights_path),
        "datasets": {key: value["sha256"] for key, value in manifests.items()},
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_files},
    }
    root = (REPO_ROOT / args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f"protocol_seed_{args.seed}.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != shared:
            raise ValueError("Paired protocol mismatch; use a new output directory for a new experiment")
    elif args.variant != "baseline":
        raise FileNotFoundError("Run baseline for this seed first to lock the shared protocol")
    else:
        write_json(lock_path, shared)
    run_dir = root / f"{args.variant}_seed_{args.seed}"
    checkpoint_path = run_dir / "latest_checkpoint.pth"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {run_dir}. Use --resume or a new output directory.")
    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"variant": args.variant, "shared": shared}
    write_json(run_dir / "data_manifest.json", manifests)
    write_json(run_dir / "protocol.json", protocol)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    write_json(run_dir / "environment.json", {"git_commit": commit, "torch": str(torch.__version__),
               "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    write_json(run_dir / "status.json", {"status": "running"})
    iteration = 0
    started = time.monotonic()
    try:
        model = InteractionPromptIR(args.variant, args.residual_cap).to(device)
        if not args.random_init:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint.get("params", checkpoint))
            model.load_promptir_state(state)
            del checkpoint, state
        base, novel = [], []
        for name, parameter in model.named_parameters():
            target = novel if args.variant != "baseline" and model.is_novel_parameter(name) else base
            target.append(parameter)
        groups = [{"params": base, "lr": args.lr}]
        if novel:
            groups.append({"params": novel, "lr": args.lr_novel})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
        amp = args.precision == "amp"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        evaluator = UnifiedEvaluator(device, round_metrics=False)
        best = {"iteration": 0, "hybrid_psnr": None}
        history = []
        initial_validation = None
        if args.resume:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            restore_checkpoint(checkpoint, model, optimizer, scheduler, scaler, protocol)
            iteration, best, history = checkpoint["iteration"], checkpoint["best"], checkpoint["history"]
            initial_validation = checkpoint["initial_validation"]
            if not (run_dir / "best_model.pth").is_file():
                raise FileNotFoundError("Resume requires the run's best_model.pth for sensitivity evaluation")
            del checkpoint
        else:
            initial_validation = evaluate_groups(evaluator, model, val_data)
            # Include iter 0 in validation selection: training may reduce quality.
            best = {"iteration": 0, "hybrid_psnr": initial_validation["psnr_hybrid"]}
            save_torch(run_dir / "best_model.pth", model.state_dict())

        print(f"{args.variant} seed={args.seed} device={device} precision={args.precision}; "
              f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
        print(f"Fixed grids: composition={len(composition_data)}, extrapolation={len(extrapolation_data)}", flush=True)
        # Each index defines its own RNG. Resume starts at the exact next sample,
        # without replaying earlier batches or consuming global RNG state.
        remaining = Subset(train_data, range(iteration * args.batch_size, len(train_data)))
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(remaining, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=generator)
        stop = args.stop_after or args.iters
        for degraded, clean in loader:
            if iteration >= stop:
                break
            iteration += 1
            loss = train_step(model, degraded.to(device), clean.to(device), optimizer, scaler, scheduler, amp)
            if iteration % 50 == 0 or iteration == stop:
                print(f"iter={iteration}/{args.iters} loss={loss:.6f} ratio={model.get_new_block_ratio():.4f}", flush=True)
            if iteration % args.val_freq == 0 or iteration == args.iters:
                validation = evaluate_groups(evaluator, model, val_data)
                history.append({"iteration": iteration, "loss": loss, "validation": validation})
                print(f"val hybrid={validation['psnr_hybrid']:.4f} dB", flush=True)
                if validation["psnr_hybrid"] > best["hybrid_psnr"]:
                    best = {"iteration": iteration, "hybrid_psnr": validation["psnr_hybrid"]}
                    save_torch(run_dir / "best_model.pth", model.state_dict())
            if iteration % args.save_every == 0 or iteration == stop:
                checkpoint = checkpoint_state(model, optimizer, scheduler, scaler, iteration, protocol, best, history)
                checkpoint["initial_validation"] = initial_validation
                save_torch(checkpoint_path, checkpoint)
                del checkpoint

        if iteration < args.iters:
            write_json(run_dir / "status.json", {"status": "paused", "iteration": iteration})
            print(f"Paused at {iteration}; rerun with --resume and the original --iters.")
            return

        save_torch(run_dir / "final_model.pth", model.state_dict())
        evaluations = {}
        for checkpoint_name in ("final", "best"):
            if checkpoint_name == "best" and best["iteration"] == args.iters:
                evaluations["best"] = copy.deepcopy(evaluations["final"])
                continue
            model.load_state_dict(torch.load(run_dir / f"{checkpoint_name}_model.pth", map_location=device, weights_only=True))
            evaluations[checkpoint_name] = {
                "validation": evaluate_groups(evaluator, model, val_data),
                "composition": evaluate_intervention(evaluator, model, composition_data),
                "extrapolation": evaluate_intervention(evaluator, model, extrapolation_data),
            }
        summary = {"status": "complete", "variant": args.variant, "seed": args.seed,
                   "smoke_test": args.smoke_test, "protocol": protocol,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "novel_parameters": sum(p.numel() for p in novel), "best_validation": best,
                   "initial_validation": initial_validation, "evaluations": evaluations,
                   "elapsed_hours": (time.monotonic() - started) / 3600, "history": history}
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "status.json", {"status": "complete", "iteration": iteration})
        print(f"Complete: {run_dir / 'summary.json'}", flush=True)
    except Exception as error:
        write_json(run_dir / "status.json", {"status": "failed", "iteration": iteration,
                   "error_type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    run_experiment(parse_args())
