"""Finite-only updates, reproducible checkpoint state and intervention evaluation."""

from collections import Counter
import random

import numpy as np
import torch

from src.models.interaction_promptir import require_finite


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def train_step(model, degraded, clean, optimizer, scaler, scheduler, amp=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    require_finite(degraded, "training input")
    require_finite(clean, "training target")
    with torch.autocast(device_type=degraded.device.type, enabled=amp):
        prediction = model(degraded)
    require_finite(prediction, "restored training image")
    # Charbonnier arithmetic must not square a half-precision residual.
    loss = ((prediction.float() - clean.float()).square() + 1e-6).sqrt().mean()
    require_finite(loss, "reconstruction loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    except RuntimeError as error:
        raise FloatingPointError("Non-finite gradient norm; no optimizer step was taken") from error
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    # Fail immediately if an update corrupts a weight, even when loss was finite.
    parameters = list(model.named_parameters())
    if not torch.stack([torch.isfinite(parameter).all() for _, parameter in parameters]).all():
        for name, parameter in parameters:
            require_finite(parameter, f"updated parameter {name}")
    return float(loss.detach())


def checkpoint_state(model, optimizer, scheduler, scaler, iteration, protocol, best, history):
    return {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "iteration": iteration, "protocol": protocol, "best": best, "history": history,
            "rng_state": rng_state()}


def restore_checkpoint(checkpoint, model, optimizer, scheduler, scaler, protocol):
    if checkpoint["protocol"] != protocol:
        raise ValueError("Resume requires exactly the same variant, data, code, weights and training schedule")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    restore_rng(checkpoint["rng_state"])


def evaluate_groups(evaluator, model, dataset):
    metrics = evaluator.evaluate(model, dataset)
    counts = Counter(record["task"] for record in dataset.records)
    groups = {
        "hybrid": lambda task: task.startswith("hybrid_"),
        "haze_then_noise": lambda task: task.endswith("_haze_then_noise"),
        "noise_then_haze": lambda task: task.endswith("_noise_then_haze"),
    }
    for group, include in groups.items():
        selected = {task: count for task, count in counts.items() if include(task)}
        total = sum(selected.values())
        if total:
            for metric in ("psnr", "ssim"):
                metrics[f"{metric}_{group}"] = sum(metrics[f"{metric}_{task}"] * count
                                                     for task, count in selected.items()) / total
    metrics["sample_count"] = len(dataset)
    return metrics


def evaluate_intervention(evaluator, model, dataset):
    if model.variant == "baseline":
        enabled = evaluate_groups(evaluator, model, dataset)
        return {"enabled": enabled, "residual_disabled": enabled.copy(),
                "delta_psnr_enabled_minus_disabled": 0.0, "mean_new_block_ratio": 0.0}
    ratios = []

    def collect_ratio(module, inputs, output):
        ratios.append(module.last_ratio)

    from src.models.interaction_promptir import InteractionResidualPrompt
    handles = [module.register_forward_hook(collect_ratio) for module in model.modules()
               if isinstance(module, InteractionResidualPrompt)]
    try:
        # Evaluate with hooks so the ratio is averaged over ALL images,
        # not read from the final sample's mutable diagnostic state.
        enabled = evaluate_groups(evaluator, model, dataset)
    finally:
        for handle in handles:
            handle.remove()
    with model.residuals_disabled():
        disabled = evaluate_groups(evaluator, model, dataset)
    return {"enabled": enabled, "residual_disabled": disabled,
            "delta_psnr_enabled_minus_disabled": enabled["psnr_mean"] - disabled["psnr_mean"],
            "mean_new_block_ratio": float(np.mean(ratios))}
