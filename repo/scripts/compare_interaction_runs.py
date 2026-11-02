"""Compare complete finite pilot-v4 runs with paired protocol checks."""

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.models.interaction_promptir import VARIANTS


def reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON number is not a valid result: {value}")


def load_complete_run(root, variant, seed):
    directory = root / f"{variant}_seed_{seed}"
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    if status["status"] != "complete" or summary["status"] != "complete" or summary["smoke_test"]:
        raise ValueError(f"Run is incomplete, failed or smoke-only: {directory}")
    if summary["variant"] != variant or summary["seed"] != seed:
        raise ValueError(f"Run identity mismatch: {directory}")
    return summary


def compare(root, variants, seeds):
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least three distinct training seeds")
    if not variants or len(set(variants)) != len(variants) or "baseline" not in variants:
        raise ValueError("Variants must be distinct and include baseline")
    runs = {(variant, seed): load_complete_run(root, variant, seed) for variant in variants for seed in seeds}
    evaluation_reference = None
    for seed in seeds:
        shared = runs["baseline", seed]["protocol"]["shared"]
        for variant in variants:
            if runs[variant, seed]["protocol"]["shared"] != shared:
                raise ValueError(f"Unpaired protocol for {variant}, seed {seed}")
        # Across seeds ONLY seed and the training dataset fingerprint may differ.
        reference = {key: value for key, value in shared.items() if key not in ("seed", "datasets")}
        reference["evaluation_datasets"] = {key: value for key, value in shared["datasets"].items() if key != "train"}
        if evaluation_reference is not None and reference != evaluation_reference:
            raise ValueError("Evaluation data or training configuration differs across seeds")
        evaluation_reference = reference
    residual_counts = {runs[variant, seed]["parameters"] for variant in variants if variant != "baseline" for seed in seeds}
    if len(residual_counts) > 1:
        raise ValueError("Residual ablations must be parameter-matched")

    rows = []
    for checkpoint in ("final", "best"):
        for split in ("composition", "extrapolation"):
            baseline = [runs["baseline", seed]["evaluations"][checkpoint][split]["enabled"]["psnr_mean"] for seed in seeds]
            for variant in variants:
                entries = [runs[variant, seed]["evaluations"][checkpoint][split] for seed in seeds]
                values = [entry["enabled"]["psnr_mean"] for entry in entries]
                differences = [value - base for value, base in zip(values, baseline)]
                rows.append({"checkpoint": checkpoint, "split": split, "variant": variant,
                             "psnr_mean": statistics.mean(values), "psnr_std": statistics.stdev(values),
                             "paired_delta_vs_baseline": differences, "mean_delta_vs_baseline": statistics.mean(differences),
                             "mean_intervention_delta": statistics.mean(entry["delta_psnr_enabled_minus_disabled"] for entry in entries),
                             "mean_new_block_ratio": statistics.mean(entry["mean_new_block_ratio"] for entry in entries),
                             "haze_then_noise_psnr": statistics.mean(entry["enabled"]["psnr_haze_then_noise"] for entry in entries),
                             "noise_then_haze_psnr": statistics.mean(entry["enabled"]["psnr_noise_then_haze"] for entry in entries)})
    baseline_primary = [runs["baseline", seed]["evaluations"]["final"]["composition"]["enabled"]["psnr_mean"] for seed in seeds]
    baseline_std = statistics.stdev(baseline_primary)
    checks = []
    if "regional_lie" in variants:
        for control in ("baseline", "regional_linear", "regional_jordan"):
            if control not in variants:
                continue
            deltas = [runs["regional_lie", seed]["evaluations"]["final"]["composition"]["enabled"]["psnr_mean"] -
                      runs[control, seed]["evaluations"]["final"]["composition"]["enabled"]["psnr_mean"] for seed in seeds]
            threshold = max(0.40, 2 * baseline_std) if control == "baseline" else max(0.25, 1.5 * baseline_std)
            checks.append({"comparison": f"regional_lie_minus_{control}", "paired_deltas": deltas,
                           "mean_delta": statistics.mean(deltas), "reference_threshold": threshold,
                           "meets_reference_threshold": statistics.mean(deltas) >= threshold,
                           "all_seeds_positive": all(delta > 0 for delta in deltas)})
    return {"seeds": seeds, "rows": rows, "primary_endpoint": "final/composition",
            "reference_checks": checks,
            "interpretation": "Engineering pilot only. Synthetic validation is not a standard single-task benchmark. "
                              "No claim of all five legacy gates or statistical significance is made."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/interaction_v4")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    args = parser.parse_args()
    root = (REPO_ROOT / args.output_dir).resolve()
    report = compare(root, args.variants, args.seeds)
    print("Checkpoint | Split | Variant | PSNR mean +/- std | Delta vs baseline | Intervention")
    for row in report["rows"]:
        print(f"{row['checkpoint']} | {row['split']} | {row['variant']} | "
              f"{row['psnr_mean']:.4f} +/- {row['psnr_std']:.4f} | "
              f"{row['mean_delta_vs_baseline']:+.4f} | {row['mean_intervention_delta']:+.4f}")
    print(report["interpretation"])
    (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
