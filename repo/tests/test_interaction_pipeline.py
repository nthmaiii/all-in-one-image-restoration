"""Regression tests for finite metrics, fair controls, order synthesis and resume."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.promptir.net.model import PromptGenBlock
from src.evaluation.unified_evaluator import UnifiedEvaluator, compute_metrics
from src.interaction_data import (
    ORDERS, InteractionEvalDataset, InteractionTrainDataset, hybrid_degradation, dataset_manifest,
)
from src.interaction_training import (
    set_seed, train_step, checkpoint_state, restore_checkpoint, evaluate_intervention,
)
from src.models.interaction_promptir import InteractionPromptIR, InteractionResidualPrompt, VARIANTS


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def image_dirs(tmp_path):
    paths = []
    for index, name in enumerate(("train", "val", "test")):
        directory = tmp_path / name
        directory.mkdir()
        image = np.random.RandomState(100 + index).randint(40, 210, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(image).save(directory / f"{name}.png")
        paths.append(directory)
    return paths


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_metrics_reject_nonfinite_before_clipping(bad):
    image = torch.ones(1, 3, 64, 64)
    broken = image.clone()
    broken[0, 0, 0, 0] = bad
    with pytest.raises(FloatingPointError):
        compute_metrics(broken, image)

    class Broken(nn.Module):
        def forward(self, x):
            return x * bad

    with pytest.raises(FloatingPointError, match="sample 0"):
        UnifiedEvaluator(torch.device("cpu")).evaluate(Broken(), [(image[0], image[0], "hybrid")])


def test_empty_evaluation_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        UnifiedEvaluator(torch.device("cpu")).evaluate(nn.Identity(), [])


def test_order_gap_and_full_paired_grid(image_dirs):
    clean = np.full((8, 8, 3), 0.3, dtype=np.float32)
    noise = np.full_like(clean, 0.02)
    nh = hybrid_degradation(clean, noise, 0.6, 0.8, ORDERS[0])
    hn = hybrid_degradation(clean, noise, 0.6, 0.8, ORDERS[1])
    np.testing.assert_allclose(nh - hn, 0.4 * noise, atol=1e-7)
    dataset = InteractionEvalDataset([image_dirs[2]], max_size=64)
    assert len(dataset) == 12
    cells = {(record["sigma"], record["t"], record["order"]) for record in dataset.records}
    assert cells == {(sigma, t, order) for sigma in (15, 35, 50) for t in (0.4, 0.8) for order in ORDERS}
    # Manifest and exact degraded pixels are independent of global/training RNG.
    first = dataset_manifest(dataset)
    set_seed(2024)
    other = InteractionEvalDataset([image_dirs[2]], max_size=64)
    assert dataset_manifest(other) == first
    assert all(torch.equal(dataset[index][0], other[index][0]) for index in range(len(dataset)))


def test_dataset_resume_indices_and_real_image_requirement(image_dirs):
    first = InteractionTrainDataset(image_dirs[0], 16, patch_size=32, seed=42)
    resumed = InteractionTrainDataset(image_dirs[0], 16, patch_size=32, seed=42)
    for index in range(8, 16):
        assert all(torch.equal(a, b) for a, b in zip(first[index], resumed[index]))
    original_manifest = dataset_manifest(first)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(first.files[0])
    assert dataset_manifest(InteractionTrainDataset(image_dirs[0], 16, 32, 42)) != original_manifest
    first.files[0].rename(image_dirs[0] / "clean_synthetic_0000.png")
    with pytest.raises(ValueError, match="Synthetic fallback"):
        InteractionTrainDataset(image_dirs[0], 16, 32, 42)


def test_identity_matches_official_forward_and_strict_loading():
    config = {"num_blocks": [0, 0, 0, 0], "num_refinement_blocks": 0}
    set_seed(42)
    baseline = InteractionPromptIR("baseline", **config).eval()
    state = copy.deepcopy(baseline.state_dict())
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = baseline(image)
    counts = []
    for variant in VARIANTS[1:]:
        model = InteractionPromptIR(variant, **config).eval()
        model.load_promptir_state({"net." + name: value for name, value in state.items()})
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        with torch.no_grad():
            actual = model(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert model.get_new_block_ratio() == 0
        with pytest.raises(ValueError, match="checkpoint"):
            model.load_promptir_state({})
        del model
    assert len(set(counts)) == 1


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_zero_init_opens_gradients_then_respects_norm_bound(variant):
    set_seed(42)
    block = InteractionResidualPrompt(PromptGenBlock(4, 3, 4, 6), variant, rank=3, hidden=8)
    x = torch.randn(2, 6, 8, 8)
    target = torch.randn(2, 4, 8, 8)
    optimizer = torch.optim.Adam(block.parameters(), lr=1e-2)
    for step in range(3):
        optimizer.zero_grad()
        loss = (block(x) - target).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert block.output.weight.grad.abs().sum() > 0
        if step > 0:
            assert block.head_a.weight.grad.abs().sum() > 0
            assert block.head_b.weight.grad.abs().sum() > 0
        optimizer.step()
    with torch.no_grad():
        block.output.weight.fill_(100)
        base = block.base_prompt(x)
        result = block(x)
        ratio = torch.linalg.vector_norm(result - base, dim=(1, 2, 3)) / torch.linalg.vector_norm(base, dim=(1, 2, 3))
    assert torch.all(ratio <= 0.200001)
    assert torch.all(ratio > 0)
    # CPU autocast exercises the FP32 island; CUDA FP16 remains a Colab check.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = block(x)
    assert torch.isfinite(result).all()


def test_resume_matches_uninterrupted_rng_optimizer_and_schedule():
    set_seed(42)
    inputs = torch.rand(6, 2, 3)
    targets = torch.rand(6, 2, 3)

    def create():
        model = nn.Sequential(nn.Linear(3, 8), nn.Dropout(0.2), nn.Linear(8, 3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 6)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return model, optimizer, scheduler, scaler

    original, optimizer, scheduler, scaler = create()
    protocol = {"iters": 6}
    for index in range(3):
        train_step(original, inputs[index], targets[index], optimizer, scaler, scheduler)
    saved = copy.deepcopy(checkpoint_state(original, optimizer, scheduler, scaler, 3, protocol, {}, []))
    for index in range(3, 6):
        train_step(original, inputs[index], targets[index], optimizer, scaler, scheduler)
    resumed, optimizer2, scheduler2, scaler2 = create()
    restore_checkpoint(saved, resumed, optimizer2, scheduler2, scaler2, protocol)
    for index in range(saved["iteration"], 6):
        train_step(resumed, inputs[index], targets[index], optimizer2, scaler2, scheduler2)
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)
    assert scheduler.state_dict() == scheduler2.state_dict()
    assert optimizer.param_groups[0]["lr"] == optimizer2.param_groups[0]["lr"]
    with pytest.raises(ValueError, match="Resume requires"):
        restore_checkpoint(saved, resumed, optimizer2, scheduler2, scaler2, {"iters": 7})


def test_nonfinite_training_never_updates_weights():
    model = nn.Linear(3, 3)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with pytest.raises(FloatingPointError):
        train_step(model, torch.full((2, 3), float("nan")), torch.zeros(2, 3), optimizer, scaler, scheduler)
    assert not optimizer.state
    for name, tensor in before.items():
        assert torch.equal(tensor, model.state_dict()[name])


def test_intervention_measures_quality_and_restores_enabled_flag(image_dirs):
    class ToyRestorer(nn.Module):
        variant = "regional_lie"

        def __init__(self):
            super().__init__()
            self.prompt = InteractionResidualPrompt(PromptGenBlock(3, 3, 4, 3), "regional_lie", rank=2, hidden=4)
            self.enabled = True

        def forward(self, image):
            # Use a real prompt block so diagnostic hooks run on every image.
            self.prompt(image)
            return image if self.enabled else torch.zeros_like(image)

        @contextmanager
        def residuals_disabled(self):
            self.enabled = False
            try:
                yield
            finally:
                self.enabled = True

    model = ToyRestorer()
    data = InteractionEvalDataset([image_dirs[2]], max_size=64)
    result = evaluate_intervention(UnifiedEvaluator(torch.device("cpu"), round_metrics=False), model, data)
    assert result["delta_psnr_enabled_minus_disabled"] > 0
    assert result["enabled"]["sample_count"] == 12
    assert model.enabled


def test_runner_pauses_resumes_and_reports_final_and_best(image_dirs, tmp_path, monkeypatch):
    import scripts.train_interaction_promptir as runner

    class ToyBaseline(nn.Module):
        variant = "baseline"

        def __init__(self, *args):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, image):
            return self.conv(image)

        def get_new_block_ratio(self):
            return 0.0

    monkeypatch.setattr(runner, "InteractionPromptIR", ToyBaseline)
    args = runner.parse_args(["--variant", "baseline", "--smoke_test", "--random_init", "--iters", "2"])
    args.train_dir, args.val_dir = str(image_dirs[0]), str(image_dirs[1])
    args.test_dirs = [str(image_dirs[2])]
    args.output_dir = str(tmp_path / "runs")
    args.patch_size = 32
    args.val_size = args.test_size = 64
    args.val_freq = args.save_every = 1
    args.stop_after = 1
    runner.run_experiment(args)
    directory = Path(args.output_dir) / "baseline_seed_42"
    assert json.loads((directory / "status.json").read_text())["status"] == "paused"
    assert not (directory / "summary.json").exists()
    args.resume, args.stop_after = True, None
    runner.run_experiment(args)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["smoke_test"]
    assert set(summary["evaluations"]) == {"final", "best"}
    assert summary["evaluations"]["final"]["composition"]["enabled"]["sample_count"] == 12
    assert summary["evaluations"]["final"]["extrapolation"]["enabled"]["sample_count"] == 8
    assert [entry["iteration"] for entry in summary["history"]] == [1, 2]


def test_comparison_requires_paired_complete_finite_runs(tmp_path):
    from scripts.compare_interaction_runs import compare

    variants = ("baseline", "regional_jordan", "regional_lie")
    seeds = (42, 123, 2024)
    for variant in variants:
        for index, seed in enumerate(seeds):
            directory = tmp_path / f"{variant}_seed_{seed}"
            directory.mkdir()
            score = 20.0 + index * 0.1 + (1.0 if variant == "regional_lie" else 0.0)
            metrics = {"psnr_mean": score, "psnr_haze_then_noise": score, "psnr_noise_then_haze": score}
            entry = {"enabled": metrics, "delta_psnr_enabled_minus_disabled": 0.2, "mean_new_block_ratio": 0.1}
            shared = {"seed": seed, "eval_seed": 999, "datasets": {"train": str(seed), "validation": "val", "composition": "comp", "extrapolation": "extra"}}
            summary = {"status": "complete", "variant": variant, "seed": seed, "smoke_test": False,
                       "parameters": 100 if variant == "baseline" else 120, "protocol": {"shared": shared},
                       "evaluations": {name: {"composition": entry, "extrapolation": entry} for name in ("final", "best")}}
            (directory / "status.json").write_text(json.dumps({"status": "complete"}))
            (directory / "summary.json").write_text(json.dumps(summary))
    report = compare(tmp_path, variants, seeds)
    assert len(report["rows"]) == 12
    assert all(check["meets_reference_threshold"] for check in report["reference_checks"])
    path = tmp_path / "regional_lie_seed_42" / "summary.json"
    original = path.read_text()
    broken = json.loads(original)
    broken["protocol"]["shared"]["eval_seed"] = 42
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Unpaired protocol"):
        compare(tmp_path, variants, seeds)
    path.write_text(original)
    status = path.parent / "status.json"
    status.write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="incomplete, failed"):
        compare(tmp_path, variants, seeds)
    status.write_text(json.dumps({"status": "complete"}))
    path.write_text(original.replace('"smoke_test": false', '"smoke_test": NaN'))
    with pytest.raises(ValueError, match="Non-finite JSON"):
        compare(tmp_path, variants, seeds)


def test_real_image_preparation_does_not_silently_fallback(tmp_path, monkeypatch):
    import scripts.prepare_data_v2 as preparation

    def unavailable(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(preparation.urllib.request, "urlretrieve", unavailable)
    pool = tmp_path / "pool"
    with pytest.raises(RuntimeError, match="No synthetic fallback"):
        preparation.download_and_extract_clean_pool(pool, target_count=2, allow_synthetic=False)
    assert not list(pool.glob("*.png"))
