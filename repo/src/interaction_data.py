"""Deterministic paired-order synthesis and full evaluation grids (pilot v4).

Synthetic degradations are generated from real clean images. Missing data is an
error: this pipeline never replaces unavailable photographs with random textures.
"""

import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


ORDERS = ("haze_then_noise", "noise_then_haze")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def image_files(directories):
    files = []
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing image directory: {directory}")
        files.extend(sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS))
    if not files:
        raise ValueError("No clean images found")
    if any(p.name.startswith("clean_synthetic_") for p in files):
        raise ValueError("Synthetic fallback images from prepare_data_v2 are not valid for pilot v4")
    return files


def sample_rng(seed, *indices):
    sequence = np.random.SeedSequence([seed, *indices])
    return np.random.RandomState(int(sequence.generate_state(1)[0]))


def hybrid_degradation(clean, noise, transmission, airlight, order):
    # Same noise realization for both orders; clip only after composition.
    if order == "haze_then_noise":
        result = clean * transmission + airlight * (1 - transmission) + noise
    elif order == "noise_then_haze":
        result = (clean + noise) * transmission + airlight * (1 - transmission)
    else:
        raise ValueError(f"Unknown degradation order: {order}")
    return np.clip(result, 0, 1).astype(np.float32)


def rain_degradation(clean, rng):
    h, w = clean.shape[:2]
    layer = np.zeros_like(clean)
    for _ in range(int(rng.randint(40, 120))):
        y, x = rng.randint(h), rng.randint(w)
        angle = rng.uniform(-0.3, 0.3)
        strength = rng.uniform(0.3, 0.7)
        for offset in range(int(rng.randint(10, 25))):
            yy, xx = y + offset, int(x + offset * angle)
            if 0 <= yy < h and 0 <= xx < w:
                layer[yy, xx] = strength
    return np.clip(clean + layer, 0, 1)


def tensor_image(array):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()


def load_image(path, max_size=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_size is not None:
            image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


class InteractionTrainDataset(Dataset):
    def __init__(self, directory, length, patch_size=128, seed=42):
        self.files = image_files([directory])
        self.images = [load_image(path) for path in self.files]
        self.length = length
        self.patch_size = patch_size
        self.seed = seed
        if any(min(image.shape[:2]) < patch_size for image in self.images):
            raise ValueError("All training images must be at least patch_size in both dimensions")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = sample_rng(self.seed, index)
        clean = self.images[rng.randint(len(self.images))]
        h, w = clean.shape[:2]
        size = self.patch_size
        top, left = rng.randint(h - size + 1), rng.randint(w - size + 1)
        clean = clean[top:top + size, left:left + size].copy()
        clean = np.rot90(clean, rng.randint(4))
        if rng.rand() < 0.5:
            clean = np.fliplr(clean)
        # Every eight samples: 2 noise, 2 rain, 2 haze, 1 of each hybrid order.
        mode = index % 4
        if mode == 0:
            sigma = rng.uniform(15, 50) / 255
            degraded = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 1)
        elif mode == 1:
            degraded = rain_degradation(clean, rng)
        elif mode == 2:
            transmission, airlight = rng.uniform(0.2, 0.8), rng.uniform(0.7, 1)
            degraded = clean * transmission + airlight * (1 - transmission)
        else:
            transmission, airlight = rng.uniform(0.3, 0.7), rng.uniform(0.7, 1)
            sigma = rng.uniform(15, 45) / 255
            noise = rng.normal(0, sigma, clean.shape).astype(np.float32)
            order = ORDERS[(index // 4) % 2]
            degraded = hybrid_degradation(clean, noise, transmission, airlight, order)
        return tensor_image(degraded.astype(np.float32)), tensor_image(clean)


class InteractionEvalDataset(Dataset):
    """All images x all grid cells x both orders, fixed independently of train seed.

    composition: partially overlaps training support, not wholly OOD.
    extrapolation: sigma and transmission outside training hybrid ranges.
    Validation also reports single-task synthetic noise/rain/haze on every image.
    """
    def __init__(self, directories, split="composition", max_images=30, seed=999, max_size=512):
        if split not in ("validation", "composition", "extrapolation"):
            raise ValueError(f"Unknown split: {split}")
        if max_images < 1 or max_size < 64:
            raise ValueError("max_images must be positive and max_size >= 64")
        # Round-robin sources rather than silently taking only BSD68 when the
        # first directory alone already contains more than max_images files.
        sources = [image_files([directory]) for directory in directories]
        self.files = [path for group in zip_longest(*sources) for path in group if path is not None][:max_images]
        self.images = [load_image(path, max_size) for path in self.files]
        if any(min(image.shape[:2]) < 64 for image in self.images):
            raise ValueError("Evaluation images must be at least 64 pixels on both axes")
        self.seed, self.split, self.max_size = seed, split, max_size
        grid = [(25, 0.5)] if split == "validation" else [(s, t) for s in (15, 35, 50) for t in (0.4, 0.8)]
        if split == "extrapolation":
            grid = [(s, t) for s in (10, 65) for t in (0.25, 0.85)]
        self.records = []
        for image_index in range(len(self.files)):
            if split == "validation":
                for task in ("denoise", "derain", "dehaze"):
                    self.records.append({"image": image_index, "task": task})
            for sigma, transmission in grid:
                for order in ORDERS:
                    self.records.append({"image": image_index, "sigma": sigma, "t": transmission,
                                         "order": order, "task": f"hybrid_s{sigma}_t{transmission}_{order}"})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        clean = self.images[record["image"]]
        task = record["task"]
        if "order" in record:
            rng = sample_rng(self.seed, record["image"], record["sigma"], int(record["t"] * 1000))
            noise = rng.normal(0, record["sigma"] / 255, clean.shape).astype(np.float32)
            airlight = 0.9 if self.split == "validation" else 1.0
            degraded = hybrid_degradation(clean, noise, record["t"], airlight, record["order"])
        else:
            rng = sample_rng(self.seed, record["image"], {"denoise": 1, "derain": 2, "dehaze": 3}[task])
            if task == "denoise":
                degraded = np.clip(clean + rng.normal(0, 35 / 255, clean.shape), 0, 1)
            elif task == "derain":
                degraded = rain_degradation(clean, rng)
            else:
                degraded = clean * 0.5 + 0.9 * 0.5
        return tensor_image(degraded.astype(np.float32)), tensor_image(clean), task


def file_manifest(files):
    return [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]


def dataset_manifest(dataset):
    manifest = {"files": file_manifest(dataset.files)}
    if isinstance(dataset, InteractionEvalDataset):
        manifest.update(seed=dataset.seed, split=dataset.split, max_size=dataset.max_size, records=dataset.records)
    else:
        manifest.update(seed=dataset.seed, length=len(dataset), patch_size=dataset.patch_size)
        hasher = hashlib.sha256()
        for index in range(min(50, len(dataset))):
            for tensor in dataset[index]:
                hasher.update(tensor.numpy().tobytes())
        manifest["first_50_full_samples_sha256"] = hasher.hexdigest()
    payload = json.dumps(manifest, sort_keys=True).encode()
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    return manifest
