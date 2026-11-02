"""Identity-initialized residual prompts with controlled interaction ablations.

No claim of physical identification is made for the learned matrix generators.
All four residual variants have the same parameters and share the official
PromptIR forward path. Only pooling and the interaction operator change.
"""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from baselines.promptir.net.model import PromptIR


VARIANTS = ("baseline", "global_linear", "regional_linear", "regional_jordan", "regional_lie")


def require_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite tensor: {name}")


class InteractionResidualPrompt(nn.Module):
    def __init__(self, base_prompt, variant, rank=16, hidden=64, residual_cap=0.20):
        super().__init__()
        if variant not in VARIANTS[1:]:
            raise ValueError(f"Unknown residual variant: {variant}")
        if not 0 < residual_cap <= 1:
            raise ValueError("residual_cap must be in (0, 1]")
        self.base_prompt = base_prompt
        self.variant = variant
        self.rank = rank
        self.residual_cap = residual_cap
        channels = base_prompt.linear_layer.in_features
        prompt_dim = base_prompt.conv3x3.out_channels
        self.descriptor = nn.Sequential(nn.LayerNorm(2 * channels), nn.Linear(2 * channels, hidden), nn.SiLU())
        self.head_a = nn.Linear(hidden, rank * rank)
        self.head_b = nn.Linear(hidden, rank * rank)
        self.value = nn.Conv2d(channels, rank, 1)
        self.output = nn.Conv2d(rank, prompt_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.enabled = True
        self.last_ratio = 0.0

    @staticmethod
    def normalize_matrix(matrix):
        return matrix / torch.linalg.vector_norm(matrix, dim=(-2, -1), keepdim=True).clamp_min(1.0)

    def forward(self, x):
        require_finite(x, "prompt input")
        base = self.base_prompt(x)
        require_finite(base, "base prompt")
        if not self.enabled:
            self.last_ratio = 0.0
            return base

        # AMP remains available for the backbone; every new operation is FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            xf = xf / xf.square().mean(dim=(1, 2, 3), keepdim=True).add(1e-6).sqrt()
            grid = 1 if self.variant == "global_linear" else 2
            mean = F.adaptive_avg_pool2d(xf, grid)
            variance = (F.adaptive_avg_pool2d(xf.square(), grid) - mean.square()).clamp_min(0)
            stats = torch.cat((mean, variance.add(1e-6).sqrt()), dim=1)
            stats = F.interpolate(stats, (2, 2), mode="nearest").flatten(2).transpose(1, 2)
            context = self.descriptor(stats)
            shape = (x.shape[0], 4, self.rank, self.rank)
            a = self.normalize_matrix(self.head_a(context).reshape(shape))
            b = self.normalize_matrix(self.head_b(context).reshape(shape))
            if self.variant == "regional_lie":
                operator = 0.5 * (a @ b - b @ a)
            elif self.variant == "regional_jordan":
                operator = 0.5 * (a @ b + b @ a)
            else:
                operator = 0.5 * (a + b)
            require_finite(operator, "normalized interaction")
            values = self.value(xf)
            pooled = F.adaptive_avg_pool2d(values, (2, 2)).flatten(2).transpose(1, 2)
            interaction = (operator @ pooled.unsqueeze(-1)).squeeze(-1)
            interaction = interaction.transpose(1, 2).reshape(x.shape[0], self.rank, 2, 2)
            interaction = F.interpolate(interaction, x.shape[-2:], mode="bilinear", align_corners=False)
            residual = self.output(F.silu(values + interaction))
            require_finite(residual, "residual before norm bound")
            base32 = base.float()
            # Exact per-sample L2 cap, zero-safe and detached; reductions stay FP32.
            with torch.no_grad():
                base_norm = torch.linalg.vector_norm(base32, dim=(1, 2, 3), keepdim=True)
                novel_norm = torch.linalg.vector_norm(residual, dim=(1, 2, 3), keepdim=True)
                scale = (self.residual_cap * base_norm / novel_norm.clamp_min(1e-12)).clamp(max=1.0)
            residual = residual * scale
            self.last_ratio = float((torch.linalg.vector_norm(residual.detach()) /
                                     torch.linalg.vector_norm(base32.detach()).clamp_min(1e-12)).item())
            result = base32 + residual
            require_finite(result, "coupled prompt")
        return result.to(base.dtype)


class InteractionPromptIR(PromptIR):
    def __init__(self, variant="regional_lie", residual_cap=0.20, **kwargs):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        kwargs.setdefault("decoder", True)
        super().__init__(**kwargs)
        if not self.decoder:
            raise ValueError("InteractionPromptIR requires decoder=True")
        self.variant = variant
        if variant != "baseline":
            for name in ("prompt1", "prompt2", "prompt3"):
                setattr(self, name, InteractionResidualPrompt(getattr(self, name), variant, residual_cap=residual_cap))

    def load_promptir_state(self, state):
        """Require a complete official backbone; never silently use missing weights."""
        cleaned = {}
        for key, value in state.items():
            while key.startswith(("net.", "module.")):
                key = key.split(".", 1)[1]
            if self.variant != "baseline" and key.split(".", 1)[0] in ("prompt1", "prompt2", "prompt3"):
                first, rest = key.split(".", 1)
                key = f"{first}.base_prompt.{rest}"
            cleaned[key] = value
        expected = {name for name in self.state_dict()
                    if self.variant == "baseline" or not self.is_novel_parameter(name)}
        missing = expected - cleaned.keys()
        unexpected = cleaned.keys() - expected
        if missing or unexpected:
            raise ValueError(f"Invalid PromptIR checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        self.load_state_dict(cleaned, strict=(self.variant == "baseline"))

    @staticmethod
    def is_novel_parameter(name):
        return name.startswith(("prompt1.", "prompt2.", "prompt3.")) and ".base_prompt." not in name

    @contextmanager
    def residuals_disabled(self):
        blocks = [module for module in self.modules() if isinstance(module, InteractionResidualPrompt)]
        states = [block.enabled for block in blocks]
        try:
            for block in blocks:
                block.enabled = False
            yield
        finally:
            for block, state in zip(blocks, states):
                block.enabled = state

    def get_new_block_ratio(self):
        ratios = [module.last_ratio for module in self.modules() if isinstance(module, InteractionResidualPrompt)]
        return sum(ratios) / len(ratios) if ratios else 0.0
