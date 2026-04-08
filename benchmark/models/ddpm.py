"""DDPM (Denoising Diffusion Probabilistic Model) benchmark model.

Simplified DDPM with a small UNet backbone for noise prediction.
Input size: (3, 64, 64) — typical DDPM training resolution.

The training step is non-standard: instead of classification, we add
noise to images and train the model to predict the noise.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from benchmark.models.base import BenchModel


class DDPMModel(BenchModel):
    """DDPM noise-prediction benchmark specification."""

    T = 1000  # diffusion timesteps

    @property
    def name(self) -> str:
        return "ddpm"

    def get_model_aliases(self) -> List[str]:
        return ["DDPM", "diffusion", "Diffusion"]

    @property
    def supports_ddp(self) -> bool:
        return True

    @property
    def supports_amp(self) -> bool:
        return True

    @property
    def supports_compile(self) -> bool:
        return True

    def create_model(self, num_classes: int = 10) -> nn.Module:
        return DiffusionUNet(in_channels=3, base_channels=128, time_embed_dim=256)

    def get_image_size(self) -> Tuple[int, ...]:
        return (3, 64, 64)

    def get_num_classes(self) -> int:
        return 0  # not a classification task

    def get_default_batch_size(self, data_type: str = "FP32") -> int:
        if data_type in ("FP16", "BF16"):
            return 64
        return 32

    # --- Diffusion-specific overrides ---

    def get_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def create_dataset(self, size: int) -> torch.utils.data.Dataset:
        from benchmark.data import FakeImageOnlyDataset
        return FakeImageOnlyDataset(
            size=size,
            image_size=self.get_image_size(),
        )

    def compute_loss(self, model, images, labels, criterion, device):
        """DDPM forward: sample t, add noise, predict noise."""
        B = images.shape[0]
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        noise = torch.randn_like(images)

        # Simplified linear schedule: alpha_bar(t) ≈ 1 - t/T
        alpha_bar = 1.0 - t.float() / self.T
        alpha_bar = alpha_bar.view(B, 1, 1, 1)

        noisy = torch.sqrt(alpha_bar) * images + torch.sqrt(1.0 - alpha_bar) * noise

        predicted_noise = model(noisy, t)
        return criterion(predicted_noise, noise)


# --------------------------------------------------------------------------- #
#  Diffusion UNet backbone
# --------------------------------------------------------------------------- #

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class DiffusionUNet(nn.Module):
    """Simplified UNet for DDPM noise prediction.

    Architecture: 3 levels, each with 2 ResBlocks.
    Channel progression: base → 2×base → 4×base.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 128, time_embed_dim: int = 256):
        super().__init__()
        C = base_channels

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(C),
            nn.Linear(C, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Encoder
        self.conv_in = nn.Conv2d(in_channels, C, 3, padding=1)
        self.enc1a = ResBlock(C, C, time_embed_dim)
        self.enc1b = ResBlock(C, C, time_embed_dim)
        self.down1 = Downsample(C)

        self.enc2a = ResBlock(C, 2 * C, time_embed_dim)
        self.enc2b = ResBlock(2 * C, 2 * C, time_embed_dim)
        self.down2 = Downsample(2 * C)

        self.enc3a = ResBlock(2 * C, 4 * C, time_embed_dim)
        self.enc3b = ResBlock(4 * C, 4 * C, time_embed_dim)

        # Bottleneck
        self.mid1 = ResBlock(4 * C, 4 * C, time_embed_dim)
        self.mid2 = ResBlock(4 * C, 4 * C, time_embed_dim)

        # Decoder
        self.up2 = Upsample(4 * C)
        self.dec2a = ResBlock(4 * C + 2 * C, 2 * C, time_embed_dim)
        self.dec2b = ResBlock(2 * C, 2 * C, time_embed_dim)

        self.up1 = Upsample(2 * C)
        self.dec1a = ResBlock(2 * C + C, C, time_embed_dim)
        self.dec1b = ResBlock(C, C, time_embed_dim)

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, C),
            nn.SiLU(),
            nn.Conv2d(C, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        # Encoder
        h = self.conv_in(x)
        h1 = self.enc1b(self.enc1a(h, t_emb), t_emb)
        h = self.down1(h1)

        h2 = self.enc2b(self.enc2a(h, t_emb), t_emb)
        h = self.down2(h2)

        h = self.enc3b(self.enc3a(h, t_emb), t_emb)

        # Bottleneck
        h = self.mid2(self.mid1(h, t_emb), t_emb)

        # Decoder
        h = self.up2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2b(self.dec2a(h, t_emb), t_emb)

        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1b(self.dec1a(h, t_emb), t_emb)

        return self.conv_out(h)
