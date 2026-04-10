"""DDPM (Denoising Diffusion Probabilistic Model) benchmark model.

UNet backbone with timestep conditioning and self-attention for noise
prediction.  Architecture follows Improved DDPM / ADM conventions:

- 4 encoder/decoder levels with channel mult [1, 2, 3, 4]
- Self-attention at 16×16 resolution and in the bottleneck
- Sinusoidal timestep embedding projected to 512-d

Input size: (3, 64, 64) — standard DDPM training resolution.

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

    @property
    def use_channels_last(self) -> bool:
        return True

    def create_model(self, num_classes: int = 10) -> nn.Module:
        return DiffusionUNet(
            in_channels=3,
            base_channels=128,
            channel_mult=(1, 2, 3, 4),
            attention_resolutions=(16,),
            time_embed_dim=512,
        )

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


class SelfAttention(nn.Module):
    """Self-attention with GroupNorm and residual connection."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)           # (B, HW, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h


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
    """UNet for DDPM noise prediction.

    4 encoder/decoder levels with configurable channel multipliers and
    self-attention at specified spatial resolutions.  Follows the design
    of Improved DDPM (Nichol & Dhariwal, 2021).

    Default config at 64×64 input (~60M params):
        channel_mult=(1, 2, 3, 4)  →  [128, 256, 384, 512]
        attention at 16×16 and bottleneck
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 3, 4),
        attention_resolutions: Tuple[int, ...] = (16,),
        time_embed_dim: int = 512,
        num_heads: int = 4,
    ):
        super().__init__()
        C = base_channels
        channels = [C * m for m in channel_mult]    # e.g. [128, 256, 384, 512]
        num_levels = len(channel_mult)
        # Spatial resolution halves at each downsample: 64 → 32 → 16 → 8
        # The resolution *entering* each level (after the preceding downsample)
        input_res = 64  # assumes 64×64 input

        # ---- Time embedding ----
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(C),
            nn.Linear(C, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # ---- Encoder ----
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.enc_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_ch = channels[0]
        res = input_res
        for i, ch in enumerate(channels):
            # Two ResBlocks per level
            self.enc_blocks.append(ResBlock(prev_ch, ch, time_embed_dim))
            self.enc_blocks.append(ResBlock(ch, ch, time_embed_dim))
            # Attention if resolution matches
            if res in attention_resolutions:
                self.enc_attns.append(SelfAttention(ch, num_heads))
            else:
                self.enc_attns.append(nn.Identity())
            # Downsample (except the last level feeds into bottleneck directly)
            if i < num_levels - 1:
                self.downsamples.append(Downsample(ch))
                res //= 2
            prev_ch = ch

        # ---- Bottleneck ----
        self.mid1 = ResBlock(channels[-1], channels[-1], time_embed_dim)
        self.mid_attn = SelfAttention(channels[-1], num_heads)
        self.mid2 = ResBlock(channels[-1], channels[-1], time_embed_dim)

        # ---- Decoder ----
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_attns = nn.ModuleList()

        for i in reversed(range(num_levels)):
            ch = channels[i]
            # Input channels: current level channels + skip connection
            skip_ch = ch
            if i < num_levels - 1:
                self.upsamples.append(Upsample(prev_ch))
                res *= 2
            in_ch = prev_ch + skip_ch if i < num_levels - 1 else channels[-1] + skip_ch
            self.dec_blocks.append(ResBlock(in_ch, ch, time_embed_dim))
            self.dec_blocks.append(ResBlock(ch, ch, time_embed_dim))
            if res in attention_resolutions:
                self.dec_attns.append(SelfAttention(ch, num_heads))
            else:
                self.dec_attns.append(nn.Identity())
            prev_ch = ch

        # ---- Output ----
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 3, padding=1),
        )

        self.num_levels = num_levels

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        # Encoder
        h = self.conv_in(x)
        skips = []
        block_idx = 0
        for i in range(self.num_levels):
            h = self.enc_blocks[block_idx](h, t_emb)
            block_idx += 1
            h = self.enc_blocks[block_idx](h, t_emb)
            block_idx += 1
            h = self.enc_attns[i](h)
            skips.append(h)
            if i < self.num_levels - 1:
                h = self.downsamples[i](h)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Decoder
        block_idx = 0
        for i, level in enumerate(reversed(range(self.num_levels))):
            skip = skips[level]
            if i > 0:
                h = self.upsamples[i - 1](h)
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[block_idx](h, t_emb)
            block_idx += 1
            h = self.dec_blocks[block_idx](h, t_emb)
            block_idx += 1
            h = self.dec_attns[i](h)

        return self.conv_out(h)
