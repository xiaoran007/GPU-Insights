"""Vision Transformer (ViT) benchmark model.

Implements ViT-Base/16 style architecture for image classification.
Input size: (3, 224, 224) — standard ImageNet-scale input.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from benchmark.models.base import BenchModel


class ViTModel(BenchModel):
    """ViT-Base benchmark specification."""

    @property
    def name(self) -> str:
        return "vit"

    def get_model_aliases(self) -> List[str]:
        return ["ViT", "vit-base", "ViT-Base"]

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
        return VisionTransformer(
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=num_classes,
        )

    def get_image_size(self) -> Tuple[int, ...]:
        return (3, 224, 224)

    def get_default_batch_size(self, data_type: str = "FP32") -> int:
        if data_type in ("FP16", "BF16"):
            return 64
        return 32

    def get_num_classes(self) -> int:
        return 10


# --------------------------------------------------------------------------- #
#  ViT architecture
# --------------------------------------------------------------------------- #

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, embed_dim, H', W') → (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for classification."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        # Classification from [CLS] token
        return self.head(x[:, 0])
