"""UNet benchmark model for semantic segmentation.

Classic encoder-decoder with skip connections.
Input size: (3, 256, 256) — typical segmentation resolution.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from benchmark.models.base import BenchModel


class UNetModel(BenchModel):
    """UNet segmentation benchmark specification."""

    @property
    def name(self) -> str:
        return "unet"

    def get_model_aliases(self) -> List[str]:
        return ["UNet", "U-Net"]

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
        return UNet(in_channels=3, out_channels=num_classes)

    def get_image_size(self) -> Tuple[int, ...]:
        return (3, 256, 256)

    def get_num_classes(self) -> int:
        return 21  # PASCAL VOC style

    def get_default_batch_size(self, data_type: str = "FP32") -> int:
        if data_type in ("FP16", "BF16"):
            return 16
        return 8

    # --- Segmentation-specific overrides ---

    def get_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def create_dataset(self, size: int) -> torch.utils.data.Dataset:
        from benchmark.data import FakeSegmentationDataset
        return FakeSegmentationDataset(
            size=size,
            image_size=self.get_image_size(),
            num_classes=self.get_num_classes(),
        )

    def compute_loss(self, model, images, labels, criterion, device):
        outputs = model(images)
        # outputs: (B, num_classes, H, W), labels: (B, H, W)
        return criterion(outputs, labels)


# --------------------------------------------------------------------------- #
#  UNet architecture
# --------------------------------------------------------------------------- #

class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscale with MaxPool then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscale then DoubleConv with skip connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed (input sizes not exactly powers of 2)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard UNet with 4 encoder/decoder levels."""

    def __init__(self, in_channels: int = 3, out_channels: int = 21):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
