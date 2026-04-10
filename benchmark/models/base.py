from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class BenchModel(ABC):
    """Abstract base class for benchmark models.

    Subclasses describe *what* to benchmark.  Device-specific and runner-
    specific behaviour lives in DeviceBackend / BenchRunner respectively.
    """

    # -------------------------------------------------------------- identity

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier used as the registry key, e.g. 'cnn', 'vit'."""
        ...

    def get_model_aliases(self) -> List[str]:
        """Alternative CLI names that should resolve to this model.

        Example: ResNet50Model might return ``["ResNet-50"]``.
        The registry uses these for CLI argument resolution.
        """
        return []

    # ----------------------------------------------------------- capabilities

    @property
    def supports_ddp(self) -> bool:
        return False

    @property
    def supports_amp(self) -> bool:
        """Whether this model should be run with AMP (autocast + GradScaler)."""
        return True

    @property
    def supports_compile(self) -> bool:
        """Whether torch.compile should be attempted for this model."""
        return True

    @property
    def use_channels_last(self) -> bool:
        """Whether to use channels_last memory format.

        Enable for conv-heavy models (UNet, DDPM, ResNet, CNN) — provides
        10–30 % speedup on modern GPUs with tensor cores.  Disable for
        models dominated by attention / linear layers (ViT).
        """
        return False

    # ------------------------------------------------------- model / data

    @abstractmethod
    def create_model(self, num_classes: int = 10) -> nn.Module:
        """Return a fresh ``nn.Module`` instance."""
        ...

    @abstractmethod
    def get_image_size(self) -> Tuple[int, ...]:
        """Return ``(C, H, W)`` expected by the model."""
        ...

    def get_num_classes(self) -> int:
        """Number of output classes (classification) or channels (segmentation)."""
        return 10

    @abstractmethod
    def get_default_batch_size(self, data_type: str = "FP32") -> int:
        """Return a sensible default batch size for *data_type*."""
        ...

    def get_data_item_bytes(self) -> int:
        """Estimated bytes per sample (float32).

        Used for ``data_size`` calculation, replacing the old hard-coded
        constant ``12296``.
        """
        C, H, W = self.get_image_size()
        return C * H * W * 4  # float32

    # -------------------------------------------------------- dataset factory

    def create_dataset(self, size: int) -> torch.utils.data.Dataset:
        """Return a synthetic dataset of *size* samples.

        The default implementation produces ``(image, class_label)`` pairs.
        Override for segmentation masks, noise targets, etc.
        """
        from benchmark.data import FakeDataset
        return FakeDataset(
            size=size,
            image_size=self.get_image_size(),
            num_classes=self.get_num_classes(),
        )

    # ----------------------------------------------------- training hooks

    def get_criterion(self) -> nn.Module:
        """Loss function.  Override for non-classification tasks."""
        return nn.CrossEntropyLoss()

    def compute_loss(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward pass + loss computation.

        The default implementation is ``criterion(model(images), labels)``.
        Override for tasks like diffusion where the forward pass is non-trivial
        (e.g. noise injection).
        """
        outputs = model(images)
        return criterion(outputs, labels)
