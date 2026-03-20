from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn


class BenchModel(ABC):
    """Abstract base class for benchmark models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier, e.g. 'cnn', 'resnet50'."""
        ...

    @property
    def supports_ddp(self) -> bool:
        return False

    @abstractmethod
    def create_model(self, num_classes: int = 10) -> nn.Module:
        """Return a fresh nn.Module instance."""
        ...

    @abstractmethod
    def get_image_size(self) -> Tuple[int, ...]:
        """Return (C, H, W) expected by the model."""
        ...

    @abstractmethod
    def get_default_batch_size(self, data_type: str = "FP32") -> int:
        """Return sensible default batch size for this model."""
        ...
