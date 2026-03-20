from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import torch


class DeviceBackend(ABC):
    """Abstract base class for device backends (CUDA, MPS, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Device type string, e.g. 'cuda', 'mps'."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this device backend is usable."""
        ...

    @abstractmethod
    def detect_devices(self, gpu_ids: List[int]) -> List[torch.device]:
        """Detect and return device objects for the given IDs."""
        ...

    @abstractmethod
    def get_device_name(self, device: torch.device) -> str:
        """Human-readable device name."""
        ...

    @abstractmethod
    def get_device_memory(self, device: torch.device) -> int:
        """Total device memory in bytes, or 0 if unavailable."""
        ...

    @abstractmethod
    def synchronize(self, device: torch.device) -> None:
        """Block until all pending operations on *device* complete."""
        ...

    @abstractmethod
    def get_autocast_context(self, device: torch.device, dtype: torch.dtype, enabled: bool):
        """Return an autocast context manager."""
        ...

    @abstractmethod
    def get_grad_scaler(self, enabled: bool) -> Optional[Any]:
        """Return a GradScaler (or None if not applicable)."""
        ...

    def supports_compile(self) -> bool:
        """Whether torch.compile is expected to work."""
        return False

    def supports_ddp(self) -> bool:
        """Whether DDP is supported on this backend."""
        return False

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs passed to torch.optim.SGD (e.g. fused=True)."""
        return {}

    def setup_precision(self, device: torch.device, is_main_process: bool = True) -> None:
        """Device-specific precision optimizations (e.g. TF32). No-op by default."""
        pass

    def print_device_info(self, devices: List[torch.device]) -> int:
        """Print device info and return total memory in bytes."""
        return 0

    def optimizer_step(self, optimizer, scaler=None, use_amp: bool = False) -> None:
        """Execute an optimizer step, handling AMP scaler if present.

        TPU overrides this to call ``xm.optimizer_step()`` instead.
        Default implementation preserves exact CUDA / MPS behavior.
        """
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    @property
    def should_reduce_logging(self) -> bool:
        """If True, runners should avoid calling loss.item() every step.

        On TPU/XLA, loss.item() triggers a graph sync and is very expensive.
        CUDA/MPS return False (no change to existing behavior).
        """
        return False
