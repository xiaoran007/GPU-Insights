from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn


class DeviceBackend(ABC):
    """Abstract base class for device backends (CUDA, MPS, TPU, NPU, …).

    Every concrete backend **must** implement the abstract methods below.
    Optional hooks (``try_compile_model``, ``wrap_dataloader``, …) have
    sensible defaults that concrete backends may override.
    """

    # ----------------------------------------------------------- identity

    @property
    @abstractmethod
    def name(self) -> str:
        """Device type string, e.g. ``'cuda'``, ``'mps'``."""
        ...

    # -------------------------------------------------------- availability

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this device backend is usable."""
        ...

    # ---------------------------------------------------- device queries

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
        """Total device memory in bytes, or ``0`` if unavailable."""
        ...

    # ------------------------------------------------------- synchronise

    @abstractmethod
    def synchronize(self, device: torch.device) -> None:
        """Block until all pending operations on *device* complete."""
        ...

    # ------------------------------------------------------------- AMP

    @abstractmethod
    def get_autocast_context(self, device: torch.device, dtype: torch.dtype, enabled: bool):
        """Return an ``autocast`` context manager."""
        ...

    @abstractmethod
    def get_grad_scaler(self, enabled: bool) -> Optional[Any]:
        """Return a ``GradScaler`` (or ``None`` if not applicable)."""
        ...

    # ------------------------------------------------------- capabilities

    def supports_compile(self) -> bool:
        """Whether ``torch.compile`` is expected to work."""
        return False

    def supports_ddp(self) -> bool:
        """Whether DDP is supported on this backend."""
        return False

    def supports_bf16(self, device: torch.device) -> bool:
        """Whether BF16 is natively supported on *device*."""
        return False

    def supports_channels_last(self) -> bool:
        """Whether channels_last memory format is fully supported.

        Enabled by default on CUDA (tensor core benefit).  Disabled on
        MPS where backward-pass failures occur with some architectures.
        """
        return False

    # -------------------------------------------------------- optimizer

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs passed to ``torch.optim.SGD`` (e.g. ``fused=True``)."""
        return {}

    def optimizer_step(self, optimizer, scaler=None, use_amp: bool = False) -> None:
        """Execute an optimizer step, handling AMP scaler if present.

        TPU overrides this to call ``xm.optimizer_step()`` instead.
        """
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    # -------------------------------------------------------- precision

    def setup_precision(self, device: torch.device, is_main_process: bool = True) -> None:
        """Device-specific precision optimisations (e.g. TF32). No-op by default."""
        pass

    # -------------------------------------------------------- device info

    def print_device_info(self, devices: List[torch.device]) -> int:
        """Print device info and return total memory in bytes."""
        return 0

    # -------------------------------------------------------- compilation

    def try_compile_model(self, model: nn.Module, is_main_process: bool = True) -> nn.Module:
        """Attempt ``torch.compile`` and return the (possibly compiled) model.

        The default implementation returns the model unchanged.  CUDA/TPU
        override this with backend-specific compile logic.
        """
        return model

    # -------------------------------------------------------- dataloader

    def wrap_dataloader(self, dataloader, device: torch.device):
        """Optionally wrap a ``DataLoader`` for device-specific optimisation.

        TPU overrides this with ``MpDeviceLoader``.  Default: pass-through.
        """
        return dataloader

    # -------------------------------------------------------- memory info

    def get_peak_memory_mb(self, device: torch.device) -> float:
        """Return peak memory usage in MB since last reset, or ``0.0``."""
        return 0.0

    def reset_peak_memory(self, device: torch.device) -> None:
        """Reset peak memory tracking.  No-op by default."""
        pass

    # -------------------------------------------------------- logging hint

    @property
    def should_reduce_logging(self) -> bool:
        """If ``True``, runners should avoid calling ``loss.item()`` every step.

        On TPU/XLA ``loss.item()`` triggers a graph sync and is very expensive.
        """
        return False
