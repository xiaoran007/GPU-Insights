"""TPU (XLA) device backend for Google Cloud TPU / Colab TPU."""

from typing import Any, Dict, List, Optional
from contextlib import contextmanager, nullcontext

import torch
from benchmark.devices.base import DeviceBackend


class TPUDeviceBackend(DeviceBackend):
    """DeviceBackend implementation for Google TPU via PyTorch/XLA."""

    def __init__(self):
        self._xm = None
        self._xla = None

    def _lazy_import(self):
        if self._xm is None:
            import torch_xla
            import torch_xla.core.xla_model as xm
            self._xla = torch_xla
            self._xm = xm

    @property
    def name(self) -> str:
        return "tpu"

    def is_available(self) -> bool:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            _ = xm.xla_device()
            return True
        except Exception:
            return False

    def detect_devices(self, gpu_ids: List[int]) -> List[torch.device]:
        self._lazy_import()
        device = self._xm.xla_device()
        return [device]

    def get_device_name(self, device: torch.device) -> str:
        self._lazy_import()
        try:
            return f"TPU ({self._xm.get_xla_supported_devices()[0]})"
        except Exception:
            return "Google TPU"

    def get_device_memory(self, device: torch.device) -> int:
        self._lazy_import()
        try:
            info = self._xm.get_memory_info(device)
            # Try common key names across torch_xla versions
            for key in ("kb_total", "bytes_limit", "total"):
                if key in info and info[key] > 0:
                    if "kb" in key:
                        return info[key] * 1024
                    return info[key]
            return 0
        except Exception:
            return 0

    def synchronize(self, device: torch.device) -> None:
        self._lazy_import()
        self._xm.mark_step()

    def get_autocast_context(self, device: torch.device, dtype: torch.dtype, enabled: bool):
        if not enabled:
            return nullcontext()
        self._lazy_import()
        try:
            import torch_xla.amp
            return torch_xla.amp.autocast(self._xm.xla_device(), dtype=dtype, enabled=enabled)
        except (ImportError, AttributeError):
            return torch.amp.autocast(device_type="xla", dtype=dtype, enabled=enabled)

    def get_grad_scaler(self, enabled: bool) -> Optional[Any]:
        if not enabled:
            return torch.amp.GradScaler(enabled=False)
        self._lazy_import()
        try:
            import torch_xla.amp
            return torch_xla.amp.GradScaler()
        except (ImportError, AttributeError):
            return torch.amp.GradScaler(enabled=False)

    def supports_compile(self) -> bool:
        return True

    def supports_ddp(self) -> bool:
        return False

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        return {}

    def setup_precision(self, device: torch.device, is_main_process: bool = True) -> None:
        if is_main_process:
            print("TPU: BF16 is the native precision. Using BF16 for best performance.")

    def print_device_info(self, devices: List[torch.device]) -> int:
        self._lazy_import()
        total_mem = 0
        for device in devices:
            name = self.get_device_name(device)
            mem = self.get_device_memory(device)
            total_mem += mem
            mem_gb = mem / (1024 ** 3) if mem > 0 else 0
            print(f"  {name}: {mem_gb:.1f} GB HBM")
        return total_mem

    def try_compile_model(self, model, is_main_process=True):
        try:
            compiled = torch.compile(model, backend="openxla")
            if is_main_process:
                print("Model compiled with openxla backend.")
            return compiled
        except Exception as e:
            if is_main_process:
                print(f"torch.compile(backend='openxla') failed: {e}")
                print("Falling back to eager mode.")
            return model

    def optimizer_step(self, optimizer, scaler=None, use_amp: bool = False) -> None:
        self._lazy_import()
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
            self._xm.mark_step()
        else:
            self._xm.optimizer_step(optimizer, barrier=True)

    @property
    def should_reduce_logging(self) -> bool:
        return True

    def wrap_dataloader(self, dataloader, device):
        """Wrap a DataLoader with MpDeviceLoader for optimal TPU performance."""
        self._lazy_import()
        try:
            import torch_xla.distributed.parallel_loader as pl
            return pl.MpDeviceLoader(dataloader, device)
        except ImportError:
            return dataloader
