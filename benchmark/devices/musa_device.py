"""Moore Threads MUSA device backend via torch_musa."""

from typing import Any, Dict, List, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
from benchmark.devices.base import DeviceBackend


class MUSADeviceBackend(DeviceBackend):
    """DeviceBackend implementation for Moore Threads GPU via torch_musa."""

    def __init__(self):
        self._musa_imported = False

    def _lazy_import(self):
        if not self._musa_imported:
            import torch_musa
            self._musa_imported = True

    @property
    def name(self) -> str:
        return "musa"

    def is_available(self) -> bool:
        try:
            import torch_musa
            return torch.musa.is_available()
        except Exception:
            return False

    def detect_devices(self, gpu_ids: List[int]) -> List[torch.device]:
        self._lazy_import()
        devices = []
        for gpu_id in gpu_ids:
            name = torch.musa.get_device_name(gpu_id)
            print(f"Found MUSA device {gpu_id}: {name}")
            devices.append(torch.device(f"musa:{gpu_id}"))
        print("----------------")
        return devices

    def get_device_name(self, device: torch.device) -> str:
        self._lazy_import()
        idx = device.index if device.index is not None else 0
        return torch.musa.get_device_name(idx)

    def get_device_memory(self, device: torch.device) -> int:
        self._lazy_import()
        try:
            idx = device.index if device.index is not None else 0
            props = torch.musa.get_device_properties(idx)
            return props.total_memory
        except Exception:
            return 0

    def synchronize(self, device: torch.device) -> None:
        self._lazy_import()
        torch.musa.synchronize(device)

    def get_autocast_context(self, device: torch.device, dtype: torch.dtype, enabled: bool):
        if not enabled:
            return nullcontext()
        try:
            from torch.amp import autocast
            return autocast(device_type="musa", dtype=dtype, enabled=enabled)
        except Exception:
            return nullcontext()

    def get_grad_scaler(self, enabled: bool) -> Optional[Any]:
        if not enabled:
            return None
        try:
            from torch.amp import GradScaler
            return GradScaler(device="musa", enabled=enabled)
        except Exception:
            return None

    def supports_compile(self) -> bool:
        return False

    def supports_ddp(self) -> bool:
        return True

    def supports_bf16(self, device: torch.device) -> bool:
        return False

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        return {}

    def setup_precision(self, device: torch.device, is_main_process: bool = True) -> None:
        pass

    def print_device_info(self, devices: List[torch.device]) -> int:
        self._lazy_import()
        total_memory = 0
        for device in devices:
            name = self.get_device_name(device)
            mem = self.get_device_memory(device)
            total_memory += mem
            mem_gb = mem / (1024 ** 3) if mem > 0 else 0
            print(f"Set MUSA device: {name}, Memory: {mem_gb:.2f} GB")
        print("----------------")
        return total_memory

    def try_compile_model(self, model: nn.Module, is_main_process: bool = True) -> nn.Module:
        if is_main_process:
            print("⚠ torch.compile not supported on MUSA")
            print("  → Continuing with standard (eager) mode...")
        return model

    def get_peak_memory_mb(self, device: torch.device) -> float:
        self._lazy_import()
        try:
            return torch.musa.max_memory_reserved(device) / (1024 * 1024)
        except Exception:
            return 0.0

    def reset_peak_memory(self, device: torch.device) -> None:
        self._lazy_import()
        try:
            torch.musa.reset_peak_memory_stats(device)
            torch.musa.empty_cache()
        except Exception:
            pass

    def release_cached_memory(self, device: torch.device) -> None:
        self._lazy_import()
        try:
            torch.musa.empty_cache()
        except Exception:
            pass
