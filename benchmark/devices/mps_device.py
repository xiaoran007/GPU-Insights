from typing import Any, Dict, List, Optional

import torch

from benchmark.devices.base import DeviceBackend

try:
    from macos_hw_detector import get_gpu_info
except ImportError:
    get_gpu_info = None


class MPSDeviceBackend(DeviceBackend):
    """MPS (Apple Metal) device backend."""

    @property
    def name(self) -> str:
        return "mps"

    def is_available(self) -> bool:
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    def detect_devices(self, gpu_ids: List[int]) -> List[torch.device]:
        device_name = "Apple GPU"
        if get_gpu_info is not None:
            try:
                info = get_gpu_info()
                if info:
                    device_name = info[0].get('name', 'Apple GPU')
            except Exception:
                pass
        print(f"Found mps device: {device_name}")
        print("----------------")
        return [torch.device("mps")]

    def get_device_name(self, device: torch.device) -> str:
        if get_gpu_info is not None:
            try:
                info = get_gpu_info()
                if info:
                    return info[0].get('name', 'Apple GPU')
            except Exception:
                pass
        return "Apple GPU"

    def get_device_memory(self, device: torch.device) -> int:
        # MPS uses shared (unified) memory; not directly queryable via PyTorch
        return 0

    def synchronize(self, device: torch.device) -> None:
        torch.mps.synchronize()

    def get_autocast_context(self, device: torch.device, dtype: torch.dtype, enabled: bool):
        # MPS has limited AMP support; return a no-op context if needed
        try:
            from torch.amp import autocast
            return autocast(device_type="mps", dtype=dtype, enabled=enabled)
        except Exception:
            import contextlib
            return contextlib.nullcontext()

    def get_grad_scaler(self, enabled: bool) -> Optional[Any]:
        # MPS has limited AMP support; return a GradScaler with enabled=False
        # for FP32 mode so the scaler passthrough works correctly
        try:
            from torch.amp import GradScaler
            return GradScaler(device="cpu", enabled=enabled)
        except ImportError:
            from torch.cuda.amp import GradScaler
            return GradScaler(enabled=enabled)

    def supports_compile(self) -> bool:
        return False

    def supports_ddp(self) -> bool:
        return False

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        return {}

    def print_device_info(self, devices: List[torch.device]) -> int:
        print(f"Set MPS device: {self.get_device_name(devices[0])}")
        print("----------------")
        return 0

    def try_compile_model(self, model, is_main_process: bool = True):
        """MPS does not support torch.compile — return model unchanged."""
        if is_main_process:
            print(f"⚠ torch.compile not supported on MPS")
            print(f"  → Continuing with standard (eager) mode...")
        return model
