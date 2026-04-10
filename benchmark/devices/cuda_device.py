from typing import Any, Dict, List, Optional
from contextlib import contextmanager

import torch
import torch.nn as nn

from benchmark.devices.base import DeviceBackend
from helper import getOS, getArch

# PyTorch 2.x detection
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
TORCH_2_PLUS = TORCH_VERSION >= (2, 0)

try:
    from torch.amp import autocast, GradScaler
    _version_flag = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
    except ImportError:
        autocast = None
        GradScaler = None
    _version_flag = False


class CudaDeviceBackend(DeviceBackend):
    """CUDA device backend."""

    @property
    def name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def detect_devices(self, gpu_ids: List[int]) -> List[torch.device]:
        devices = []
        for gpu_id in gpu_ids:
            print(f"Found cuda device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            devices.append(torch.device(f"cuda:{gpu_id}"))
        print("----------------")
        return devices

    def get_device_name(self, device: torch.device) -> str:
        return torch.cuda.get_device_name(device)

    def get_device_memory(self, device: torch.device) -> int:
        return torch.cuda.get_device_properties(device).total_memory

    def synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)

    def get_autocast_context(self, device: torch.device, dtype: torch.dtype, enabled: bool):
        if _version_flag:
            return autocast(device_type="cuda", dtype=dtype, enabled=enabled)
        else:
            return autocast(dtype=dtype, enabled=enabled)

    def get_grad_scaler(self, enabled: bool) -> Optional[Any]:
        if GradScaler is None:
            return None
        if _version_flag:
            return GradScaler(device="cuda", enabled=enabled)
        else:
            return GradScaler(enabled=enabled)

    def supports_compile(self) -> bool:
        if not TORCH_2_PLUS or not hasattr(torch, 'compile'):
            return False
        if getArch() == "aarch64" and getOS() == "linux":
            return False
        return True

    def supports_ddp(self) -> bool:
        return True

    def supports_bf16(self, device: torch.device) -> bool:
        try:
            return torch.cuda.is_bf16_supported(including_emulation=False)
        except Exception:
            return False

    def supports_channels_last(self) -> bool:
        return True

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        if TORCH_2_PLUS:
            return {"fused": True}
        return {}

    def setup_precision(self, device: torch.device, is_main_process: bool = True) -> None:
        if not TORCH_2_PLUS:
            return

        torch.set_float32_matmul_precision('high')

        props = torch.cuda.get_device_properties(device)
        compute_capability = props.major + props.minor / 10.0
        if compute_capability >= 8.0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if is_main_process:
                print(f"✓ Enabled TF32 + high matmul precision for {props.name} (CC {props.major}.{props.minor})")
        else:
            if is_main_process:
                print(f"ℹ TF32 not available for {props.name} (CC {props.major}.{props.minor}, requires 8.0+)")

    def print_device_info(self, devices: List[torch.device]) -> int:
        total_memory = 0
        for device in devices:
            props = torch.cuda.get_device_properties(device)
            print(f"Set cuda device: {props.name}, CUDA architecture: {props.major}.{props.minor}\nFound {props.total_memory / 1024 / 1024:.2f} MB CUDA memory available.")
            total_memory += props.total_memory
        print("----------------")
        return total_memory

    def try_compile_model(self, model: nn.Module, is_main_process: bool = True) -> nn.Module:
        if not TORCH_2_PLUS or not hasattr(torch, 'compile'):
            return model

        if getArch() == "aarch64" and getOS() == "linux":
            if is_main_process:
                print("⚠ torch.compile disabled: aarch64 Linux backend not yet supported")
                print(f"  → Continuing with standard (eager) mode...")
            return model

        try:
            model = torch.compile(model, mode='default')
            if is_main_process:
                print(f"✓ Model compiled with torch.compile (PyTorch {torch.__version__})")
        except Exception as e:
            error_msg = str(e)
            if is_main_process:
                if "C compiler" in error_msg or "CC environment" in error_msg:
                    print(f"⚠ torch.compile disabled: C/C++ compiler not found")
                    print(f"  Install: macOS: xcode-select --install | Linux: sudo apt install build-essential")
                elif "triton" in error_msg.lower():
                    print(f"⚠ torch.compile disabled: Triton compiler issue")
                else:
                    print(f"⚠ torch.compile not supported on cuda: {error_msg[:100]}")
                print(f"  → Continuing with standard (eager) mode...")
        return model

    def get_peak_memory_mb(self, device: torch.device) -> float:
        return torch.cuda.max_memory_reserved(device) / (1024 * 1024)

    def reset_peak_memory(self, device: torch.device) -> None:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
