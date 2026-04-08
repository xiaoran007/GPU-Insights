from typing import Dict, List, Optional

import torch

from benchmark.devices.base import DeviceBackend
from benchmark.devices.cuda_device import CudaDeviceBackend
from benchmark.devices.mps_device import MPSDeviceBackend
from benchmark.devices.tpu_device import TPUDeviceBackend
from benchmark.devices.npu_device import NPUDeviceBackend
from benchmark.devices.musa_device import MUSADeviceBackend


_DEVICE_REGISTRY: Dict[str, DeviceBackend] = {}


def register_device(backend: DeviceBackend) -> None:
    _DEVICE_REGISTRY[backend.name] = backend


def get_device_backend(name: str) -> DeviceBackend:
    if name not in _DEVICE_REGISTRY:
        raise ValueError(f"Unknown device backend: {name}. Available: {list(_DEVICE_REGISTRY.keys())}")
    return _DEVICE_REGISTRY[name]


def list_device_backends():
    return list(_DEVICE_REGISTRY.keys())


def auto_detect_backend(device: str = "auto") -> Optional[DeviceBackend]:
    """Auto-detect or explicitly select a device backend.

    Args:
        device: One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"npu"``, ``"musa"``, ``"tpu"``.
                ``"auto"`` probes in order: CUDA → NPU → MUSA → MPS → None.
    """
    if device != "auto":
        if device in _DEVICE_REGISTRY and _DEVICE_REGISTRY[device].is_available():
            return _DEVICE_REGISTRY[device]
        print(f"Warning: Requested device '{device}' is not available.")
        return None

    # Auto detection priority: CUDA → NPU → MUSA → MPS
    for name in ("cuda", "npu", "musa", "mps"):
        if name in _DEVICE_REGISTRY and _DEVICE_REGISTRY[name].is_available():
            return _DEVICE_REGISTRY[name]
    return None


# Auto-register built-in backends (try/except for optional dependencies)
register_device(CudaDeviceBackend())
register_device(MPSDeviceBackend())

try:
    register_device(TPUDeviceBackend())
except Exception:
    pass

try:
    register_device(NPUDeviceBackend())
except Exception:
    pass

try:
    register_device(MUSADeviceBackend())
except Exception:
    pass
