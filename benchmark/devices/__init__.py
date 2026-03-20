from typing import Dict, List, Optional

import torch

from benchmark.devices.base import DeviceBackend
from benchmark.devices.cuda_device import CudaDeviceBackend
from benchmark.devices.mps_device import MPSDeviceBackend
from benchmark.devices.tpu_device import TPUDeviceBackend


_DEVICE_REGISTRY: Dict[str, DeviceBackend] = {}


def register_device(backend: DeviceBackend) -> None:
    _DEVICE_REGISTRY[backend.name] = backend


def get_device_backend(name: str) -> DeviceBackend:
    if name not in _DEVICE_REGISTRY:
        raise ValueError(f"Unknown device backend: {name}. Available: {list(_DEVICE_REGISTRY.keys())}")
    return _DEVICE_REGISTRY[name]


def list_device_backends():
    return list(_DEVICE_REGISTRY.keys())


def auto_detect_backend(huawei: bool = False, mthreads: bool = False, tpu: bool = False) -> Optional[DeviceBackend]:
    """Auto-detect available device backend, matching original Bench._get_gpu_device logic."""
    if huawei:
        return None
    if mthreads:
        return None

    # Explicit TPU request
    if tpu:
        if "tpu" in _DEVICE_REGISTRY and _DEVICE_REGISTRY["tpu"].is_available():
            return _DEVICE_REGISTRY["tpu"]
        print("Warning: TPU requested but torch_xla is not available.")
        return None

    # Standard detection order: CUDA → MPS → None
    if "cuda" in _DEVICE_REGISTRY and _DEVICE_REGISTRY["cuda"].is_available():
        return _DEVICE_REGISTRY["cuda"]
    if "mps" in _DEVICE_REGISTRY and _DEVICE_REGISTRY["mps"].is_available():
        return _DEVICE_REGISTRY["mps"]
    return None


# Auto-register built-in backends
register_device(CudaDeviceBackend())
register_device(MPSDeviceBackend())

try:
    _tpu_backend = TPUDeviceBackend()
    register_device(_tpu_backend)
except Exception:
    pass
