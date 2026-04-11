from dataclasses import dataclass
from typing import List, Optional

import torch

from benchmark.cli import resolve_data_type
from benchmark.devices import auto_detect_backend
from benchmark.models import get_model, resolve_model_name


@dataclass
class LaunchPlan:
    model: str
    backend: str
    device_ids: List[int]
    use_ddp: bool
    world_size: int
    auto_batch_size: bool
    batch_size_override: Optional[int]
    precisions: List[str]
    data_size_mb: int
    epochs: int


def parse_gpu_ids(raw_gpu_ids: str, backend_name: str) -> List[int]:
    """Resolve the GPU selection string into concrete device ids."""
    if backend_name != "cuda":
        return [0]

    value = raw_gpu_ids.strip().lower()
    if value == "all":
        count = torch.cuda.device_count()
        if count <= 0:
            raise RuntimeError("CUDA was selected but no visible CUDA devices were found.")
        return list(range(count))

    gpu_ids = []
    for chunk in raw_gpu_ids.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        gpu_ids.append(int(chunk))

    if not gpu_ids:
        raise ValueError("At least one CUDA GPU id must be provided, or use 'all'.")

    return gpu_ids


def resolve_precisions(
    model_name: str,
    backend_name: str,
    device_ids: List[int],
    requested_dtype: Optional[str] = None,
) -> List[str]:
    """Choose the precision list for the smart launcher."""
    if requested_dtype is not None:
        return [resolve_data_type(requested_dtype)]

    model_spec = get_model(model_name)
    if not model_spec.supports_amp:
        return ["FP32"]

    if backend_name == "cuda":
        primary_device = torch.device(f"cuda:{device_ids[0]}")
        backend = auto_detect_backend("cuda")
        if backend is not None and backend.supports_bf16(primary_device):
            return ["BF16", "FP32"]
        return ["FP16", "FP32"]

    if backend_name == "tpu":
        return ["BF16", "FP32"]

    if backend_name in ("mps", "npu", "musa"):
        return ["FP16", "FP32"]

    return ["FP32"]


def build_launch_plan(
    model: str,
    size: int = 1024,
    epochs: int = 5,
    device: str = "auto",
    gpu_id: str = "all",
    requested_dtype: Optional[str] = None,
    batch: int = 0,
    no_abs: bool = False,
    single_process: bool = False,
    allow_unavailable: bool = False,
) -> LaunchPlan:
    """Build the smart launcher plan from user intent plus device capabilities."""
    model_name = resolve_model_name(model)
    model_spec = get_model(model_name)

    backend = auto_detect_backend(device=device)
    if backend is None:
        if not allow_unavailable:
            raise RuntimeError(f"No available device backend matched request '{device}'.")

        backend_name = device if device != "auto" else "auto"
        fallback_device_ids = [0]
        precisions = resolve_precisions(
            model_name=model_name,
            backend_name=backend_name,
            device_ids=fallback_device_ids,
            requested_dtype=requested_dtype,
        )
        return LaunchPlan(
            model=model_name,
            backend=backend_name,
            device_ids=fallback_device_ids,
            use_ddp=False,
            world_size=1,
            auto_batch_size=batch <= 0 and not no_abs,
            batch_size_override=batch if batch > 0 else None,
            precisions=precisions,
            data_size_mb=size,
            epochs=epochs,
        )

    device_ids = parse_gpu_ids(gpu_id, backend.name)
    use_ddp = (
        backend.name == "cuda"
        and len(device_ids) > 1
        and model_spec.supports_ddp
        and not single_process
    )
    world_size = len(device_ids) if use_ddp else 1
    auto_batch_size = batch <= 0 and not no_abs
    batch_size_override = batch if batch > 0 else None
    precisions = resolve_precisions(
        model_name=model_name,
        backend_name=backend.name,
        device_ids=device_ids,
        requested_dtype=requested_dtype,
    )

    return LaunchPlan(
        model=model_name,
        backend=backend.name,
        device_ids=device_ids,
        use_ddp=use_ddp,
        world_size=world_size,
        auto_batch_size=auto_batch_size,
        batch_size_override=batch_size_override,
        precisions=precisions,
        data_size_mb=size,
        epochs=epochs,
    )
