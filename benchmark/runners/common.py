"""Shared training utilities for SingleRunner and DDPRunner."""

import torch
import torch.nn as nn

from benchmark.devices.base import DeviceBackend
from benchmark.models.base import BenchModel


def train_step(
    model_spec: BenchModel,
    backend: DeviceBackend,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    use_fp16: bool,
    use_bf16: bool,
    device: torch.device,
) -> torch.Tensor:
    """Execute one training step with proper AMP handling.

    Uses ``model_spec.compute_loss()`` so that non-standard models (DDPM, etc.)
    can override the forward-pass + loss logic.
    """
    if use_bf16:
        with backend.get_autocast_context(device, torch.bfloat16, True):
            loss = model_spec.compute_loss(model, images, labels, criterion, device)
        loss.backward()
        backend.optimizer_step(optimizer)
    elif use_fp16:
        with backend.get_autocast_context(device, torch.float16, True):
            loss = model_spec.compute_loss(model, images, labels, criterion, device)
        scaler.scale(loss).backward()
        backend.optimizer_step(optimizer, scaler=scaler, use_amp=True)
    else:
        loss = model_spec.compute_loss(model, images, labels, criterion, device)
        loss.backward()
        backend.optimizer_step(optimizer, scaler=scaler, use_amp=False)
    return loss


def find_optimal_batch_size(
    backend: DeviceBackend,
    device: torch.device,
    use_fp16: bool,
    use_bf16: bool,
    is_main_process: bool = True,
) -> int:
    """Compute an optimal batch size based on available device memory.

    Falls back to ``4`` for backends that don't report memory.
    """
    if backend.name not in ('cuda', 'npu', 'musa'):
        if is_main_process:
            print(f"Warning: Auto batch size only supports CUDA/NPU/MUSA, current: {backend.name}")
            print(f"Falling back to default batch size: 4")
        return 4

    total_memory = backend.get_device_memory(device)
    if total_memory <= 0:
        return 4

    reserved_memory = 500 * 1024 * 1024  # 500 MB
    available_memory = total_memory - reserved_memory

    if is_main_process:
        print(f"GPU: {backend.get_device_name(device)}")
        print(f"Total Memory: {total_memory / 1024**2:.2f} MB")
        print(f"Available Memory: {available_memory / 1024**2:.2f} MB")

    if use_fp16 or use_bf16:
        model_memory = 164 * 1024 * 1024
        per_sample_memory = 14.4 * 1024 * 1024
    else:
        model_memory = 246 * 1024 * 1024
        per_sample_memory = 27.9 * 1024 * 1024

    memory_for_batch = available_memory - model_memory
    max_batch_size = int(memory_for_batch / per_sample_memory)
    safe_batch_size = int(max_batch_size * 0.99)

    if safe_batch_size >= 32:
        power = int(torch.log2(torch.tensor(float(safe_batch_size))).item())
        safe_batch_size = 2 ** power
    else:
        safe_batch_size = max(4, safe_batch_size)

    if use_fp16 or use_bf16:
        safe_batch_size = max(32, min(safe_batch_size, 4096))
    else:
        safe_batch_size = max(16, min(safe_batch_size, 2048))

    if is_main_process:
        print(f"\nCalculated batch size: {safe_batch_size}")
        print(f"  - Model fixed memory: {model_memory / 1024**2:.0f} MB")
        print(f"  - Per-sample memory: {per_sample_memory / 1024**2:.1f} MB")
        print(f"  - Theoretical max batch size: {max_batch_size}")
        print(f"  - Safe batch size (99%): {safe_batch_size} (power of 2)")
        print(f"  - Estimated total memory: {(model_memory + safe_batch_size * per_sample_memory) / 1024**3:.1f} GB")

    return safe_batch_size
