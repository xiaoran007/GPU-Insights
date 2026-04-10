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
