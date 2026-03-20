"""Backward-compatible wrapper.

Preserves the original ResNet50Bench API and import paths while delegating
to the new modular architecture internally.

Important: calibrate_memory.py imports ``ResNet50`` and ``FakeDataset`` from
this module, so those names MUST remain importable at the top level.
"""
import torch
import torch.nn as nn

# Re-export model classes and dataset for backward compatibility
from benchmark.models.resnet50 import ResNet50, ResNet, Bottleneck
from benchmark.data import FakeDataset
from benchmark.devices import auto_detect_backend
from benchmark.runners.single_runner import SingleRunner
from benchmark.runners.ddp_runner import DDPRunner
from benchmark.models import get_model


class ResNet50Bench(object):
    """Legacy ResNet50Bench — delegates to SingleRunner / DDPRunner."""

    def __init__(self, gpu_device, cpu_device, epochs=5, batch_size=4, lr=0.001, data_size=1000, image_size=(3, 32, 32), num_classes=10, use_fp16=False, use_bf16=False, monitor_performance=False, auto_batch_size=False, use_ddp=False, ddp_rank=0, ddp_world_size=1):
        self.gpu_devices = gpu_device
        self.cpu_device = cpu_device
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_size = data_size
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.auto_batch_size = auto_batch_size
        self.use_ddp = use_ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.is_main_process = (ddp_rank == 0)

    def start(self):
        if self.gpu_devices is None:
            if self.is_main_process:
                print("GPU is not available")
            return

        if self.is_main_process:
            print("DEBUG mode.")

        backend = auto_detect_backend()
        if backend is None:
            if self.is_main_process:
                print("No compatible device backend found.")
            return

        devices = self.gpu_devices if isinstance(self.gpu_devices, list) else [self.gpu_devices]
        model_spec = get_model("resnet50")

        if self.use_ddp:
            runner = DDPRunner(
                model_spec=model_spec,
                device_backend=backend,
                devices=devices,
                data_size=self.data_size,
                batch_size=self.batch_size,
                epochs=self.epochs,
                use_fp16=self.use_fp16,
                use_bf16=self.use_bf16,
                auto_batch_size=self.auto_batch_size,
                ddp_rank=self.ddp_rank,
                ddp_world_size=self.ddp_world_size,
            )
        else:
            runner = SingleRunner(
                model_spec=model_spec,
                device_backend=backend,
                devices=devices,
                data_size=self.data_size,
                batch_size=self.batch_size,
                epochs=self.epochs,
                use_fp16=self.use_fp16,
                use_bf16=self.use_bf16,
                auto_batch_size=self.auto_batch_size,
                is_main_process=self.is_main_process,
            )

        runner.run()


if __name__ == "__main__":
    from torchinfo import summary

    model = ResNet50()
    print(model)

    print("\n----------\n")

    t_batch_size = 1024
    summary(model, input_size=(t_batch_size, 3, 32, 32))

