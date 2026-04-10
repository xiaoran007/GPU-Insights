import json
import os
from dataclasses import asdict

import torch

from benchmark.models import get_model
from benchmark.devices import auto_detect_backend
from benchmark.runners.single_runner import SingleRunner
from benchmark.runners.ddp_runner import DDPRunner


class Bench(object):
    """Orchestrator — configures device, model, and runner.

    The ``device`` parameter replaces the legacy ``huawei`` / ``mthreads`` /
    ``tpu`` boolean flags:

        device="auto"  → probe CUDA → NPU → MUSA → MPS
        device="cuda"  → explicit CUDA
        device="npu"   → explicit NPU (Huawei Ascend)
        device="musa"  → explicit MUSA (Moore Threads)
        device="tpu"   → explicit TPU
        device="mps"   → explicit Apple MPS
    """

    def __init__(
        self,
        method="resnet50",
        device="auto",
        size=1024,
        epochs=10,
        batch_size=0,
        cudnn_benchmark=True,
        data_type="FP32",
        gpu_ids=None,
        auto_batch_size=False,
        use_ddp=False,
        ddp_rank=0,
        ddp_world_size=1,
    ):
        if gpu_ids is None:
            gpu_ids = [0]

        self.device_request = device
        self.data_type = data_type
        self.use_ddp = use_ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.is_main_process = (ddp_rank == 0)
        torch.backends.cudnn.benchmark = cudnn_benchmark

        # --- Device detection via registry ---
        self.device_backend, self.gpu_devices = self._detect_devices(gpu_ids)
        self.cpu_device = torch.device("cpu")

        # --- Build runner ---
        self.runner = self._build_runner(
            method=method, size=size, epochs=epochs,
            batch_size=batch_size, data_type=data_type,
            auto_batch_size=auto_batch_size,
        )

    def start(self):
        result = self.runner.run()
        result.extra.update({
            "dtype": self._get_effective_data_type(),
            "backend": self.device_backend.name if self.device_backend is not None else "none",
            "mode": "ddp" if self.use_ddp else "single",
            "world_size": self.ddp_world_size if self.use_ddp else len([d for d in self.gpu_devices if d is not None]),
        })

        result_path = os.environ.get("GPU_INSIGHTS_RESULT_JSON")
        if result_path and self.is_main_process:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2)

        return result

    # ------------------------------------------------------------------ private

    def _detect_devices(self, gpu_ids):
        """Detect devices through the unified device registry."""
        backend = auto_detect_backend(device=self.device_request)
        if backend is not None:
            devices = backend.detect_devices(gpu_ids)
            return backend, devices
        else:
            if self.is_main_process:
                print("No GPU device found.")
                print("----------------")
            return None, [None]

    def _build_runner(self, method, size, epochs, batch_size, data_type, auto_batch_size):
        """Build the appropriate runner."""

        # --- Resolve model ---
        model_spec = get_model(method)

        # --- Data size calculation ---
        if self.device_backend is not None:
            self.device_backend.print_device_info(
                [d for d in self.gpu_devices if d is not None]
            )

        item_bytes = model_spec.get_data_item_bytes()
        data_size = int(size * 1024 * 1024 / item_bytes)

        if self.is_main_process:
            print(f"Set model [{model_spec.name}], data size: {data_size} samples, "
                  f"memory: {data_size * item_bytes / 1024 / 1024:.2f} MB")

        # --- Resolve batch_size defaults ---
        if batch_size == 0 and not auto_batch_size:
            batch_size = model_spec.get_default_batch_size(data_type)

        # --- Parse data type flags ---
        use_fp16 = (data_type == "FP16")
        use_bf16 = (data_type == "BF16")

        # --- GPU not available → early return with a no-op runner ---
        if self.gpu_devices[0] is None or self.device_backend is None:
            if self.is_main_process:
                print("GPU is not available")
            return _NoOpRunner()

        # --- Choose runner ---
        if self.use_ddp:
            return DDPRunner(
                model_spec=model_spec,
                device_backend=self.device_backend,
                devices=self.gpu_devices,
                data_size=data_size,
                batch_size=batch_size,
                epochs=epochs,
                use_fp16=use_fp16,
                use_bf16=use_bf16,
                auto_batch_size=auto_batch_size,
                ddp_rank=self.ddp_rank,
                ddp_world_size=self.ddp_world_size,
            )
        else:
            return SingleRunner(
                model_spec=model_spec,
                device_backend=self.device_backend,
                devices=self.gpu_devices,
                data_size=data_size,
                batch_size=batch_size,
                epochs=epochs,
                use_fp16=use_fp16,
                use_bf16=use_bf16,
                auto_batch_size=auto_batch_size,
                is_main_process=self.is_main_process,
            )

    def _get_effective_data_type(self):
        if self.data_type == "BF16" and self.device_backend is not None and self.device_backend.name == "cuda":
            main_device = self.gpu_devices[0]
            if main_device is not None and not self.device_backend.supports_bf16(main_device):
                return "FP16"
        return self.data_type


class _NoOpRunner:
    """Placeholder when no GPU is available."""
    def run(self):
        from benchmark.runners.base import BenchResult
        return BenchResult()
